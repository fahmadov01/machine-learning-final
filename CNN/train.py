"""Train the traffic sign classifier on MTSD

Training runs in two stages

  Stage 1 is the first 5 epochs with the backbone frozen and the head training from scratch.
  Fast convergence without disturbing the pretrained backbone features

  Stage 2 is the remaining epochs where everything unfreezes and fine tunes end to end
  the backbone gets a lower learning rate than the head so its pretrained representations dont get overwritten.
  early stopping on val accuracy

Usage
  python train.py
  python train.py --mtsd-root /path/to/mtsd
  python train.py --use-hf --epochs 40 --batch-size 64
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import get_datasets, mixup_collate
from model import TrafficSignClassifier, save_checkpoint


def seed_everything(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN auto tune conv kernels for the fixed input shape which is the common case speed win
    # incompatible with deterministic mode so we pick one.
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


class MixupCrossEntropy(nn.Module):
    """cross entropy loss that handles both plain and mixup mixed targets"""

    def __init__(self, label_smoothing=config.LABEL_SMOOTHING):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, target):
        if isinstance(target, tuple):
            y_a, y_b, lam = target
            return lam * self.ce(logits, y_a) + (1-lam)* self.ce(logits, y_b)
        return self.ce(logits, target)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val  = val
        self.sum  += val * n
        self.count += n
        self.avg  = self.sum / self.count


def accuracy(logits, labels, topk=(1, 5)):
    with torch.no_grad():
        maxk  = max(topk)
        batch= labels.size(0)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct= pred.eq(labels.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum() / batch for k in topk]


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer, scaler):
    model.train()
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()
    t0 = time.perf_counter()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # mixup passes labels as a tuple y_a y_b lam
        # keep y_a around for accuracy since top 1 on a blended label is not a meaningful training signal
        if isinstance(labels, (list, tuple)) and len(labels) == 3:
            labels =(labels[0].to(device, non_blocking=True),labels[1].to(device, non_blocking=True), labels[2])
            plain_labels = labels[0]
        else:
            labels = labels.to(device, non_blocking=True)
            plain_labels=labels

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss= criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        top1, top5 = accuracy(logits, plain_labels, topk=(1,5))
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        acc1_m.update(top1.item(), bs)
        acc5_m.update(top5.item(), bs)

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch:03d} | step {step:04d}/{len(loader):04d} | "
                f"loss={loss_m.avg:.4f} | top1={acc1_m.avg:.4f} | "
                f"top5={acc5_m.avg:.4f} | {elapsed:.1f}s"
            )

    global_step = epoch * len(loader)
    writer.add_scalar("train/loss",loss_m.avg, global_step)
    writer.add_scalar("train/top1", acc1_m.avg, global_step)
    writer.add_scalar("train/top5", acc5_m.avg,global_step)
    return loss_m.avg, acc1_m.avg


@torch.no_grad()
def validate(model, loader, criterion,device, epoch, writer):
    model.eval()
    loss_m = AverageMeter()
    acc1_m = AverageMeter()
    acc5_m = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        bs = images.size(0)
        loss_m.update(loss.item(), bs)
        acc1_m.update(top1.item(), bs)
        acc5_m.update(top5.item(), bs)

    global_step = epoch *len(loader)
    writer.add_scalar("val/loss", loss_m.avg, global_step)
    writer.add_scalar("val/top1", acc1_m.avg, global_step)
    writer.add_scalar("val/top5",acc5_m.avg, global_step)

    print(f"  Val  epoch {epoch:03d} | loss={loss_m.avg:.4f} | "f"top1={acc1_m.avg:.4f} | top5={acc5_m.avg:.4f}")
    return loss_m.avg, acc1_m.avg


def build_scheduler(optimizer, total_epochs, warmup_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1,end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=config.MIN_LR)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],milestones=[warmup_epochs])


def train(args):
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, num_classes, label_map = get_datasets(root=args.mtsd_root, use_hf=args.use_hf)

    collate_fn = mixup_collate(alpha=config.MIXUP_ALPHA) if config.MIXUP_ALPHA > 0 else None
    pin        = config.PIN_MEMORY and device.type =="cuda"
    persistent = config.NUM_WORKERS >0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers= config.NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size =args.batch_size * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    model = TrafficSignClassifier(num_classes=num_classes).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,} total | "f"{params['trainable']:,} trainable")

    criterion = MixupCrossEntropy(label_smoothing=config.LABEL_SMOOTHING)
    scaler  = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # stage 1 freezes the backbone and trains only the newly initialized head
    stage1_epochs = min(5, args.epochs)
    print(f"\n{'=' * 60}")
    print(f"  Stage 1: head-only training ({stage1_epochs} epochs)")
    print(f"{'=' *60}\n")

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, stage1_epochs, warmup_epochs=1)

    best_val_acc = 0.0

    for epoch in range(1, stage1_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, scaler)
        _, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,optimizer, epoch, val_acc, label_map)

    # Stage 2 unfreezes everything and fine tunes end to end.
    remaining = args.epochs - stage1_epochs
    if remaining <= 0:
        print("Skipping Stage 2 (not enough epochs).")
    else:
        print(f"\n{'=' * 60}")
        print(f"  Stage 2: end-to-end fine-tuning ({remaining} epochs)")
        print(f"{'=' * 60}\n")

        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(model.get_param_groups(), weight_decay=config.WEIGHT_DECAY)
        scheduler = build_scheduler( optimizer, remaining, warmup_epochs=config.WARMUP_EPOCHS)
        no_improve = 0

        for epoch in range(stage1_epochs + 1, args.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, scaler)
            _, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
            scheduler.step()

            writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("lr/head", optimizer.param_groups[1]["lr"], epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, label_map)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    writer.close()
    print(f"\nTraining complete. Best val top-1: {best_val_acc:.4f}")
    print(f"Best checkpoint: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train traffic sign classifier")
    parser.add_argument("--mtsd-root", default=config.DATASET_ROOT,help="Directory containing the mtsd_fully_annotated_* folders")
    parser.add_argument("--use-hf", action="store_true", help="Use the HuggingFace dataset mirror")
    parser.add_argument("--epochs",     type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.BATCH_SIZE = args.batch_size
    train(args)
