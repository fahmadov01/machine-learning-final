"""Evaluate a trained checkpoint on the MTSD val or test split

Outputs
  top 1 and top 5 accuracy printed to stdout
  macro averaged precision recall and F1 printed to stdout
  per class precision recall F1 and support written as CSV to --output-dir

Note that --split test will fail because the official MTSD test set ships without public annotations.
Mapillary holds them for their benchmark server so use --split val as the held out evaluation

Usage
  python evaluate.py --checkpoint checkpoints/best_model.pt
  python evaluate.py --checkpoint checkpoints/best_model.pt --split val
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import config
from dataset import MTSDLocalDataset, MTSDHuggingFaceDataset, build_transforms
from model import load_checkpoint


@torch.no_grad()
def run_inference(model, loader, device):
    """run the whole loader through the model and return prediction arrays"""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_topk_accuracy(model, loader, device, k=5):
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, topk = logits.topk(k, dim=1)
            correct1 += (topk[:, 0] == labels).sum().item()
            correct5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
            total   += labels.size(0)
    return correct1 / total, correct5 / total


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model, label_map, _, _ = load_checkpoint(args.checkpoint, device)

    id_to_label = {v: k for k, v in label_map.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]

    transform = build_transforms("val")   # no augmentation at eval time

    if args.use_hf:
        ds = MTSDHuggingFaceDataset(split=args.split, transform=transform, label_map=label_map)
    else:
        root    = args.mtsd_root
        ann_dir = str(Path(root) / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated")
        if args.split == "train":
            img_dirs = [str(Path(root) / f"mtsd_fully_annotated_images.train.{i}" / "images") for i in range(3)]
        else:
            img_dirs = [str(Path(root) / f"mtsd_fully_annotated_images.{args.split}" / "images")]

        ds = MTSDLocalDataset(
            ann_dir=ann_dir, img_dirs=img_dirs, split=args.split,
            transform=transform, label_map=label_map,
        )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    print(f"\nComputing top-k accuracy on the {args.split} split...")
    top1, top5 = compute_topk_accuracy(model, loader, device, k=5)
    print(f"  Top-1 accuracy: {top1:.4f} ({top1 * 100:.2f}%)")
    print(f"  Top-5 accuracy: {top5:.4f} ({top5 * 100:.2f}%)")

    print("\nRunning full inference pass for per-class metrics...")
    preds, labels = run_inference(model, loader, device)

    report = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    macro = report["macro avg"]
    print("\nClassification report (macro averages):")
    print(f"  Precision : {macro['precision']:.4f}")
    print(f"  Recall    : {macro['recall']:.4f}")
    print(f"  F1        : {macro['f1-score']:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"report_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in class_names:
            if name in report:
                r = report[name]
                writer.writerow([name, r["precision"], r["recall"], r["f1-score"], r["support"]])
    print(f"Per-class report saved to {csv_path}")

    return {"top1": top1, "top5": top5, "macro_f1": macro["f1-score"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate traffic sign classifier")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--mtsd-root", default=config.DATASET_ROOT, help="Directory containing the mtsd_fully_annotated_* folders")
    parser.add_argument("--split",  default="val", choices=["val", "test"])
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--batch-size",type=int, default=64)
    parser.add_argument("--output-dir", default="./eval_results")
    args = parser.parse_args()
    evaluate(args)
