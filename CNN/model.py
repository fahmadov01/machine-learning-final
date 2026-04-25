"""EfficientNet-B3 traffic sign classifier and checkpoint utilities"""

import os

import torch
import torch.nn as nn
from transformers import EfficientNetConfig, EfficientNetModel

import config


class TrafficSignClassifier(nn.Module):
    """EfficientNet-B3 backbone with a dropout and linear classification head

    The backbone is pretrained on ImageNet.
    fine tuning happens in two stages
    first the backbone is frozen so the head can catch up
    then end to end with a lower LR on the backbone so its pretrained features adapt without being overwritten
    """

    FEATURE_DIM = 1536   # efficientNet-B3 pooler output size

    def __init__(self, num_classes, dropout=config.DROPOUT, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        if pretrained:
            print(f"Loading pretrained backbone: {config.HF_MODEL_NAME}")
            self.backbone = EfficientNetModel.from_pretrained(config.HF_MODEL_NAME)
        else:
            cfg = EfficientNetConfig.from_pretrained(config.HF_MODEL_NAME)
            self.backbone = EfficientNetModel(cfg)

        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.FEATURE_DIM, num_classes))
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True
        print("Backbone frozen. Training head only.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen. Training end to end.")

    def get_param_groups(self):
        # two parameter groups so the optimizer can apply separate LRs to the pretrained backbone vs the freshly initialized head
        return [
            {"params": self.backbone.parameters(),   "lr": config.LR_BACKBONE},
            {"params": self.classifier.parameters(),"lr": config.LR_HEAD},
        ]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def predict_topk(self, pixel_values, k=config.TOP_K, label_names=None):
        """Return the k highest probability class predictions for the input."""
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self(pixel_values), dim=-1)
            top_p,top_idx = torch.topk(probs, k=k, dim=-1)

        results = []
        for rank, (p, idx) in enumerate(zip(top_p[0].tolist(), top_idx[0].tolist())):
            results.append({
                "rank":  rank + 1,
                "class_id":  idx,
                "class_name": label_names[idx] if label_names else str(idx),
                "confidence": round(p, 4),
            })
        return results


def save_checkpoint(model, optimizer, epoch, val_acc, label_map, path=config.BEST_MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "val_acc":         val_acc,
        "label_map":       label_map,
        "num_classes":     model.num_classes,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint to {path} (epoch {epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = TrafficSignClassifier(num_classes=ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"Loaded checkpoint from {path} " f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return model, ckpt["label_map"], ckpt["epoch"], ckpt["val_acc"]
