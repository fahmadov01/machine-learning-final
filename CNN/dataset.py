"""Dataset classes for the Mapillary Traffic Sign Dataset (MTSD)

Two backends are supported

  MTSDLocalDataset
    Reads the official MTSD download
    Each sample is a cropped bounding box patch around one sign
    The annotation index is built once by walking every images JSON and pickled to disk so subsequent runs load in under a second

  MTSDHuggingFaceDataset
    Thin wrapper around the sparshgarg57/mapillary_traffic_signs HuggingFace mirror which provides precropped sign patches
"""

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


# ImageNet normalization stats used by every torchvision ImageNet model
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(split):
    """Return the torchvision transform pipeline for the given split

    Training uses strong augmentation like RandAugment random crop rotation and color jitter to help the model generalize across a long tailed label distribution
    val and test just resize and normalize
    """
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train" and config.AUGMENT_TRAIN:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32,config.IMG_SIZE+32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandAugment(num_ops=config.RAND_AUG_N, magnitude=config.RAND_AUG_M),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


class MTSDLocalDataset(Dataset):
    """sign patch dataset backed by a local MTSD download"""

    def __init__(
        self,
        ann_dir,
        img_dirs,
        split="train",
        transform=None,
        label_map=None,
        min_bbox_area=400,
        skip_occluded=config.SKIP_OCCLUDED,
        skip_ambiguous=config.SKIP_AMBIGUOUS,
        skip_out_of_frame=config.SKIP_OUT_OF_FRAME,
        skip_other_sign=config.SKIP_OTHER_SIGN,
        cache_dir=config.CACHE_DIR,
    ):
        assert split in ("train","val","test"), f"Unknown split: {split}"

        self.ann_dir   = Path(ann_dir)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.split= split
        self.transform = transform or build_transforms(split)

        splits_file = self.ann_dir / "splits" / f"{split}.txt"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {splits_file}\n"
                "Expected the mtsd_v2_fully_annotated/ directory from the "
                "official MTSD download. Set USE_HF_DATASET=True in config.py "
                "to use the HuggingFace mirror instead."
            )

        # Encode every filter setting in the cache filename so changing a flag forces a clean rebuild instead of silently reusing stale data
        cache_tag = (
            f"{split}_occ{int(skip_occluded)}_amb{int(skip_ambiguous)}"
            f"_oof{int(skip_out_of_frame)}_oth{int(skip_other_sign)}"
            f"_area{min_bbox_area}"
        )
        cache_path = Path(cache_dir) / f"index_{cache_tag}.pkl"

        if cache_path.exists():
            print(f"Loading cached annotation index: {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.samples = cached["samples"]
            raw_label_map = cached["label_map"]
        else:
            print(f"Building annotation index for '{split}' split...")
            self.samples, raw_label_map = self._build_index(
                splits_file,
                min_bbox_area,
                skip_occluded,
                skip_ambiguous,
                skip_out_of_frame,
                skip_other_sign,
            )
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"samples": self.samples, "label_map": raw_label_map}, f)
            print(f"Index cached to {cache_path}")

        # val and test reuse the train label map so label IDs line up across splits
        # Samples with labels outside the provided map get dropped
        if label_map is not None:
            self.label_map = label_map
            self.samples = [s for s in self.samples if s["label"] in self.label_map]
        else:
            self.label_map = raw_label_map

        self.num_classes = len(self.label_map)
        for s in self.samples:
            s["label_id"] = self.label_map[s["label"]]

        print(f"[{split}] {len(self.samples):,} sign patches | "
              f"{self.num_classes} classes")

    def _build_index(self, splits_file, min_bbox_area,skip_occluded, skip_ambiguous,skip_out_of_frame, skip_other_sign):
        # Scan all image directories once to build a key to path lookup so we dont iterate the filesystem for every key in the split file.
        img_lookup= {}
        for img_dir in self.img_dirs:
            if not img_dir.exists():
                continue
            for p in img_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    img_lookup[p.stem] = str(p)

        ann_dir = self.ann_dir / "annotations"
        image_keys = [k.strip() for k in splits_file.read_text().splitlines()
                      if k.strip()]

        samples=[]
        labels_seen = set()
        skipped_no_image = skipped_filter = 0

        for key in image_keys:
            img_path =img_lookup.get(key)
            if img_path is None:
                skipped_no_image += 1
                continue

            ann_path = ann_dir/ f"{key}.json"
            if not ann_path.exists():
                skipped_no_image += 1
                continue

            with open(ann_path) as f:
                ann = json.load(f)

            for obj in ann.get("objects", []):
                label = obj.get("label", "")

                if skip_other_sign and label == "other-sign":
                    skipped_filter += 1
                    continue

                props = obj.get("properties", {})
                if skip_occluded and props.get("occluded",False):
                    skipped_filter += 1
                    continue
                if skip_ambiguous and props.get("ambiguous", False):
                    skipped_filter += 1
                    continue
                if skip_out_of_frame and props.get("out-of-frame", False):
                    skipped_filter += 1
                    continue

                bbox = obj.get("bbox", {})
                # panorama signs that wrap across the image boundary would need two crops stitched together
                # they are a tiny minority so we drop them rather than complicate the loader
                if "cross_boundary" in bbox:
                    skipped_filter += 1
                    continue

                xmin = bbox.get("xmin", 0)
                ymin = bbox.get("ymin", 0)
                xmax = bbox.get("xmax", 0)
                ymax = bbox.get("ymax", 0)
                if (xmax - xmin) * (ymax - ymin) < min_bbox_area:
                    skipped_filter += 1
                    continue

                samples.append({
                    "img_path": img_path,
                    "bbox":     (xmin, ymin, xmax, ymax),
                    "label":    label,
                    "label_id": 0,   # filled in after label_map is finalized
                })
                labels_seen.add(label)

        label_map = {lbl: i for i, lbl in enumerate(sorted(labels_seen))}
        print(f"  Scanned {len(image_keys):,} images | "
              f"kept {len(samples):,} signs | "
              f"skipped {skipped_no_image:,} (no image/ann) + "
              f"{skipped_filter:,} (filtered)")
        return samples, label_map

    @staticmethod
    def _crop_with_padding(img, bbox, padding):
        W, H = img.size
        x1, y1, x2, y2 = bbox
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        return img.crop((
            max(0, int(x1 - pw)),
            max(0, int(y1 - ph)),
            min(W, int(x2 + pw)),
            min(H, int(y2 + ph)),
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img = self._crop_with_padding(img, s["bbox"], config.CROP_PADDING)
        if self.transform:
            img = self.transform(img)
        return img, s["label_id"]


class MTSDHuggingFaceDataset(Dataset):
    """HuggingFace mirror backend where images arrive precropped"""

    def __init__(self, split="train", transform=None, label_map=None):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  # required for HF mirror")

        hf_split = {"train": "train", "val":"validation", "test":"test"}.get(split,split)
        print(f"Loading HuggingFace dataset {config.HF_DATASET}/{hf_split}...")
        raw = load_dataset(config.HF_DATASET,  split=hf_split, trust_remote_code=True)

        self.transform = transform or build_transforms(split)

        all_labels = sorted(set(raw["label"]))
        self.label_map = label_map or {lbl: i for i, lbl in enumerate(all_labels)}
        self.num_classes = len(self.label_map)
        self.data = [row for row in raw if row["label"] in self.label_map]

        print(f"[{split}] HF dataset: {len(self.data):,} samples, "
              f"{self.num_classes} classes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = row["image"].convert("RGB")
        label = self.label_map[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_datasets(root=config.DATASET_ROOT, use_hf=config.USE_HF_DATASET):
    """Build train_ds val_ds num_classes and label_map

    falls back to the HuggingFace mirror automatically if the local MTSD download cannot be found at root
    """
    if not use_hf:
        ann_dir = (Path(root) / "mtsd_fully_annotated_annotation"/"mtsd_v2_fully_annotated")
        if not (ann_dir / "splits" / "train.txt").exists():
            print(f"Local MTSD not found at {root}. " f"Falling back to HuggingFace mirror.")
            use_hf = True

    if use_hf:
        train_ds = MTSDHuggingFaceDataset(split="train")
        val_ds   = MTSDHuggingFaceDataset(split="val", label_map=train_ds.label_map)
    else:
        ann_dir = str(Path(root) / "mtsd_fully_annotated_annotation" / "mtsd_v2_fully_annotated")
        train_img_dirs = [str(Path(root) / f"mtsd_fully_annotated_images.train.{i}" / "images") for i in range(3)]
        val_img_dirs = [str(Path(root) /"mtsd_fully_annotated_images.val" / "images")]

        train_ds = MTSDLocalDataset(ann_dir=ann_dir, img_dirs=train_img_dirs, split="train")
        val_ds = MTSDLocalDataset(ann_dir=ann_dir, img_dirs=val_img_dirs, split="val",label_map=train_ds.label_map)

    return train_ds, val_ds, train_ds.num_classes, train_ds.label_map


class MixupCollate:
    """collate function that applies mixup to a batch

    Implemented as a class rather than a closure
    With alpha=0 it degrades to the default collate
    """

    def __init__(self, alpha=config.MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        import torch
        from torch.utils.data.dataloader import default_collate

        images, labels = default_collate(batch)
        if self.alpha <= 0 or not torch.is_grad_enabled():
            return images, labels

        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))
        mixed = lam * images + (1 - lam) * images[idx]
        return mixed, (labels, labels[idx], lam)


def mixup_collate(alpha=config.MIXUP_ALPHA):
    return MixupCollate(alpha=alpha)
