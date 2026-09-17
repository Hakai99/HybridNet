"""
HybridNet - utils/dataloader.py
================================
YOLO-Format Dataset & DataLoader

Reads:
    data.yaml         → class names, paths
    images/train/     → image files
    labels/train/     → .txt YOLO annotation files

YOLO annotation format (per line):
    class_id cx cy w h   (all normalized 0-1)

Author: Your Name
"""

import os
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class YOLODataset(Dataset):
    """
    Loads images and YOLO-format annotations.

    Folder structure expected:
        dataset/
        ├── images/
        │   ├── train/   ← images here
        │   └── val/
        └── labels/
            ├── train/   ← .txt files here (same name as image)
            └── val/

    Each .txt file contains one object per line:
        class_id cx cy w h
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        """
        Args:
            img_dir   (str): path to images folder (e.g. dataset/images/train)
            label_dir (str): path to labels folder (e.g. dataset/labels/train)
            img_size  (int): resize all images to this square size
            augment  (bool): apply data augmentation (training only)
        """
        self.img_size   = img_size
        self.augment    = augment
        self.img_dir    = Path(img_dir)
        self.label_dir  = Path(label_dir)

        # Collect all valid image paths
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        assert len(self.img_paths) > 0, f"No images found in {img_dir}"
        print(f"Found {len(self.img_paths)} images in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # ── Load Image ──────────────────────────────────
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Load Labels ─────────────────────────────────
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.read().strip().splitlines():
                    vals = list(map(float, line.strip().split()))
                    if len(vals) == 5:
                        labels.append(vals)   # [class, cx, cy, w, h]

        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        # ── Augmentation ────────────────────────────────
        if self.augment:
            img, labels = self._augment(img, labels)

        # ── Resize ──────────────────────────────────────
        img, ratio, pad = self._letterbox(img, self.img_size)

        # Adjust labels for letterbox padding
        if labels.shape[0] > 0:
            labels[:, 1] = ratio * labels[:, 1] * img.shape[1] / self.img_size + pad[0] / self.img_size
            labels[:, 2] = ratio * labels[:, 2] * img.shape[0] / self.img_size + pad[1] / self.img_size
            labels[:, 3] = ratio * labels[:, 3]
            labels[:, 4] = ratio * labels[:, 4]

        # ── To Tensor ───────────────────────────────────
        img = img.astype(np.float32) / 255.0         # normalize to [0,1]
        img = torch.from_numpy(img).permute(2, 0, 1) # HWC → CHW

        labels_tensor = torch.from_numpy(labels)

        return img, labels_tensor, str(img_path)

    def _letterbox(self, img, size):
        """Resize image with padding to maintain aspect ratio"""
        h, w = img.shape[:2]
        ratio = min(size / h, size / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = cv2.resize(img, (new_w, new_h))

        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2

        img = cv2.copyMakeBorder(
            img, pad_h, size - new_h - pad_h,
            pad_w, size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img, ratio, (pad_w, pad_h)

    def _augment(self, img, labels):
        """Basic augmentations for training"""
        # Horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            if labels.shape[0] > 0:
                labels[:, 1] = 1 - labels[:, 1]   # flip cx

        # HSV color jitter
        img = self._hsv_jitter(img, hgain=0.015, sgain=0.7, vgain=0.4)

        return img, labels

    def _hsv_jitter(self, img, hgain, sgain, vgain):
        """Random HSV color augmentation"""
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv.astype(np.float32))
        h = (h * r[0]) % 180
        s = np.clip(s * r[1], 0, 255)
        v = np.clip(v * r[2], 0, 255)
        img = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


# ─────────────────────────────────────────────
# Collate Function (handles variable labels)
# ─────────────────────────────────────────────

def collate_fn(batch):
    """
    Custom collate — images stack normally,
    labels get an image index prepended and concatenated.

    Output labels shape: (total_objects, 6)
    Format: [img_idx, class, cx, cy, w, h]
    """
    imgs, labels, paths = zip(*batch)

    # Stack images: list of (3,H,W) → (B,3,H,W)
    imgs = torch.stack(imgs, dim=0)

    # Add image index to labels
    all_labels = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0] > 0:
            idx = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
            all_labels.append(torch.cat([idx, lbl], dim=1))  # (N, 6)

    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)  # (total, 6)
    else:
        all_labels = torch.zeros((0, 6))

    return imgs, all_labels, paths


# ─────────────────────────────────────────────
# data.yaml Parser
# ─────────────────────────────────────────────

def parse_data_yaml(yaml_path):
    """
    Parse YOLO-format data.yaml

    Expected format:
        train: images/train
        val:   images/val
        nc:    80
        names: ['class1', 'class2', ...]

    Returns:
        dict with keys: train, val, nc, names
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Resolve paths relative to yaml location
    yaml_dir = Path(yaml_path).parent
    for key in ['train', 'val']:
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str(yaml_dir / data[key])

    assert 'nc' in data,    "data.yaml must contain 'nc' (number of classes)"
    assert 'names' in data, "data.yaml must contain 'names' (class list)"
    assert len(data['names']) == data['nc'], \
        f"nc={data['nc']} but {len(data['names'])} names found"

    return data


# ─────────────────────────────────────────────
# DataLoader Builder
# ─────────────────────────────────────────────

def build_dataloader(yaml_path, split='train', img_size=640, batch_size=16, workers=4):
    """
    Build DataLoader from data.yaml.

    Args:
        yaml_path  (str): path to data.yaml
        split      (str): 'train' or 'val'
        img_size   (int): input image size
        batch_size (int): batch size
        workers    (int): number of dataloader workers

    Returns:
        loader:    DataLoader
        num_classes: int
        class_names: list of str
    """
    data = parse_data_yaml(yaml_path)

    img_dir   = data[split]
    label_dir = img_dir.replace('images', 'labels')
    augment   = (split == 'train')

    dataset = YOLODataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )

    return loader, data['nc'], data['names']


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils/dataloader.py path/to/data.yaml")
        sys.exit(0)

    loader, nc, names = build_dataloader(sys.argv[1], split='train', batch_size=4)
    print(f"Classes: {nc} → {names}")
    print(f"Batches: {len(loader)}")

    imgs, labels, paths = next(iter(loader))
    print(f"Image batch: {imgs.shape}")
    print(f"Labels:      {labels.shape}")
    print(f"Sample path: {paths[0]}")
