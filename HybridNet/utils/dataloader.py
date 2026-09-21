
Dataloader · PY
"""
HybridNet - utils/dataloader.py
================================
YOLO-Format Dataset & DataLoader
Colab-compatible version (num_workers=0, pin_memory=False)
 
Author: Dev Kr Lahkar
"""
 
import os
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
 
 
class YOLODataset(Dataset):
    """
    Loads images and YOLO-format annotations.
 
    Folder structure:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
 
    Each .txt file: class_id cx cy w h (normalized)
    """
 
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
 
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_size  = img_size
        self.augment   = augment
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
 
        self.img_paths = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])
 
        assert len(self.img_paths) > 0, "No images found in " + str(img_dir)
        print("Found " + str(len(self.img_paths)) + " images in " + str(img_dir))
 
    def __len__(self):
        return len(self.img_paths)
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
 
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        # Save original size BEFORE letterbox ← FIX
        orig_h, orig_w = img.shape[:2]
 
        # Load labels
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            try:
                with open(label_path) as f:
                    for line in f.read().strip().splitlines():
                        vals = list(map(float, line.strip().split()))
                        if len(vals) == 5:
                            labels.append(vals)
            except Exception:
                pass
 
        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
 
        # Augmentation (before letterbox)
        if self.augment:
            img, labels = self._augment(img, labels)
 
        # Resize with letterbox
        img, ratio, pad = self._letterbox(img, self.img_size)
 
        # Fix label coords for letterbox ← FIX
        if labels.shape[0] > 0:
            # cx, cy adjusted for padding and ratio
            labels[:, 1] = labels[:, 1] * ratio * orig_w / self.img_size + pad[0] / self.img_size
            labels[:, 2] = labels[:, 2] * ratio * orig_h / self.img_size + pad[1] / self.img_size
            # w, h adjusted for ratio
            labels[:, 3] = labels[:, 3] * ratio * orig_w / self.img_size
            labels[:, 4] = labels[:, 4] * ratio * orig_h / self.img_size
            # Clamp all to valid range
            labels[:, 1:] = labels[:, 1:].clip(0, 1)
 
        # To tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        labels_tensor = torch.from_numpy(labels)
 
        return img, labels_tensor, str(img_path)
 
    def _letterbox(self, img, size):
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
        # Horizontal flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            if labels.shape[0] > 0:
                labels[:, 1] = 1 - labels[:, 1]
        # HSV color jitter
        img = self._hsv_jitter(img, hgain=0.015, sgain=0.7, vgain=0.4)
        return img, labels
 
    def _hsv_jitter(self, img, hgain, sgain, vgain):
        try:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv.astype(np.float32))
            h = (h * r[0]) % 180
            s = np.clip(s * r[1], 0, 255)
            v = np.clip(v * r[2], 0, 255)
            img = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        except Exception:
            return img
 
 
def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    all_labels = []
    for i, lbl in enumerate(labels):
        if lbl.shape[0] > 0:
            idx = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
            all_labels.append(torch.cat([idx, lbl], dim=1))
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
    else:
        all_labels = torch.zeros((0, 6))
    return imgs, all_labels, paths
 
 
def parse_data_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
 
    yaml_dir = Path(yaml_path).parent
    for key in ['train', 'val']:
        if key in data and not Path(str(data[key])).is_absolute():
            data[key] = str(yaml_dir / data[key])
 
    assert 'nc' in data,    "data.yaml must contain 'nc'"
    assert 'names' in data, "data.yaml must contain 'names'"
 
    # Handle both list and dict format for names
    if isinstance(data['names'], dict):
        data['names'] = [data['names'][i] for i in sorted(data['names'].keys())]
 
    return data
 
 
def build_dataloader(yaml_path, split='train', img_size=640, batch_size=16, workers=0):
    """
    Build DataLoader from data.yaml.
    workers=0 is default for Colab compatibility.
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
 
    # Colab-safe DataLoader settings
    # num_workers=0  → no multiprocessing (prevents freeze)
    # pin_memory=False → no CUDA pinning (prevents crash)
    # persistent_workers=False → safe for num_workers=0
    # drop_last=False → use all images
    # timeout=0 → no timeout issues
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=False,
        timeout=0,
    )
 
    return loader, data['nc'], data['names']
 
 
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils/dataloader.py path/to/data.yaml")
        sys.exit(0)
    loader, nc, names = build_dataloader(sys.argv[1], split='train', batch_size=4)
    print("Classes: " + str(nc))
    print("Batches: " + str(len(loader)))
    imgs, labels, paths = next(iter(loader))
    print("Image batch: " + str(imgs.shape))
    print("Labels: " + str(labels.shape))
 
