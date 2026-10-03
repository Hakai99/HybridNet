"""
HybridNet - head.py
===================
AnchorFreeHead — Anchor-Free Detection Head

Predicts on 3 scales (small / medium / large objects):
    - Box:        (x, y, w, h) normalized coordinates
    - Class:      sigmoid per class (multi-label friendly)
    - Centerness: how close to object center (filters bad preds)

No anchors needed — clean, modern, research-friendly.

Author: Dev Kr Lahkar
"""

import torch
import torch.nn as nn


class ConvBNReLU6(nn.Module):
    """Conv → BN → ReLU6"""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Single Scale Head
# ─────────────────────────────────────────────

class ScaleHead(nn.Module):
    """
    Detection head for ONE scale (P3 or P4 or P5).

    Architecture:
        Shared Conv Tower (3 layers) →
            ├── Box Branch    → 4 values (x,y,w,h)
            ├── Class Branch  → num_classes values
            └── Center Branch → 1 value (centerness)
    """

    def __init__(self, in_ch, num_classes, mid_ch=128):
        super().__init__()

        # Shared feature tower (same weights used for box & class)
        self.tower = nn.Sequential(
            ConvBNReLU6(in_ch, mid_ch, kernel=3),
            ConvBNReLU6(mid_ch, mid_ch, kernel=3),
            ConvBNReLU6(mid_ch, mid_ch, kernel=3),
        )

        # Box branch: predict (x, y, w, h)
        self.box_head = nn.Conv2d(mid_ch, 4, kernel_size=1)

        # Class branch: predict class probabilities
        self.cls_head = nn.Conv2d(mid_ch, num_classes, kernel_size=1)

        # Centerness branch: predict how centered the point is
        self.ctr_head = nn.Conv2d(mid_ch, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize heads with small weights for stable training start"""
        for m in [self.box_head, self.cls_head, self.ctr_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.tower(x)

        box = self.box_head(feat)       # (B, 4, H, W)
        cls = self.cls_head(feat)       # (B, num_classes, H, W)
        ctr = self.ctr_head(feat)       # (B, 1, H, W)

        return box, cls, ctr


# ─────────────────────────────────────────────
# Full Multi-Scale Head
# ─────────────────────────────────────────────

class AnchorFreeHead(nn.Module):
    """
    AnchorFreeHead — Multi-Scale Anchor-Free Detection Head
    ========================================================
    Runs ScaleHead on all 3 feature scales independently.
    
    Each scale is responsible for different object sizes:
        out3 (80x80) → small objects
        out4 (40x40) → medium objects
        out5 (20x20) → large objects

    Returns raw predictions (no sigmoid/softmax applied here).
    Activations applied inside the loss function for numerical stability.
    """

    def __init__(self, in_ch=128, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # One head per scale (shared architecture, separate weights)
        self.head_small  = ScaleHead(in_ch, num_classes)   # P3 → small objects
        self.head_medium = ScaleHead(in_ch, num_classes)   # P4 → medium objects
        self.head_large  = ScaleHead(in_ch, num_classes)   # P5 → large objects

    def forward(self, out3, out4, out5):
        """
        Args:
            out3: (B, 128, 80, 80) — from neck small scale
            out4: (B, 128, 40, 40) — from neck medium scale
            out5: (B, 128, 20, 20) — from neck large scale

        Returns:
            List of (box, cls, ctr) tuples for each scale
            box: (B, 4, H, W)
            cls: (B, num_classes, H, W)
            ctr: (B, 1, H, W)
        """
        pred_small  = self.head_small(out3)    # small objects
        pred_medium = self.head_medium(out4)   # medium objects
        pred_large  = self.head_large(out5)    # large objects

        return [pred_small, pred_medium, pred_large]

    def decode_predictions(self, predictions, conf_thresh=0.25, strides=(8, 16, 32), input_size=640):
        """
        Decode raw head outputs into (boxes, scores, classes).
        Used during inference / detection.

        IMPORTANT: box_pred[...,0:2] are offsets WITHIN a grid cell (sigmoid, 0-1),
        added to the cell's own (grid_x, grid_y) position, then divided by the
        grid size to land back in normalized [0,1] image coordinates.
        box_pred[...,2:4] are log-scale sizes relative to the stride, decoded
        with exp() so they stay positive and can represent both small and large
        boxes without the sigmoid's [0,1] ceiling.

        Args:
            predictions: list of (box, cls, ctr) per scale
            conf_thresh: minimum confidence to keep a detection
            strides:     stride of each scale, matching predictions order (P3,P4,P5)
            input_size:  network input resolution used during training

        Returns:
            all_boxes:   (N, 4) — cx,cy,w,h normalized [0,1]
            all_scores:  (N,)   — confidence scores
            all_classes: (N,)   — class indices
        """
        all_boxes, all_scores, all_classes = [], [], []

        for (box_pred, cls_pred, ctr_pred), stride in zip(predictions, strides):
            B, _, H, W = box_pred.shape
            device = box_pred.device

            # Build grid of cell coordinates (same for every image in the batch)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )  # (H, W) each

            # Center offset within the cell: sigmoid keeps it in [0,1] of one cell
            cx = (grid_x.unsqueeze(0) + torch.sigmoid(box_pred[:, 0])) / W   # (B,H,W)
            cy = (grid_y.unsqueeze(0) + torch.sigmoid(box_pred[:, 1])) / H   # (B,H,W)

            # Width/height: exp() keeps positive, scaled by stride so scale
            # differences between P3/P4/P5 are represented consistently
            w = torch.exp(box_pred[:, 2].clamp(max=6)) * stride / input_size
            h = torch.exp(box_pred[:, 3].clamp(max=6)) * stride / input_size

            box_decoded = torch.stack([cx, cy, w, h], dim=-1)  # (B,H,W,4)

            cls_pred = torch.sigmoid(cls_pred)   # class probabilities
            ctr_pred = torch.sigmoid(ctr_pred)   # centerness score

            # Centerness-weighted class score (filters bad predictions)
            # score = class_prob * centerness
            scores, class_idx = (cls_pred * ctr_pred).max(dim=1, keepdim=True)

            # Flatten spatial dims
            scores      = scores.view(B, -1)               # (B, H*W)
            class_idx   = class_idx.view(B, -1)             # (B, H*W)
            box_decoded = box_decoded.reshape(B, -1, 4)     # (B, H*W, 4)

            # Filter by threshold
            mask = scores[0] > conf_thresh
            all_boxes.append(box_decoded[0][mask])
            all_scores.append(scores[0][mask])
            all_classes.append(class_idx[0][mask].float())

        if len(all_boxes) == 0:
            return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)

        return (
            torch.cat(all_boxes, dim=0),
            torch.cat(all_scores, dim=0),
            torch.cat(all_classes, dim=0)
        )


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    head = AnchorFreeHead(in_ch=128, num_classes=80)

    out3 = torch.randn(1, 128, 80, 80)
    out4 = torch.randn(1, 128, 40, 40)
    out5 = torch.randn(1, 128, 20, 20)

    preds = head(out3, out4, out5)
    for i, (box, cls, ctr) in enumerate(preds):
        print(f"Scale {i}: box={box.shape}, cls={cls.shape}, ctr={ctr.shape}")

    total = sum(p.numel() for p in head.parameters())
    print(f"Head params: {total/1e6:.2f}M")
