"""
HybridNet - utils/loss.py
=========================
HybridLoss — Original Combined Loss Function

Components:
    L_box    = CIoU Loss       (accurate bounding box regression)
    L_cls    = Focal Loss      (handles class imbalance)
    L_center = BCE Loss        (centerness prediction)

YOUR NOVELTY:
    λ weights are NOT fixed.
    They use uncertainty weighting — learned during training!
    The model itself figures out how much each loss matters.

Author: Dev Kr Lahkar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# IoU Utilities
# ─────────────────────────────────────────────

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU Loss (CIoU)
    ========================
    Better than plain IoU — also considers:
        - Overlap area
        - Center distance
        - Aspect ratio consistency

    Args:
        pred_boxes:   (N, 4) — predicted x,y,w,h (normalized 0-1)
        target_boxes: (N, 4) — ground truth x,y,w,h (normalized 0-1)

    Returns:
        ciou_loss: scalar tensor
    """
    # Convert xywh → x1y1x2y2
    def xywh2xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    pb = xywh2xyxy(pred_boxes)
    tb = xywh2xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.max(pb[:, 0], tb[:, 0])
    inter_y1 = torch.max(pb[:, 1], tb[:, 1])
    inter_x2 = torch.min(pb[:, 2], tb[:, 2])
    inter_y2 = torch.min(pb[:, 3], tb[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area  = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    tgt_area   = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
    union_area = pred_area + tgt_area - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box (smallest box containing both)
    enc_x1 = torch.min(pb[:, 0], tb[:, 0])
    enc_y1 = torch.min(pb[:, 1], tb[:, 1])
    enc_x2 = torch.max(pb[:, 2], tb[:, 2])
    enc_y2 = torch.max(pb[:, 3], tb[:, 3])
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Center distance
    pred_cx = (pb[:, 0] + pb[:, 2]) / 2
    pred_cy = (pb[:, 1] + pb[:, 3]) / 2
    tgt_cx  = (tb[:, 0] + tb[:, 2]) / 2
    tgt_cy  = (tb[:, 1] + tb[:, 3]) / 2
    center_dist = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Aspect ratio consistency term (v)
    pw = pred_boxes[:, 2].clamp(eps)
    ph = pred_boxes[:, 3].clamp(eps)
    tw = target_boxes[:, 2].clamp(eps)
    th = target_boxes[:, 3].clamp(eps)

    v = (4 / (torch.pi ** 2)) * (
        torch.atan(tw / th) - torch.atan(pw / ph)
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / enc_diag) - alpha * v
    return (1 - ciou).mean()


# ─────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for classification.
    Reduces loss for easy (well-classified) examples,
    focuses training on hard examples.

    Args:
        pred:   (N, C) raw logits
        target: (N, C) binary targets
        alpha:  weighting factor
        gamma:  focusing parameter

    Returns:
        focal loss scalar
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-bce)
    # alpha_t: use `alpha` for positive targets, `(1-alpha)` for negatives,
    # instead of a flat `alpha` on everything (which was just a constant
    # rescale of the loss and did nothing to correct class imbalance).
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - p_t) ** gamma * bce
    return focal.mean()


# ─────────────────────────────────────────────
# YOUR NOVEL MODULE: Uncertainty Weighting
# ─────────────────────────────────────────────

class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-Based Loss Weighting — Original Module
    ====================================================
    Instead of fixed λ values, each loss component has a
    LEARNABLE log-variance (log_σ²) parameter.

    Formula (from Kendall et al. 2018):
        L_total = Σ [ (1/2σ²) * L_i + log(σ) ]

    The model learns σ for each task:
        - Small σ → trust this loss more
        - Large σ → trust this loss less

    This is a real research technique used in multi-task learning!
    """

    def __init__(self, num_losses=3):
        super().__init__()
        # log(σ²) for each loss — initialized to 0 (σ=1, equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        """
        Args:
            losses: list of scalar loss tensors [L_box, L_cls, L_center]

        Returns:
            total_loss: weighted sum with learned uncertainty
            weights:    current effective weights (for logging)
        """
        total = 0
        weights = []
        for i, loss in enumerate(losses):
            sigma2 = torch.exp(self.log_vars[i])         # σ²
            weight = 1.0 / (2.0 * sigma2)                # 1/2σ²
            reg    = 0.5 * self.log_vars[i]              # log(σ) regularizer
            total  = total + weight * loss + reg
            weights.append(weight.item())

        return total, weights


# ─────────────────────────────────────────────
# Target Builder (YOLO → training targets)
# ─────────────────────────────────────────────

def build_targets(predictions, targets, device, input_size=640):
    """
    Convert YOLO-format targets to per-scale training targets.

    Each object is assigned to exactly ONE scale, chosen by its box size
    (small objects -> P3, medium -> P4, large -> P5), following the same
    idea as FCOS's size-based scale assignment. Without this, every object
    was previously being written as a positive target on all three scales,
    which defeats the point of having multiple detection scales.

    On a collision (two objects landing in the same grid cell at the same
    scale), the SMALLER-area object wins, since it's more likely to be the
    one that scale is actually meant to detect, and to avoid one silently
    overwriting the other with no rule.

    Args:
        predictions: list of (box, cls, ctr) per scale, ordered P3,P4,P5
        targets:     (N, 6) — [img_idx, class, cx, cy, w, h] normalized
        device:      torch device
        input_size:  network input resolution (for converting normalized
                     w/h to pixel size when picking a scale)

    Returns:
        scale_targets: list of dicts with boxes, classes, masks per scale
    """
    strides = [8, 16, 32]       # P3=stride8, P4=stride16, P5=stride32
    # Pixel-size regression ranges per scale, FCOS-style: small objects go
    # to the high-res scale, large objects go to the low-res scale.
    size_ranges = [(0, 64), (64, 256), (256, float('inf'))]

    B_by_scale = [box_pred.shape[0] for box_pred, _, _ in predictions]
    H_W_by_scale = [(box_pred.shape[2], box_pred.shape[3]) for box_pred, _, _ in predictions]
    C = predictions[0][1].shape[1]

    scale_targets = []
    # area (in pixel^2) of whichever object currently owns each cell, so we
    # can resolve collisions by keeping the smaller object
    owner_area = []

    for scale_idx, (box_pred, cls_pred, ctr_pred) in enumerate(predictions):
        B, _, H, W = box_pred.shape
        tgt_boxes   = torch.zeros(B, H, W, 4, device=device)
        tgt_classes = torch.zeros(B, H, W, C, device=device)
        tgt_center  = torch.zeros(B, H, W, 1, device=device)
        tgt_mask    = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        scale_targets.append({
            'boxes': tgt_boxes, 'classes': tgt_classes,
            'center': tgt_center, 'mask': tgt_mask
        })
        owner_area.append(torch.full((B, H, W), float('inf'), device=device))

    if targets.shape[0] == 0:
        return scale_targets

    for t in targets:
        img_idx  = int(t[0])
        cls_idx  = int(t[1])
        cx, cy, w, h = t[2].item(), t[3].item(), t[4].item(), t[5].item()

        # Pick the scale based on the object's pixel size
        px_size = max(w, h) * input_size
        scale_idx = next(
            (i for i, (lo, hi) in enumerate(size_ranges) if lo <= px_size < hi),
            len(size_ranges) - 1  # fall back to the largest-object scale
        )

        stride = strides[scale_idx]
        H, W = H_W_by_scale[scale_idx]

        grid_x = min(max(int(cx * W), 0), W - 1)
        grid_y = min(max(int(cy * H), 0), H - 1)

        area = w * h
        cur_owner_area = owner_area[scale_idx][img_idx, grid_y, grid_x].item()

        if area < cur_owner_area:
            tgt = scale_targets[scale_idx]
            # zero out any previous (larger) object's class one-hot before
            # writing the new one, since this cell is being reassigned
            tgt['classes'][img_idx, grid_y, grid_x, :] = 0.0
            tgt['boxes'][img_idx, grid_y, grid_x]      = torch.tensor([cx, cy, w, h], device=device)
            tgt['classes'][img_idx, grid_y, grid_x, cls_idx] = 1.0
            tgt['center'][img_idx, grid_y, grid_x, 0]  = 1.0
            tgt['mask'][img_idx, grid_y, grid_x]        = True
            owner_area[scale_idx][img_idx, grid_y, grid_x] = area

    return scale_targets


# ─────────────────────────────────────────────
# MAIN LOSS: HybridLoss
# ─────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    HybridLoss — Original Multi-Task Detection Loss
    =================================================
    L_total = uncertainty_weight(L_box, L_cls, L_center)

    Where:
        L_box    = CIoU loss on positive samples
        L_cls    = Focal loss on all samples
        L_center = BCE loss on centerness

    Uncertainty weighting (your novelty) automatically
    balances these 3 losses during training.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.uncertainty = UncertaintyWeighting(num_losses=3)

    def forward(self, predictions, targets, device):
        """
        Args:
            predictions: list of (box, cls, ctr) per scale
            targets:     (N, 6) YOLO format [img, cls, cx, cy, w, h]
            device:      torch device

        Returns:
            total_loss:  scalar tensor (backprop this)
            loss_dict:   dict with individual losses for logging
        """
        scale_targets = build_targets(predictions, targets, device)
        strides = [8, 16, 32]
        input_size = 640

        l_box_total = torch.tensor(0.0, device=device)
        l_cls_total = torch.tensor(0.0, device=device)
        l_ctr_total = torch.tensor(0.0, device=device)
        num_scales  = len(predictions)

        for i, ((box_pred, cls_pred, ctr_pred), tgt) in enumerate(
            zip(predictions, scale_targets)
        ):
            mask    = tgt['mask']          # (B, H, W) bool
            B, H, W = mask.shape
            stride  = strides[i]

            # ── Box Loss (only on positive/matched cells) ──
            if mask.sum() > 0:
                device_ = box_pred.device
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=device_),
                    torch.arange(W, device=device_),
                    indexing='ij'
                )
                # Decode predicted boxes grid-relative, exactly like
                # decode_predictions, so training and inference agree on
                # what the raw head outputs mean.
                cx = (grid_x.unsqueeze(0) + torch.sigmoid(box_pred[:, 0])) / W
                cy = (grid_y.unsqueeze(0) + torch.sigmoid(box_pred[:, 1])) / H
                w  = torch.exp(box_pred[:, 2].clamp(max=6)) * stride / input_size
                h  = torch.exp(box_pred[:, 3].clamp(max=6)) * stride / input_size
                box_decoded = torch.stack([cx, cy, w, h], dim=-1)  # (B,H,W,4)

                pred_boxes_pos = box_decoded[mask]   # (K, 4)
                tgt_boxes_pos  = tgt['boxes'][mask]  # (K, 4)
                l_box = ciou_loss(pred_boxes_pos, tgt_boxes_pos)
            else:
                l_box = torch.tensor(0.0, device=device)

            # ── Class Loss (all cells) ──
            cls_pred_flat = cls_pred.permute(0,2,3,1).reshape(-1, self.num_classes)
            tgt_cls_flat  = tgt['classes'].reshape(-1, self.num_classes)
            l_cls = focal_loss(cls_pred_flat, tgt_cls_flat)

            # ── Centerness Loss (all cells) ──
            ctr_pred_flat = ctr_pred.permute(0,2,3,1).reshape(-1, 1)
            tgt_ctr_flat  = tgt['center'].reshape(-1, 1)
            l_ctr = F.binary_cross_entropy_with_logits(ctr_pred_flat, tgt_ctr_flat)

            l_box_total = l_box_total + l_box / num_scales
            l_cls_total = l_cls_total + l_cls / num_scales
            l_ctr_total = l_ctr_total + l_ctr / num_scales

        # Uncertainty-weighted total loss
        total_loss, weights = self.uncertainty([l_box_total, l_cls_total, l_ctr_total])

        loss_dict = {
            'total': total_loss.item(),
            'box':   l_box_total.item(),
            'cls':   l_cls_total.item(),
            'ctr':   l_ctr_total.item(),
            'w_box': weights[0],
            'w_cls': weights[1],
            'w_ctr': weights[2],
        }

        return total_loss, loss_dict


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from model.hybridnet import HybridNet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = HybridNet(num_classes=3).to(device)
    criterion = HybridLoss(num_classes=3).to(device)

    imgs = torch.randn(2, 3, 640, 640).to(device)
    # Fake targets: [img_idx, class, cx, cy, w, h]
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.3, 0.4],
        [0, 1, 0.2, 0.3, 0.1, 0.2],
        [1, 2, 0.7, 0.6, 0.2, 0.3],
    ], device=device)

    preds = model(imgs)
    loss, loss_dict = criterion(preds, targets, device)

    print(f"Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
