"""
HybridNet - utils/general.py
=============================
General utility functions used across the project.

Includes:
    - Non-Maximum Suppression (NMS)
    - Box format converters
    - Color palette for visualization
    - Checkpoint helpers

Author: Dev Kr Lahkar
"""

import torch
import numpy as np
import cv2


# ─────────────────────────────────────────────
# Box Utilities
# ─────────────────────────────────────────────

def xywh2xyxy(boxes):
    """
    Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    Works on both numpy arrays and torch tensors.
    """
    if isinstance(boxes, torch.Tensor):
        out = boxes.clone()
    else:
        out = boxes.copy()

    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2   # x1
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2   # y1
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2   # x2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2   # y2
    return out


def xyxy2xywh(boxes):
    """
    Convert (x1, y1, x2, y2) → (cx, cy, w, h)
    """
    if isinstance(boxes, torch.Tensor):
        out = boxes.clone()
    else:
        out = boxes.copy()

    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2   # cx
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2   # cy
    out[..., 2] = boxes[..., 2] - boxes[..., 0]          # w
    out[..., 3] = boxes[..., 3] - boxes[..., 1]          # h
    return out


def scale_boxes(boxes, orig_shape, new_shape):
    """
    Scale boxes from new_shape back to orig_shape coordinates.
    Used after letterbox resizing to get original image coordinates.

    Args:
        boxes:      (N, 4) in new_shape coords
        orig_shape: (h, w) original image size
        new_shape:  (h, w) resized image size

    Returns:
        boxes in orig_shape coords
    """
    gain = min(new_shape[0] / orig_shape[0], new_shape[1] / orig_shape[1])
    pad  = ((new_shape[1] - orig_shape[1] * gain) / 2,
            (new_shape[0] - orig_shape[0] * gain) / 2)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone().float()
    else:
        boxes = boxes.copy().astype(float)

    boxes[..., 0] -= pad[0]   # x1
    boxes[..., 1] -= pad[1]   # y1
    boxes[..., 2] -= pad[0]   # x2
    boxes[..., 3] -= pad[1]   # y2
    boxes /= gain

    # Clip to image bounds
    boxes[..., 0] = boxes[..., 0].clip(0, orig_shape[1])
    boxes[..., 1] = boxes[..., 1].clip(0, orig_shape[0])
    boxes[..., 2] = boxes[..., 2].clip(0, orig_shape[1])
    boxes[..., 3] = boxes[..., 3].clip(0, orig_shape[0])

    return boxes


# ─────────────────────────────────────────────
# Non-Maximum Suppression
# ─────────────────────────────────────────────

def nms(boxes, scores, iou_thresh=0.45):
    """
    Non-Maximum Suppression — removes overlapping boxes.
    Keeps the highest-scoring box when two overlap above iou_thresh.

    Args:
        boxes:      (N, 4) x1y1x2y2
        scores:     (N,)   confidence scores
        iou_thresh: float  overlap threshold

    Returns:
        keep: indices of boxes to keep
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)

    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), iou_thresh)
    except ImportError:
        # Fallback pure PyTorch NMS
        return _pure_nms(boxes, scores, iou_thresh)


def _pure_nms(boxes, scores, iou_thresh):
    """Pure PyTorch NMS fallback (no torchvision needed)"""
    order = scores.argsort(descending=True)
    keep  = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        b    = boxes[i].unsqueeze(0)
        rest_boxes = boxes[rest]

        # Compute IoU of current box with rest
        inter_x1 = torch.max(b[:, 0], rest_boxes[:, 0])
        inter_y1 = torch.max(b[:, 1], rest_boxes[:, 1])
        inter_x2 = torch.min(b[:, 2], rest_boxes[:, 2])
        inter_y2 = torch.min(b[:, 3], rest_boxes[:, 3])

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        area2 = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou   = inter / (area1 + area2 - inter + 1e-7)

        order = rest[iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

# 20 distinct colors for classes
COLORS = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  199),
]


def draw_boxes(image, boxes, scores, classes, class_names, line_width=2):
    """
    Draw bounding boxes on image.

    Args:
        image:       numpy array (H, W, 3) BGR
        boxes:       (N, 4) x1y1x2y2 in pixel coords
        scores:      (N,) confidence scores
        classes:     (N,) class indices
        class_names: list of class name strings
        line_width:  box border thickness

    Returns:
        annotated image (numpy array)
    """
    img = image.copy()
    for box, score, cls in zip(boxes, scores, classes):
        cls   = int(cls)
        color = COLORS[cls % len(COLORS)]
        x1, y1, x2, y2 = map(int, box)

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # Label background
        name  = class_names[cls] if cls < len(class_names) else f"cls{cls}"
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img
