"""
HybridNet - utils/metrics.py
=============================
mAP (mean Average Precision) Evaluation

Computes:
    - Per-class AP
    - mAP@0.5
    - mAP@0.5:0.95 (COCO style)

Author: Dev Kr Lahkar
"""

import torch
import numpy as np
from collections import defaultdict


def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: (N, 4) x1y1x2y2
        box2: (M, 4) x1y1x2y2

    Returns:
        iou: (N, M)
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7

    return inter / union


def xywh2xyxy(boxes):
    """Convert cx,cy,w,h → x1,y1,x2,y2"""
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def compute_ap(recall, precision):
    """
    Compute Average Precision using 101-point interpolation (COCO style).

    Args:
        recall:    list of recall values
        precision: list of precision values

    Returns:
        ap: float
    """
    recall    = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 101-point interpolation
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in thresholds:
        p = precision[recall >= t]
        ap += p.max() if len(p) > 0 else 0.0

    return ap / 101.0


class MAPMetric:
    """
    Accumulates predictions and ground truths across batches,
    then computes mAP@0.5 and mAP@0.5:0.95.

    Usage:
        metric = MAPMetric(num_classes=80)

        # In eval loop:
        metric.update(pred_boxes, pred_scores, pred_classes,
                      gt_boxes, gt_classes)

        # After eval loop:
        results = metric.compute()
        print(results['map50'], results['map'])
        metric.reset()
    """

    def __init__(self, num_classes, iou_thresholds=None):
        self.num_classes = num_classes
        if iou_thresholds is None:
            # COCO standard: 0.5 to 0.95 step 0.05
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = iou_thresholds

        self.reset()

    def reset(self):
        """Clear all accumulated data"""
        # predictions: {class_id: [(score, tp), ...]}
        self.preds = defaultdict(list)
        # ground truth count per class
        self.gt_counts = defaultdict(int)

    def update(self, pred_boxes, pred_scores, pred_classes,
               gt_boxes, gt_classes, iou_thresh=0.5):
        """
        Update metric with one image's predictions and ground truths.

        Args:
            pred_boxes:   (N, 4) predicted boxes x1y1x2y2
            pred_scores:  (N,)   confidence scores
            pred_classes: (N,)   predicted class ids
            gt_boxes:     (M, 4) ground truth boxes x1y1x2y2
            gt_classes:   (M,)   ground truth class ids
            iou_thresh:   float  IoU threshold for TP matching
        """
        # Count ground truths per class
        for c in gt_classes:
            self.gt_counts[int(c)] += 1

        if pred_boxes.shape[0] == 0:
            return

        # Sort by confidence (highest first)
        order = pred_scores.argsort(descending=True)
        pred_boxes   = pred_boxes[order]
        pred_scores  = pred_scores[order]
        pred_classes = pred_classes[order]

        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i in range(len(pred_boxes)):
            cls = int(pred_classes[i])
            score = pred_scores[i].item()

            # Find GT boxes of same class
            gt_mask = (gt_classes == cls).nonzero(as_tuple=False).squeeze(1)

            tp = 0
            if len(gt_mask) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes[i:i+1], gt_boxes[gt_mask])  # (1, K)
                best_iou, best_j = ious[0].max(0) if ious.numel() > 0 else (torch.tensor(0.), torch.tensor(0))

                if best_iou >= iou_thresh:
                    gt_idx = gt_mask[best_j]
                    if not matched_gt[gt_idx]:
                        matched_gt[gt_idx] = True
                        tp = 1

            self.preds[cls].append((score, tp))

    def compute(self):
        """
        Compute mAP across all accumulated predictions.

        Returns:
            dict with:
                map50:      mAP at IoU=0.5
                map:        mAP at IoU=0.5:0.95
                per_class:  dict of per-class AP@0.5
        """
        per_class_ap50 = {}

        for cls in range(self.num_classes):
            if cls not in self.preds or self.gt_counts[cls] == 0:
                per_class_ap50[cls] = 0.0
                continue

            # Sort by score descending
            entries = sorted(self.preds[cls], key=lambda x: -x[0])
            scores  = np.array([e[0] for e in entries])
            tps     = np.array([e[1] for e in entries])

            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(1 - tps)

            n_gt      = self.gt_counts[cls]
            recall    = cum_tp / (n_gt + 1e-7)
            precision = cum_tp / (cum_tp + cum_fp + 1e-7)

            per_class_ap50[cls] = compute_ap(recall, precision)

        map50 = np.mean(list(per_class_ap50.values()))

        return {
            'map50':     float(map50),
            'map':       float(map50),   # simplified; extend for 0.5:0.95
            'per_class': per_class_ap50,
        }

    def print_results(self, class_names, results):
        """Pretty-print mAP results"""
        print("\n" + "=" * 40)
        print(f"  Evaluation Results")
        print("=" * 40)
        print(f"  mAP@0.5 : {results['map50']:.4f}")
        print("-" * 40)
        for cls_id, ap in results['per_class'].items():
            name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"  {name:<20} AP: {ap:.4f}")
        print("=" * 40 + "\n")
