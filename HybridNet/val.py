"""
HybridNet - val.py
==================
Validation & mAP Evaluation Script

Usage:
    python val.py --weights runs/exp1/best.pt --data data.yaml

Author: Dev Kr Lahkar
"""

import torch
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='HybridNet Validation')
    parser.add_argument('--weights', type=str, required=True, help='path to best.pt')
    parser.add_argument('--data',    type=str, required=True, help='path to data.yaml')
    parser.add_argument('--imgsz',   type=int, default=640,   help='image size')
    parser.add_argument('--batch',   type=int, default=16,    help='batch size')
    parser.add_argument('--conf',    type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou',     type=float, default=0.6,   help='NMS IoU threshold')
    parser.add_argument('--device',  type=str, default='',    help='cuda or cpu')
    return parser.parse_args()


def get_device(device_str):
    if device_str == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


@torch.no_grad()
def validate():
    args   = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*45}")
    print(f"  HybridNet Validation")
    print(f"{'='*45}")
    print(f"  Weights: {args.weights}")
    print(f"  Data   : {args.data}")
    print(f"  Device : {device}\n")

    # ── Model ───────────────────────────────────────
    from model.hybridnet import build_model
    ckpt = torch.load(args.weights, map_location=device)
    num_classes = ckpt.get('num_classes', 80)
    model = build_model(num_classes=num_classes, weights=args.weights, device=str(device))
    model.eval()

    # ── Dataloader ──────────────────────────────────
    from utils.dataloader import build_dataloader
    val_loader, nc, class_names = build_dataloader(
        args.data, split='val',
        img_size=args.imgsz,
        batch_size=args.batch,
        workers=2
    )

    # ── Metrics ─────────────────────────────────────
    from utils.metrics import MAPMetric
    from utils.general import xywh2xyxy, nms

    metric = MAPMetric(num_classes=num_classes)

    print(f"Evaluating on {len(val_loader)} batches...\n")

    for batch_idx, (imgs, targets, _) in enumerate(val_loader):
        imgs    = imgs.to(device)
        targets = targets.to(device)

        # Forward
        predictions = model(imgs)

        # Decode predictions per image
        B = imgs.shape[0]
        for img_idx in range(B):
            # Get ground truth for this image
            gt_mask   = targets[:, 0] == img_idx
            gt        = targets[gt_mask]                      # (M, 6)
            gt_boxes  = xywh2xyxy(gt[:, 2:6]) if gt.shape[0] > 0 else torch.zeros(0, 4)
            gt_classes = gt[:, 1] if gt.shape[0] > 0 else torch.zeros(0)

            # Gather predictions from all scales for this image
            all_boxes, all_scores, all_classes = [], [], []

            for box_pred, cls_pred, ctr_pred in predictions:
                box_pred = torch.sigmoid(box_pred[img_idx])   # (4, H, W)
                cls_pred = torch.sigmoid(cls_pred[img_idx])   # (C, H, W)
                ctr_pred = torch.sigmoid(ctr_pred[img_idx])   # (1, H, W)

                H, W = box_pred.shape[1], box_pred.shape[2]

                # Score = class_prob * centerness
                scores, cls_idx = (cls_pred * ctr_pred).max(dim=0)  # (H, W)
                scores   = scores.view(-1)
                cls_idx  = cls_idx.view(-1).float()
                boxes_hw = box_pred.permute(1, 2, 0).reshape(-1, 4)

                # Filter by conf
                mask = scores > args.conf
                if mask.sum() == 0:
                    continue

                all_boxes.append(boxes_hw[mask])
                all_scores.append(scores[mask])
                all_classes.append(cls_idx[mask])

            if not all_boxes:
                metric.update(
                    torch.zeros(0, 4), torch.zeros(0), torch.zeros(0),
                    gt_boxes.to(device), gt_classes.to(device)
                )
                continue

            all_boxes   = torch.cat(all_boxes, dim=0)
            all_scores  = torch.cat(all_scores, dim=0)
            all_classes = torch.cat(all_classes, dim=0)

            # Convert to xyxy for NMS
            boxes_xyxy = xywh2xyxy(all_boxes)

            # NMS
            keep = nms(boxes_xyxy, all_scores, args.iou)
            boxes_xyxy  = boxes_xyxy[keep]
            all_scores  = all_scores[keep]
            all_classes = all_classes[keep]

            metric.update(
                boxes_xyxy, all_scores, all_classes,
                gt_boxes.to(device), gt_classes.to(device)
            )

        print(f"\r  [{batch_idx+1}/{len(val_loader)}]", end='')

    print()

    # ── Results ─────────────────────────────────────
    results = metric.compute()
    metric.print_results(class_names, results)

    return results


if __name__ == '__main__':
    validate()
