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

    # ── Dataloader ──────────────────────────────────
    from model.hybridnet import build_model
    from utils.dataloader import build_dataloader

    # Load validation dataset and get actual class count
    val_loader, nc, class_names = build_dataloader(
        args.data,
        split='val',
        img_size=args.imgsz,
        batch_size=args.batch,
        workers=0
    )

    num_classes = nc

    # ── Model ───────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        weights=args.weights,
        device=str(device)
    )
    model.eval()

    # ── Metrics ─────────────────────────────────────
    from utils.metrics import MAPMetric
    from utils.general import xywh2xyxy
    from model.head import decode_scale
    from torchvision.ops import batched_nms

    STRIDES  = (8, 16, 32)
    MAX_PRE  = 1000    # top-scoring candidates kept before NMS
    MAX_DET  = 300     # detections kept per image

    metric = MAPMetric(num_classes=num_classes)

    print(f"Evaluating on {len(val_loader)} batches...\n")

    for batch_idx, (imgs, targets, _) in enumerate(val_loader):
        imgs = imgs.to(device)

        predictions = model(imgs)

        # Decode every scale with the SAME encoding the loss uses
        per_scale = []
        for (box_pred, cls_pred, ctr_pred), stride in zip(predictions, STRIDES):
            B = box_pred.shape[0]
            boxes = decode_scale(box_pred, stride, args.imgsz).reshape(B, -1, 4)   # cxcywh, normalized
            probs = torch.sigmoid(cls_pred.float()) * torch.sigmoid(ctr_pred.float())   # (B,C,H,W)
            probs = probs.permute(0, 2, 3, 1).reshape(B, -1, num_classes)             # (B,HW,C)
            per_scale.append((boxes, probs))

        boxes_all = torch.cat([p[0] for p in per_scale], dim=1)   # (B, N, 4)
        probs_all = torch.cat([p[1] for p in per_scale], dim=1)   # (B, N, C)

        for img_idx in range(imgs.shape[0]):
            gt         = targets[targets[:, 0] == img_idx].cpu()
            gt_boxes   = xywh2xyxy(gt[:, 2:6]) if gt.shape[0] > 0 else torch.zeros(0, 4)
            gt_classes = gt[:, 1] if gt.shape[0] > 0 else torch.zeros(0)

            scores, cls_idx = probs_all[img_idx].max(dim=1)       # best class per cell
            keep = scores > args.conf
            if keep.sum() == 0:
                metric.update(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0),
                              gt_boxes, gt_classes)
                continue

            scores  = scores[keep]
            cls_idx = cls_idx[keep]
            boxes   = xywh2xyxy(boxes_all[img_idx][keep])

            if scores.numel() > MAX_PRE:
                top = scores.topk(MAX_PRE).indices
                scores, cls_idx, boxes = scores[top], cls_idx[top], boxes[top]

            # class-aware NMS
            k = batched_nms(boxes, scores, cls_idx, args.iou)[:MAX_DET]

            metric.update(
                boxes[k].cpu(), scores[k].cpu(), cls_idx[k].float().cpu(),
                gt_boxes, gt_classes
            )

        print(f"\r  [{batch_idx+1}/{len(val_loader)}]", end='')

    print()

    # ── Results ─────────────────────────────────────
    results = metric.compute()
    metric.print_results(class_names, results)

    return results


if __name__ == '__main__':
    validate()
