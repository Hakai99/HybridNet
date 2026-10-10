import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    def xywh2xyxy(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    pb = xywh2xyxy(pred_boxes)
    tb = xywh2xyxy(target_boxes)

    inter_x1 = torch.max(pb[:, 0], tb[:, 0])
    inter_y1 = torch.max(pb[:, 1], tb[:, 1])
    inter_x2 = torch.min(pb[:, 2], tb[:, 2])
    inter_y2 = torch.min(pb[:, 3], tb[:, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    pred_area = (pb[:, 2] - pb[:, 0]).clamp(0) * (pb[:, 3] - pb[:, 1]).clamp(0)
    tgt_area  = (tb[:, 2] - tb[:, 0]).clamp(0) * (tb[:, 3] - tb[:, 1]).clamp(0)
    union     = pred_area + tgt_area - inter + eps
    iou       = inter / union

    enc_x1   = torch.min(pb[:, 0], tb[:, 0])
    enc_y1   = torch.min(pb[:, 1], tb[:, 1])
    enc_x2   = torch.max(pb[:, 2], tb[:, 2])
    enc_y2   = torch.max(pb[:, 3], tb[:, 3])
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    pred_cx = (pb[:, 0] + pb[:, 2]) / 2
    pred_cy = (pb[:, 1] + pb[:, 3]) / 2
    tgt_cx  = (tb[:, 0] + tb[:, 2]) / 2
    tgt_cy  = (tb[:, 1] + tb[:, 3]) / 2
    rho2    = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    pw = pred_boxes[:, 2].clamp(eps)
    ph = pred_boxes[:, 3].clamp(eps)
    tw = target_boxes[:, 2].clamp(eps)
    th = target_boxes[:, 3].clamp(eps)
    v  = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / enc_diag - alpha * v
    return (1 - ciou).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce   = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t   = torch.exp(-bce)
    focal = alpha * (1 - p_t) ** gamma * bce
    return focal.mean()


class UncertaintyWeighting(nn.Module):
    def __init__(self, num_losses=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total   = 0
        weights = []
        for i, loss in enumerate(losses):
            sigma2 = torch.exp(self.log_vars[i])
            weight = 1.0 / (2.0 * sigma2)
            reg    = 0.5 * self.log_vars[i]
            total  = total + weight * loss + reg
            weights.append(weight.item())
        return total, weights


def build_targets(predictions, targets, device):
    radius = 2
    scale_targets = []

    for (box_pred, cls_pred, ctr_pred) in predictions:
        B, _, H, W = box_pred.shape

        tgt_boxes   = torch.zeros(B, H, W, 4,                 device=device)
        tgt_classes = torch.zeros(B, H, W, cls_pred.shape[1], device=device)
        tgt_center  = torch.zeros(B, H, W, 1,                 device=device)
        tgt_mask    = torch.zeros(B, H, W, dtype=torch.bool,  device=device)

        if targets.shape[0] == 0:
            scale_targets.append({
                'boxes': tgt_boxes, 'classes': tgt_classes,
                'center': tgt_center, 'mask': tgt_mask
            })
            continue

        for t in targets:
            img_idx      = int(t[0])
            if img_idx >= B:
                continue
            cls_idx      = int(t[1])
            cx, cy, w, h = t[2].item(), t[3].item(), t[4].item(), t[5].item()

            if w <= 0 or h <= 0:
                continue

            grid_cx = cx * W
            grid_cy = cy * H

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    gx = int(grid_cx) + dx
                    gy = int(grid_cy) + dy

                    if gx < 0 or gx >= W or gy < 0 or gy >= H:
                        continue

                    cell_cx = (gx + 0.5) / W
                    cell_cy = (gy + 0.5) / H

                    if (abs(cell_cx - cx) < w / 2 and
                            abs(cell_cy - cy) < h / 2):
                        tgt_boxes[img_idx, gy, gx]            = torch.tensor([cx, cy, w, h], device=device)
                        tgt_classes[img_idx, gy, gx, cls_idx] = 1.0
                        tgt_center[img_idx, gy, gx, 0]        = 1.0
                        tgt_mask[img_idx, gy, gx]             = True

        scale_targets.append({
            'boxes': tgt_boxes, 'classes': tgt_classes,
            'center': tgt_center, 'mask': tgt_mask
        })

    return scale_targets


def _soft_guidance(student_preds, soft_preds, device, temperature=4.0, lam=0.3):
    total = torch.zeros(1, device=device)
    count = 0
    for (s_box, s_cls, s_ctr), (t_box, t_cls, t_ctr) in zip(student_preds, soft_preds):
        B, C, H, W = s_cls.shape
        s_flat = s_cls.permute(0,2,3,1).reshape(-1, C)
        t_flat = t_cls.permute(0,2,3,1).reshape(-1, C).detach()
        s_soft = F.log_softmax(s_flat / temperature, dim=-1)
        t_soft = F.softmax(t_flat / temperature, dim=-1)
        kd     = F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

        s_ctr_flat = s_ctr.permute(0,2,3,1).reshape(-1, 1)
        t_ctr_flat = t_ctr.permute(0,2,3,1).reshape(-1, 1).detach()
        ctr_kd = F.mse_loss(torch.sigmoid(s_ctr_flat), torch.sigmoid(t_ctr_flat))

        total = total + kd + 0.5 * ctr_kd
        count += 1

    if count > 0:
        total = total / count
    return total * lam


class HybridLoss(nn.Module):
    def __init__(self, num_classes, box_weight=3.0, cls_weight=1.0, ctr_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight  = box_weight
        self.cls_weight  = cls_weight
        self.ctr_weight  = ctr_weight
        self.uncertainty = UncertaintyWeighting(num_losses=3)

    def forward(self, predictions, targets, device, soft_preds=None):
        scale_targets = build_targets(predictions, targets, device)

        l_box_total = torch.zeros(1, device=device)
        l_cls_total = torch.zeros(1, device=device)
        l_ctr_total = torch.zeros(1, device=device)
        num_scales  = len(predictions)
        num_pos     = 0

        for i, ((box_pred, cls_pred, ctr_pred), tgt) in enumerate(zip(predictions, scale_targets)):
            mask  = tgt['mask']
            n_pos = mask.sum().item()
            num_pos += n_pos

            if n_pos > 0:
                l_box = ciou_loss(
                    torch.sigmoid(box_pred.permute(0,2,3,1)[mask]),
                    tgt['boxes'][mask]
                ) * self.box_weight
            else:
                l_box = torch.zeros(1, device=device)

            cls_flat     = cls_pred.permute(0,2,3,1).reshape(-1, self.num_classes)
            tgt_cls_flat = tgt['classes'].reshape(-1, self.num_classes)
            l_cls        = focal_loss(cls_flat, tgt_cls_flat) * self.cls_weight

            ctr_flat     = ctr_pred.permute(0,2,3,1).reshape(-1, 1)
            tgt_ctr_flat = tgt['center'].reshape(-1, 1)
            l_ctr        = F.binary_cross_entropy_with_logits(ctr_flat, tgt_ctr_flat) * self.ctr_weight

            l_box_total = l_box_total + l_box / num_scales
            l_cls_total = l_cls_total + l_cls / num_scales
            l_ctr_total = l_ctr_total + l_ctr / num_scales

        total_loss, weights = self.uncertainty([
            l_box_total.squeeze(),
            l_cls_total.squeeze(),
            l_ctr_total.squeeze()
        ])

        if soft_preds is not None:
            total_loss = total_loss + _soft_guidance(predictions, soft_preds, device)

        loss_dict = {
            'total': total_loss.item(),
            'box':   l_box_total.item(),
            'cls':   l_cls_total.item(),
            'ctr':   l_ctr_total.item(),
            'w_box': weights[0],
            'w_cls': weights[1],
            'w_ctr': weights[2],
            'n_pos': num_pos,
        }

        return total_loss, loss_dict
