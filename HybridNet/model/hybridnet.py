"""
HybridNet - hybridnet.py
========================
Full Model Assembly

Connects:
    LiteHybridNet (Backbone)
    → AdaFPN (Neck)
    → AnchorFreeHead (Head)

Usage:
    model = HybridNet(num_classes=80)
    preds = model(images)

Author: Dev Kr Lahkar
"""

import torch
import torch.nn as nn

from model.backbone import LiteHybridNet
from model.neck import AdaFPN
from model.head import AnchorFreeHead


class HybridNet(nn.Module):
    """
    HybridNet — Full Object Detection Model
    ========================================
    Original lightweight anchor-free detector.
    YOLO-annotation compatible training.
    TFLite export friendly.

    Args:
        num_classes (int): number of object classes (from data.yaml)
        neck_ch     (int): internal channel size for neck (default 128)
    """

    def __init__(self, num_classes=80, neck_ch=128):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ────────────────────────────────────
        self.backbone = LiteHybridNet()
        # Outputs: P3(128), P4(256), P5(512)

        # ── Neck ────────────────────────────────────────
        self.neck = AdaFPN(
            in_channels=(128, 256, 512),
            out_ch=neck_ch
        )
        # Outputs: out3(128), out4(128), out5(128)

        # ── Head ────────────────────────────────────────
        self.head = AnchorFreeHead(
            in_ch=neck_ch,
            num_classes=num_classes
        )
        # Outputs: list of (box, cls, ctr) per scale

        self._init_weights()

    def _init_weights(self):
        """Apply sensible default initialization across all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image tensor, H=W=640

        Returns (training):
            list of (box, cls, ctr) tuples — one per detection scale
        """
        # Backbone: extract multi-scale features
        p3, p4, p5 = self.backbone(x)

        # Neck: fuse features across scales
        out3, out4, out5 = self.neck(p3, p4, p5)

        # Head: predict boxes, classes, centerness
        predictions = self.head(out3, out4, out5)

        return predictions

    @torch.no_grad()
    def predict(self, x, conf_thresh=0.25, iou_thresh=0.45):
        """
        Full inference with NMS.
        Use this for detection (not training).

        Args:
            x:           (1, 3, H, W) single image tensor
            conf_thresh: minimum confidence score
            iou_thresh:  NMS IoU threshold

        Returns:
            boxes:   (N, 4) x1,y1,x2,y2
            scores:  (N,)
            classes: (N,)
        """
        self.eval()
        predictions = self.forward(x)

        # Decode raw predictions
        boxes, scores, classes = self.head.decode_predictions(
            predictions, conf_thresh=conf_thresh
        )

        if boxes.shape[0] == 0:
            return boxes, scores, classes

        # Apply NMS
        from torchvision.ops import nms
        keep = nms(boxes, scores, iou_thresh)

        return boxes[keep], scores[keep], classes[keep]

    def info(self):
        """Print model summary"""
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 45)
        print(f"  HybridNet Object Detector")
        print("=" * 45)
        print(f"  Classes     : {self.num_classes}")
        print(f"  Total params: {total/1e6:.2f}M")
        print(f"  Trainable   : {trainable/1e6:.2f}M")
        print(f"  Input size  : 640x640")
        print(f"  Scales      : 3 (80x80, 40x40, 20x20)")
        print("=" * 45)


def build_model(num_classes, weights=None, device='cpu'):
    """
    Build HybridNet model.
    Optionally load pretrained weights.

    Args:
        num_classes (int): number of classes
        weights (str):     path to .pt weights file (optional)
        device (str):      'cpu' or 'cuda'

    Returns:
        model: HybridNet ready to train or infer
    """
    model = HybridNet(num_classes=num_classes)

    if weights is not None:
        ckpt = torch.load(weights, map_location=device)
        # Support both raw state_dict and checkpoint dict
        state = ckpt.get('model_state', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from: {weights}")

    model = model.to(device)
    return model


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HybridNet(num_classes=80).to(device)
    model.info()

    dummy = torch.randn(2, 3, 640, 640).to(device)
    preds = model(dummy)

    print("\nPrediction shapes per scale:")
    for i, (box, cls, ctr) in enumerate(preds):
        print(f"  Scale {i}: box={box.shape}, cls={cls.shape}, ctr={ctr.shape}")
