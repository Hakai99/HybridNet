"""
HybridNet - neck.py
===================
AdaFPN — Adaptive Feature Pyramid Network (Original)

Key novelty:
    Each feature fusion point has LEARNABLE weights.
    The model itself learns how much to trust each scale
    during training → adaptive, not fixed like standard FPN.

Flow:
    P3(80x80x128) ─┐
    P4(40x40x256) ─┼─→ AdaFPN ─→ out3, out4, out5
    P5(20x20x512) ─┘

Author: Dev Kr Lahkar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU6(nn.Module):
    """Conv → BN → ReLU6 (reused here to avoid circular import)"""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# YOUR NOVEL MODULE: Adaptive Weighted Fusion
# ─────────────────────────────────────────────

class AdaptiveFusion(nn.Module):
    """
    Adaptive Weighted Fusion — Original Module
    ==========================================
    Fuses two feature maps using LEARNABLE weights.
    
    Instead of simple add or concat, weights are learned:
        w = softmax([w1, w2])
        out = (w[0] * feat1 + w[1] * feat2) / sum(w) + ε
    
    This lets the model decide at each scale:
    "How much should I trust the top-down path vs bottom-up?"
    """

    def __init__(self, num_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        # Learnable weights — initialized equally
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, feats):
        """
        feats: list of feature tensors (all same shape)
        """
        # Normalize weights to sum to 1, always positive.
        # Using a small floor instead of a hard ReLU(x) here: ReLU has zero
        # gradient for any negative weight, so once a weight dips below 0
        # it can never recover during training. clamp(min=eps) keeps a
        # nonzero gradient everywhere so a weight can still climb back up.
        w = self.weights.clamp(min=self.eps)
        w = w / (w.sum() + self.eps)

        # Weighted sum
        out = sum(w[i] * feats[i] for i in range(len(feats)))
        return out


# ─────────────────────────────────────────────
# NECK: AdaFPN
# ─────────────────────────────────────────────

class AdaFPN(nn.Module):
    """
    AdaFPN — Adaptive Feature Pyramid Network
    ==========================================
    Two-path feature fusion neck:
        Path 1: Top-Down   (P5 → P4 → P3) — large features guide small
        Path 2: Bottom-Up  (P3 → P4 → P5) — small features refine large
    
    All fusion points use AdaptiveFusion (learnable weights).
    All channels aligned to `out_ch` (default 128) for efficiency.

    Outputs:
        out3 → 80x80x128  (for small object detection)
        out4 → 40x40x128  (for medium object detection)
        out5 → 20x20x128  (for large object detection)
    """

    def __init__(self, in_channels=(128, 256, 512), out_ch=128):
        super().__init__()
        c3, c4, c5 = in_channels

        # ── Lateral convs — align all to out_ch ────────
        self.lat3 = ConvBNReLU6(c3, out_ch, kernel=1)
        self.lat4 = ConvBNReLU6(c4, out_ch, kernel=1)
        self.lat5 = ConvBNReLU6(c5, out_ch, kernel=1)

        # ── Top-Down path ───────────────────────────────
        # P5 fuses into P4
        self.td_fuse_p4 = AdaptiveFusion(num_inputs=2)
        self.td_conv_p4 = ConvBNReLU6(out_ch, out_ch, kernel=3)

        # Fused P4 fuses into P3
        self.td_fuse_p3 = AdaptiveFusion(num_inputs=2)
        self.td_conv_p3 = ConvBNReLU6(out_ch, out_ch, kernel=3)

        # ── Bottom-Up path ──────────────────────────────
        # Fused P3 fuses into P4
        self.bu_fuse_p4 = AdaptiveFusion(num_inputs=2)
        self.bu_conv_p4 = ConvBNReLU6(out_ch, out_ch, kernel=3)

        # Fused P4 fuses into P5
        self.bu_fuse_p5 = AdaptiveFusion(num_inputs=2)
        self.bu_conv_p5 = ConvBNReLU6(out_ch, out_ch, kernel=3)

        # Downsample: stride-2 conv (for bottom-up path)
        self.down1 = ConvBNReLU6(out_ch, out_ch, kernel=3, stride=2)
        self.down2 = ConvBNReLU6(out_ch, out_ch, kernel=3, stride=2)

    def forward(self, p3, p4, p5):
        # ── Align channels ──────────────────────────────
        p3 = self.lat3(p3)   # 80x80x128
        p4 = self.lat4(p4)   # 40x40x128
        p5 = self.lat5(p5)   # 20x20x128

        # ── Top-Down path ───────────────────────────────
        # P5 → upsample to P4 size, fuse with P4
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        td_p4 = self.td_conv_p4(self.td_fuse_p4([p5_up, p4]))   # 40x40x128

        # Fused P4 → upsample to P3 size, fuse with P3
        td_p4_up = F.interpolate(td_p4, size=p3.shape[2:], mode='nearest')
        td_p3 = self.td_conv_p3(self.td_fuse_p3([td_p4_up, p3]))  # 80x80x128

        # ── Bottom-Up path ──────────────────────────────
        # Fused P3 → downsample to P4 size, fuse with td_P4
        p3_down = self.down1(td_p3)                                # 40x40x128
        bu_p4 = self.bu_conv_p4(self.bu_fuse_p4([p3_down, td_p4]))  # 40x40x128

        # Fused P4 → downsample to P5 size, fuse with P5
        p4_down = self.down2(bu_p4)                                # 20x20x128
        bu_p5 = self.bu_conv_p5(self.bu_fuse_p5([p4_down, p5]))    # 20x20x128

        # out3 = small objects, out4 = medium, out5 = large
        return td_p3, bu_p4, bu_p5


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    neck = AdaFPN()
    p3 = torch.randn(1, 128, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 512, 20, 20)

    out3, out4, out5 = neck(p3, p4, p5)
    print(f"out3: {out3.shape}")   # expect [1, 128, 80, 80]
    print(f"out4: {out4.shape}")   # expect [1, 128, 40, 40]
    print(f"out5: {out5.shape}")   # expect [1, 128, 20, 20]

    total = sum(p.numel() for p in neck.parameters())
    print(f"Neck params: {total/1e6:.2f}M")
