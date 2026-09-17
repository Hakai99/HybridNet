"""
HybridNet - backbone.py
=======================
LiteHybridNet Backbone with Context Attention Block (CAB)

Architecture:
    Input → Stem → Stage1 → Stage2(P3) → Stage3+CAB(P4) → Stage4+CAB(P5)

Author: Your Name
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Basic Building Blocks
# ─────────────────────────────────────────────

class ConvBNReLU6(nn.Module):
    """Standard Conv → BatchNorm → ReLU6 block (TFLite friendly)"""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LiteBlock(nn.Module):
    """
    Lightweight Depthwise Separable Block with Residual Connection.
    Depthwise Conv (per-channel) + Pointwise Conv (1x1) = efficient!
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.block = nn.Sequential(
            # Depthwise conv (each channel separately)
            ConvBNReLU6(in_ch, in_ch, kernel=3, stride=stride, groups=in_ch),
            # Pointwise conv (mix channels)
            ConvBNReLU6(in_ch, out_ch, kernel=1)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x   # skip connection
        return out


# ─────────────────────────────────────────────
# YOUR NOVEL MODULE: Context Attention Block (CAB)
# ─────────────────────────────────────────────

class CAB(nn.Module):
    """
    Context Attention Block — Original Module
    ==========================================
    Combines THREE attention mechanisms:
        1. Channel Attention   — WHAT features matter
        2. Spatial Attention   — WHERE features matter
        3. Local Context Gate  — HOW WIDE the context is (5x5 depthwise)

    Most models use channel OR spatial attention.
    CAB fuses all three → your research novelty!
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        # 1. Channel Attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # squeeze spatial → (B, C, 1, 1)
            nn.Flatten(),                                      # (B, C)
            nn.Linear(channels, channels // reduction),
            nn.ReLU6(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # 2. Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),  # (B, 1, H, W)
            nn.Sigmoid()
        )

        # 3. Local Context Gate (wider receptive field via 5x5 depthwise)
        self.context_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Fusion layer — combine all 3 into final output
        self.fuse = ConvBNReLU6(channels * 3, channels, kernel=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Channel attention → scale each channel
        ch_w = self.channel_att(x).view(B, C, 1, 1)
        x_ch = x * ch_w                          # (B, C, H, W)

        # 2. Spatial attention → scale each spatial location
        sp_w = self.spatial_att(x)
        x_sp = x * sp_w                          # (B, C, H, W)

        # 3. Local context gate → wider context per channel
        x_ctx = x * self.context_gate(x)         # (B, C, H, W)

        # Fuse all three attention outputs
        out = self.fuse(torch.cat([x_ch, x_sp, x_ctx], dim=1))

        return out + x   # residual


# ─────────────────────────────────────────────
# BACKBONE: LiteHybridNet
# ─────────────────────────────────────────────

class LiteHybridNet(nn.Module):
    """
    LiteHybridNet Backbone
    ======================
    Lightweight CNN backbone with CAB attention blocks at deeper stages.
    Outputs 3 feature maps (P3, P4, P5) for the neck.

    Feature map sizes (with 640x640 input):
        P3 → 80x80x128   (small objects)
        P4 → 40x40x256   (medium objects)
        P5 → 20x20x512   (large objects)
    """

    def __init__(self):
        super().__init__()

        # ── Stage 0: Stem ──────────────────────────────
        # 640x640x3 → 320x320x32
        self.stem = ConvBNReLU6(3, 32, kernel=3, stride=2)

        # ── Stage 1 ────────────────────────────────────
        # 320x320x32 → 160x160x64
        self.stage1 = nn.Sequential(
            LiteBlock(32, 64, stride=2),
            LiteBlock(64, 64, stride=1),
        )

        # ── Stage 2 → P3 ───────────────────────────────
        # 160x160x64 → 80x80x128
        self.stage2 = nn.Sequential(
            LiteBlock(64, 128, stride=2),
            LiteBlock(128, 128, stride=1),
            LiteBlock(128, 128, stride=1),
        )

        # ── Stage 3 → P4 (with CAB) ────────────────────
        # 80x80x128 → 40x40x256
        self.stage3 = nn.Sequential(
            LiteBlock(128, 256, stride=2),
            LiteBlock(256, 256, stride=1),
            LiteBlock(256, 256, stride=1),
            LiteBlock(256, 256, stride=1),
        )
        self.cab3 = CAB(256)   # attention on P4

        # ── Stage 4 → P5 (with CAB) ────────────────────
        # 40x40x256 → 20x20x512
        self.stage4 = nn.Sequential(
            LiteBlock(256, 512, stride=2),
            LiteBlock(512, 512, stride=1),
            LiteBlock(512, 512, stride=1),
        )
        self.cab4 = CAB(512)   # attention on P5

    def forward(self, x):
        x = self.stem(x)       # 320x320x32
        x = self.stage1(x)     # 160x160x64
        p3 = self.stage2(x)    # 80x80x128  ← P3

        p4 = self.stage3(p3)   # 40x40x256
        p4 = self.cab3(p4)     # attention  ← P4

        p5 = self.stage4(p4)   # 20x20x512
        p5 = self.cab4(p5)     # attention  ← P5

        return p3, p4, p5


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = LiteHybridNet()
    dummy = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = model(dummy)
    print(f"P3: {p3.shape}")   # expect [1, 128, 80, 80]
    print(f"P4: {p4.shape}")   # expect [1, 256, 40, 40]
    print(f"P5: {p5.shape}")   # expect [1, 512, 20, 20]

    total = sum(p.numel() for p in model.parameters())
    print(f"Backbone params: {total/1e6:.2f}M")
