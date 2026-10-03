"""
HybridNet - export.py
======================
Export trained model to ONNX and TFLite

Pipeline:
    best.pt  →  model.onnx  →  model_float32.tflite
                             →  model_int8.tflite  (quantized, smaller/faster)

Usage:
    # Export to ONNX only
    python export.py --weights runs/exp1/best.pt --format onnx

    # Export to TFLite (float32)
    python export.py --weights runs/exp1/best.pt --format tflite

    # Export to TFLite INT8 (quantized — best for mobile)
    python export.py --weights runs/exp1/best.pt --format tflite --int8 --data data.yaml

Author: Dev Kr Lahkar
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='HybridNet Export')
    parser.add_argument('--weights', type=str, required=True,    help='path to best.pt')
    parser.add_argument('--format',  type=str, default='tflite', choices=['onnx', 'tflite'], help='export format')
    parser.add_argument('--imgsz',   type=int, default=640,      help='input image size')
    parser.add_argument('--int8',    action='store_true',         help='INT8 quantization (TFLite only)')
    parser.add_argument('--data',    type=str, default=None,      help='data.yaml (needed for INT8 calibration)')
    parser.add_argument('--device',  type=str, default='cpu',     help='export device (cpu recommended)')
    return parser.parse_args()


# ─────────────────────────────────────────────
# Export Wrapper (makes model TFLite-friendly)
# ─────────────────────────────────────────────

class HybridNetExport(nn.Module):
    """
    Export wrapper for HybridNet.

    Outputs a single concatenated tensor instead of a list of tuples,
    which is required for ONNX/TFLite compatibility.

    Output shape: (1, total_predictions, 5 + num_classes)
    Format per prediction: [x, y, w, h, centerness, class0, class1, ...]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        predictions = self.model(x)

        outputs = []
        for box_pred, cls_pred, ctr_pred in predictions:
            B, _, H, W = box_pred.shape

            # Apply activations
            box = torch.sigmoid(box_pred)   # (B, 4, H, W)
            cls = torch.sigmoid(cls_pred)   # (B, C, H, W)
            ctr = torch.sigmoid(ctr_pred)   # (B, 1, H, W)

            # Reshape: (B, 4+1+C, H, W) → (B, H*W, 5+C)
            out = torch.cat([box, ctr, cls], dim=1)          # (B, 5+C, H, W)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, out.shape[1])

            outputs.append(out)

        # Concat all scales: (B, total_preds, 5+C)
        return torch.cat(outputs, dim=1)


# ─────────────────────────────────────────────
# Export to ONNX
# ─────────────────────────────────────────────

def export_onnx(model, imgsz, save_path, device):
    """Export model to ONNX format"""
    try:
        import onnx
    except ImportError:
        print("Installing onnx...")
        os.system("pip install onnx -q")
        import onnx

    export_model = HybridNetExport(model).to(device)
    export_model.eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)

    print(f"\nExporting to ONNX: {save_path}")
    torch.onnx.export(
        export_model,
        dummy,
        save_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,    # fixed size for TFLite compat
        do_constant_folding=True,
        verbose=False
    )

    # Verify
    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)

    size_mb = Path(save_path).stat().st_size / 1e6
    print(f"  ✅ ONNX export success → {save_path} ({size_mb:.1f} MB)")
    return save_path


# ─────────────────────────────────────────────
# Export to TFLite
# ─────────────────────────────────────────────

def export_tflite(onnx_path, save_dir, int8=False, data_yaml=None, imgsz=640):
    """
    Convert ONNX → TensorFlow SavedModel → TFLite

    Requires: onnx2tf
        pip install onnx2tf
    """
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf...")
        os.system("pip install onnx2tf onnx tensorflow -q")

    save_dir = Path(save_dir)
    tf_dir   = save_dir / 'tf_model'

    # ── Step 1: ONNX → TF SavedModel ──────────────
    print(f"\nConverting ONNX → TensorFlow SavedModel...")
    os.system(
        f"onnx2tf -i {onnx_path} -o {tf_dir} -nuo --non_verbose"
    )
    print(f"  ✅ TF SavedModel saved → {tf_dir}")

    # ── Step 2: TF SavedModel → TFLite ─────────────
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        print("\nApplying INT8 quantization (needs calibration data)...")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8

        # Calibration dataset from YOLO images
        def representative_dataset():
            import cv2, yaml
            if data_yaml is None:
                # Use random data if no dataset provided
                for _ in range(100):
                    data = np.random.randint(0, 256, (1, imgsz, imgsz, 3), dtype=np.uint8)
                    yield [data.astype(np.float32) / 255.0]
                return

            with open(data_yaml) as f:
                d = yaml.safe_load(f)
            img_dir = Path(data_yaml).parent / d.get('val', d.get('train', 'images/val'))
            imgs    = list(img_dir.rglob('*.jpg'))[:100]
            for p in imgs:
                img = cv2.imread(str(p))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                img = img.astype(np.float32) / 255.0
                yield [img[np.newaxis]]

        converter.representative_dataset = representative_dataset
        tflite_name = 'model_int8.tflite'
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_name = 'model_float32.tflite'

    tflite_bytes = converter.convert()
    tflite_path  = save_dir / tflite_name

    with open(tflite_path, 'wb') as f:
        f.write(tflite_bytes)

    size_mb = tflite_path.stat().st_size / 1e6
    print(f"  ✅ TFLite export success → {tflite_path} ({size_mb:.1f} MB)")
    return str(tflite_path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def export():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*45}")
    print(f"  HybridNet Export")
    print(f"{'='*45}")
    print(f"  Weights: {args.weights}")
    print(f"  Format : {args.format}")
    print(f"  Size   : {args.imgsz}x{args.imgsz}")
    if args.format == 'tflite':
        print(f"  INT8   : {args.int8}")
    print(f"{'='*45}\n")

    # ── Load model ──────────────────────────────────
    from model.hybridnet import build_model
    ckpt = torch.load(args.weights, map_location=device)
    num_classes = ckpt.get('num_classes', 80)
    model = build_model(num_classes=num_classes, weights=args.weights, device=str(device))
    model.eval()

    save_dir  = Path(args.weights).parent
    onnx_path = str(save_dir / 'model.onnx')

    # ── Export ──────────────────────────────────────
    if args.format == 'onnx':
        export_onnx(model, args.imgsz, onnx_path, device)

    elif args.format == 'tflite':
        # First export to ONNX, then convert
        export_onnx(model, args.imgsz, onnx_path, device)
        export_tflite(
            onnx_path, save_dir,
            int8=args.int8,
            data_yaml=args.data,
            imgsz=args.imgsz
        )

    print(f"\n✅ Export complete! Files in: {save_dir}")

    if args.format == 'tflite':
        print("\n📱 To use in Android/iOS:")
        print("   - Copy the .tflite file to your mobile app")
        print("   - Input : float32 tensor [1, 640, 640, 3] normalized 0-1")
        print("   - Output: [1, total_preds, 5+num_classes]")
        print("   - Apply NMS in app code (score_threshold, iou_threshold)")


if __name__ == '__main__':
    export()
