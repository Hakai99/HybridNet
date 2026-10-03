"""
HybridNet - detect.py
======================
Run inference on images, folders, or video.

Usage:
    # Single image
    python detect.py --weights runs/exp1/best.pt --source image.jpg

    # Folder of images
    python detect.py --weights runs/exp1/best.pt --source images/

    # Video file
    python detect.py --weights runs/exp1/best.pt --source video.mp4

    # Webcam
    python detect.py --weights runs/exp1/best.pt --source 0

Author: Dev Kr Lahkar
"""

import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def parse_args():
    parser = argparse.ArgumentParser(description='HybridNet Detect')
    parser.add_argument('--weights',    type=str,   required=True,      help='path to best.pt')
    parser.add_argument('--source',     type=str,   required=True,      help='image/folder/video/webcam(0)')
    parser.add_argument('--data',       type=str,   default=None,       help='data.yaml (for class names)')
    parser.add_argument('--imgsz',      type=int,   default=640,        help='inference size')
    parser.add_argument('--conf',       type=float, default=0.25,       help='confidence threshold')
    parser.add_argument('--iou',        type=float, default=0.45,       help='NMS IoU threshold')
    parser.add_argument('--save-dir',   type=str,   default='runs/detect', help='save results here')
    parser.add_argument('--name',       type=str,   default='exp',      help='run name')
    parser.add_argument('--device',     type=str,   default='',         help='cuda or cpu')
    parser.add_argument('--no-save',    action='store_true',            help='do not save results')
    parser.add_argument('--show',       action='store_true',            help='show results live')
    return parser.parse_args()


def get_device(device_str):
    if device_str == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def letterbox(img, size=640):
    """Resize with padding to maintain aspect ratio"""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h))

    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_h, size - new_h - pad_h,
        pad_w, size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_padded, ratio, (pad_w, pad_h)


def preprocess(img_bgr, size=640, device='cpu'):
    """Preprocess image for model input"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, ratio, pad = letterbox(img_rgb, size)

    tensor = torch.from_numpy(img_lb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    return tensor, ratio, pad


def postprocess(boxes, scores, classes, orig_shape, ratio, pad, img_size):
    """
    Convert model output back to original image coordinates.

    boxes are in normalized [0,1] coords at img_size.
    We scale them back to orig_shape pixel coords.
    """
    if boxes.shape[0] == 0:
        return boxes, scores, classes

    # Scale from normalized to img_size pixels
    boxes = boxes.clone()
    boxes[:, 0] *= img_size
    boxes[:, 1] *= img_size
    boxes[:, 2] *= img_size
    boxes[:, 3] *= img_size

    # Remove letterbox padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, 2] -= pad[0]
    boxes[:, 3] -= pad[1]

    # Scale to original image size
    boxes /= ratio

    # Clip to image bounds
    oh, ow = orig_shape
    boxes[:, 0] = boxes[:, 0].clamp(0, ow)
    boxes[:, 1] = boxes[:, 1].clamp(0, oh)
    boxes[:, 2] = boxes[:, 2].clamp(0, ow)
    boxes[:, 3] = boxes[:, 3].clamp(0, oh)

    return boxes, scores, classes


def draw_boxes(img, boxes, scores, classes, class_names):
    """Draw detections on image"""
    COLORS = [
        (255,56,56),(255,157,151),(255,112,31),(255,178,29),(207,210,49),
        (72,249,10),(146,204,23),(61,219,134),(26,147,52),(0,212,187),
        (44,153,168),(0,194,255),(52,69,147),(100,115,255),(0,24,236),
        (132,56,255),(82,0,133),(203,56,255),(255,149,200),(255,55,199),
    ]
    for box, score, cls in zip(boxes, scores, classes):
        cls   = int(cls)
        color = COLORS[cls % len(COLORS)]
        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name  = class_names[cls] if cls < len(class_names) else f"cls{cls}"
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return img


def get_source_list(source):
    """Return list of image paths, or video path, or webcam id"""
    source = str(source)

    # Webcam
    if source.isdigit():
        return 'webcam', int(source)

    p = Path(source)

    # Video
    if p.suffix.lower() in VID_EXTENSIONS:
        return 'video', source

    # Single image
    if p.suffix.lower() in IMG_EXTENSIONS:
        return 'images', [source]

    # Folder
    if p.is_dir():
        imgs = sorted([str(f) for f in p.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
        return 'images', imgs

    raise ValueError(f"Unknown source: {source}")


def load_class_names(args):
    """Load class names from data.yaml or use generic names"""
    if args.data and Path(args.data).exists():
        with open(args.data) as f:
            import yaml
            data = yaml.safe_load(f)
        return data.get('names', [])
    # Try to infer from checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'class_names' in ckpt:
        return ckpt['class_names']
    nc = ckpt.get('num_classes', 80)
    return [f"class_{i}" for i in range(nc)]


def detect():
    args   = parse_args()
    device = get_device(args.device)

    # ── Save dir ────────────────────────────────────
    save_dir = Path(args.save_dir) / args.name
    if not args.no_save:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ───────────────────────────────────────
    print(f"Loading model: {args.weights}")
    from model.hybridnet import build_model

    ckpt = torch.load(args.weights, map_location=device)
    num_classes = ckpt.get('num_classes', 80)
    model = build_model(num_classes=num_classes, weights=args.weights, device=str(device))
    model.eval()

    # ── Class names ─────────────────────────────────
    class_names = load_class_names(args)
    print(f"Classes: {class_names}\n")

    # ── Source ──────────────────────────────────────
    source_type, source_data = get_source_list(args.source)

    # ── Run detection ───────────────────────────────
    if source_type == 'images':
        for img_path in source_data:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  Warning: could not read {img_path}")
                continue

            orig_shape = img_bgr.shape[:2]
            tensor, ratio, pad = preprocess(img_bgr, args.imgsz, device)

            with torch.no_grad():
                boxes, scores, classes = model.predict(
                    tensor, conf_thresh=args.conf, iou_thresh=args.iou
                )

            boxes, scores, classes = postprocess(
                boxes, scores, classes, orig_shape, ratio, pad, args.imgsz
            )

            n = len(boxes)
            print(f"  {img_path} → {n} detection{'s' if n != 1 else ''}")

            # Annotate
            result = draw_boxes(img_bgr.copy(), boxes, scores, classes, class_names)

            if args.show:
                cv2.imshow('HybridNet', result)
                cv2.waitKey(0)

            if not args.no_save:
                out_path = save_dir / Path(img_path).name
                cv2.imwrite(str(out_path), result)

        cv2.destroyAllWindows()

    elif source_type in ('video', 'webcam'):
        cap = cv2.VideoCapture(source_data)
        assert cap.isOpened(), f"Failed to open: {source_data}"

        # Video writer
        writer = None
        if not args.no_save and source_type == 'video':
            fps   = cap.get(cv2.CAP_PROP_FPS)
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = save_dir / (Path(str(source_data)).stem + '_out.mp4')
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_shape = frame.shape[:2]
            tensor, ratio, pad = preprocess(frame, args.imgsz, device)

            with torch.no_grad():
                boxes, scores, classes = model.predict(
                    tensor, conf_thresh=args.conf, iou_thresh=args.iou
                )

            boxes, scores, classes = postprocess(
                boxes, scores, classes, orig_shape, ratio, pad, args.imgsz
            )

            result = draw_boxes(frame.copy(), boxes, scores, classes, class_names)

            frame_count += 1
            print(f"\r  Frame {frame_count} | {len(boxes)} detections", end='')

            if args.show:
                cv2.imshow('HybridNet', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if writer:
                writer.write(result)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print()

    if not args.no_save:
        print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    detect()
