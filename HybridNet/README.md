# HybridNet 

> A novel lightweight anchor-free object detector optimized for mobile deployment (TFLite), trained on YOLO-format datasets.

---

##  Novel Contributions

| Module | Novelty |
|---|---|
| **CAB** (Context Attention Block) | Fuses channel + spatial + local context attention in one block |
| **AdaFPN** (Adaptive FPN) | Learnable weighted multi-scale feature fusion |
| **HybridLoss** | Uncertainty-weighted CIoU + Focal + Centerness loss |

---

##  Architecture

```
Input (640×640)
    ↓
LiteHybridNet (Backbone)
  Lightweight CNN + CAB Attention
    ↓ P3, P4, P5
AdaFPN (Neck)
  Adaptive Weighted Multi-Scale Fusion
    ↓ out3, out4, out5
AnchorFreeHead (Head)
  Per-pixel anchor-free prediction
    ↓
Boxes + Classes + Centerness
```

---

##  Project Structure

```
HybridNet/
├── model/
│   ├── backbone.py       ← LiteHybridNet + CAB
│   ├── neck.py           ← AdaFPN
│   ├── head.py           ← AnchorFreeHead
│   └── hybridnet.py      ← Full model
├── utils/
│   ├── dataloader.py     ← YOLO format loader
│   ├── loss.py           ← HybridLoss
│   ├── metrics.py        ← mAP evaluation
│   └── general.py        ← NMS, drawing, utils
├── train.py              ← Training script
├── val.py                ← Validation / mAP
├── detect.py             ← Inference
├── export.py             ← ONNX / TFLite export
├── data.yaml             ← Dataset config template
└── requirements.txt
```

---

##  Quick Start

### 1. Install
```bash
git clone https://github.com/YOUR_USERNAME/HybridNet
cd HybridNet
pip install -r requirements.txt
```

### 2. Prepare Dataset 
```
dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`data.yaml`:
```yaml
train: images/train
val:   images/val
nc: 3
names: ['cat', 'dog', 'person']
```

YOLO label format (`.txt`):
```
# class cx cy w h  (all normalized 0-1)
0 0.5 0.5 0.3 0.4
```

### 3. Train
```bash
python train.py --data dataset/data.yaml --epochs 100 --batch 16
```

### 4. Validate (mAP)
```bash
python val.py --weights runs/exp/best.pt --data dataset/data.yaml
```

### 5. Detect
```bash
# Image
python detect.py --weights runs/exp/best.pt --source image.jpg

# Folder
python detect.py --weights runs/exp/best.pt --source images/

# Video
python detect.py --weights runs/exp/best.pt --source video.mp4
```

### 6. Export to TFLite
```bash
# Float32
python export.py --weights runs/exp/best.pt --format tflite

# INT8 Quantized (fastest on mobile)
python export.py --weights runs/exp/best.pt --format tflite --int8 --data dataset/data.yaml
```

---

##  Train on Google Colab

```python
# Clone repo
!git clone https://github.com/YOUR_USERNAME/HybridNet
%cd HybridNet
!pip install -r requirements.txt

# Train (save to Google Drive so you don't lose progress)
!python train.py \
    --data /content/drive/MyDrive/dataset/data.yaml \
    --epochs 100 --batch 16 \
    --save-dir /content/drive/MyDrive/HybridNet_runs

# Resume if Colab disconnected
!python train.py \
    --weights /content/drive/MyDrive/HybridNet_runs/exp/last.pt \
    --data /content/drive/MyDrive/dataset/data.yaml \
    --epochs 100
```

---

##  Model Stats

| Property | Value |
|---|---|
| Input Size | 640×640 |
| Parameters | ~8-12M |
| Detection Scales | 3 |
| Annotation Format | YOLO (.txt) |
| Config Format | data.yaml |
| Export | ONNX, TFLite |
| Quantization | INT8 |

---

##  Mobile Deployment

After exporting to TFLite:
- Input: `float32 [1, 640, 640, 3]` normalized 0–1
- Output: `[1, total_predictions, 5 + num_classes]`
- Apply NMS in app code with score and IoU thresholds

---

##  License

MIT License — free to use for research and commercial projects.
