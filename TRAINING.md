# HybridNet — Training Guide

## Before You Start
- Open Google Colab
- Go to **Runtime → Change runtime type → GPU (T4)**
- Make sure your dataset is in **Google Drive** in YOLO format

---

## Dataset Format Required

Your dataset must follow this structure in Google Drive:

```
your_dataset/
├── data.yaml
├── images/
│   ├── train/     ← training images (.jpg or .png)
│   └── val/       ← validation images
└── labels/
    ├── train/     ← YOLO .txt annotation files
    └── val/
```

Your `data.yaml` must look like this:

```yaml
train: images/train
val: images/val
nc: 3
names:
  - cat
  - dog
  - person
```

> Replace `nc` with your number of classes and `names` with your class names.

---

## Cell 1 — Check GPU

```python
import torch

print("=" * 40)
print("GPU Check")
print("=" * 40)
if torch.cuda.is_available():
    print("GPU: " + torch.cuda.get_device_name(0))
    print("CUDA: " + torch.version.cuda)
    print("Status: READY!")
else:
    print("NO GPU FOUND!")
    print("Go to Runtime > Change runtime type > GPU")
print("=" * 40)
```

---

## Cell 2 — Mount Google Drive

```python
import os
from google.colab import drive

os.chdir('/content')

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')
    print("Drive mounted!")
else:
    print("Drive already mounted!")

print("Drive contents:")
print(os.listdir('/content/drive/MyDrive'))
```

---

## Cell 3 — Clone HybridNet

```python
import os

os.chdir('/content')

if os.path.exists('/content/HybridNet'):
    import shutil
    shutil.rmtree('/content/HybridNet')

os.system('git clone https://github.com/Hakai99/HybridNet.git /content/HybridNet')
os.chdir('/content/HybridNet/HybridNet')

print("Files in repo:")
print(os.listdir('.'))
```

---

## Cell 4 — Fix DataLoader for Colab

```python
import os
os.chdir('/content/HybridNet/HybridNet')

with open('utils/dataloader.py', 'r') as f:
    content = f.read()

content = content.replace('num_workers=workers', 'num_workers=0')
content = content.replace('num_workers=4', 'num_workers=0')
content = content.replace('num_workers=2', 'num_workers=0')
content = content.replace('pin_memory=True', 'pin_memory=False')

with open('utils/dataloader.py', 'w') as f:
    f.write(content)

print("DataLoader fixed!")
os.system('grep -n "num_workers\|pin_memory" utils/dataloader.py')
```

---

## Cell 5 — Install Requirements

```python
import os
os.chdir('/content/HybridNet/HybridNet')

os.system('pip install -r requirements.txt -q')
print("Done!")
```

---

## Cell 6 — Check Your Dataset

> Change `DATASET_PATH` to your dataset folder in Google Drive

```python
import os
import yaml

DATASET_PATH = '/content/drive/MyDrive/YOUR_DATASET'   # ← CHANGE THIS
DATA_YAML    = DATASET_PATH + '/data.yaml'

print("=" * 45)
print("Dataset Check")
print("=" * 45)

if not os.path.exists(DATASET_PATH):
    print("MISSING: Dataset folder not found!")
    print("Check your DATASET_PATH above!")
else:
    print("FOUND: " + DATASET_PATH)

    if os.path.exists(DATA_YAML):
        with open(DATA_YAML) as f:
            data = yaml.safe_load(f)
        print("data.yaml found!")
        print("Classes: " + str(data['nc']))
        print("Names: " + str(data['names']))
    else:
        print("MISSING: data.yaml not found!")

    for folder in ['images', 'labels']:
        for split in ['train', 'val']:
            path = DATASET_PATH + '/' + folder + '/' + split
            if os.path.exists(path):
                count = len(os.listdir(path))
                print(folder + '/' + split + ': ' + str(count) + ' files')
            else:
                print("MISSING: " + folder + '/' + split)

print("=" * 45)
```

---

## Cell 7 — Test Model

```python
import os
import sys
import torch
import yaml

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

from model.hybridnet import HybridNet

DATA_YAML = '/content/drive/MyDrive/YOUR_DATASET/data.yaml'   # ← CHANGE THIS

with open(DATA_YAML) as f:
    data = yaml.safe_load(f)

NUM_CLASSES = data['nc']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = HybridNet(num_classes=NUM_CLASSES).to(device)
model.info()

dummy = torch.randn(1, 3, 640, 640).to(device)
preds = model(dummy)
print("Forward pass OK!")
```

---

## Cell 8 — Train From Scratch

> Change `DATASET_PATH` and `RUN_NAME` before running

```python
import os
os.chdir('/content/HybridNet/HybridNet')

DATASET_PATH = '/content/drive/MyDrive/YOUR_DATASET'   # ← CHANGE THIS
DATA_YAML    = DATASET_PATH + '/data.yaml'
SAVE_DIR     = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME     = 'run_v1'    # ← give your run a name
EPOCHS       = 30
BATCH        = 8            # use 8 for free Colab
IMG_SIZE     = 640

os.makedirs(SAVE_DIR, exist_ok=True)

os.system(
    'python train.py'
    ' --data ' + DATA_YAML +
    ' --epochs ' + str(EPOCHS) +
    ' --batch ' + str(BATCH) +
    ' --imgsz ' + str(IMG_SIZE) +
    ' --device cuda' +
    ' --save-dir ' + SAVE_DIR +
    ' --name ' + RUN_NAME
)
```

---

## Cell 9 — Resume If Colab Disconnected

> Run this instead of Cell 8 if your session disconnected

```python
import os
os.chdir('/content/HybridNet/HybridNet')

DATASET_PATH = '/content/drive/MyDrive/YOUR_DATASET'   # ← CHANGE THIS
DATA_YAML    = DATASET_PATH + '/data.yaml'
SAVE_DIR     = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME     = 'run_v1'    # ← same name as Cell 8!
EPOCHS       = 30
BATCH        = 8
IMG_SIZE     = 640

LAST_PT = SAVE_DIR + '/' + RUN_NAME + '/last.pt'

if not os.path.exists(LAST_PT):
    print("No last.pt found! Run Cell 8 first!")
else:
    print("Resuming from: " + LAST_PT)
    os.system(
        'python train.py'
        ' --data ' + DATA_YAML +
        ' --weights ' + LAST_PT +
        ' --epochs ' + str(EPOCHS) +
        ' --batch ' + str(BATCH) +
        ' --imgsz ' + str(IMG_SIZE) +
        ' --device cuda' +
        ' --save-dir ' + SAVE_DIR +
        ' --name ' + RUN_NAME
    )
```

---

## Cell 10 — Check Weights Saved

```python
import os

SAVE_DIR = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME = 'run_v1'   # ← same name as above
run_path = SAVE_DIR + '/' + RUN_NAME

print("=" * 45)
for f in ['best.pt', 'last.pt']:
    path = run_path + '/' + f
    if os.path.exists(path):
        size = round(os.path.getsize(path) / 1e6, 1)
        print("FOUND " + f + " — " + str(size) + " MB")
    else:
        print("MISSING " + f)
print("=" * 45)
print("Location: " + run_path)
```

---

## Cell 11 — Validate (mAP Score)

```python
import os
os.chdir('/content/HybridNet/HybridNet')

BEST_PT   = '/content/drive/MyDrive/HybridNet_runs/run_v1/best.pt'   # ← CHANGE RUN NAME
DATA_YAML = '/content/drive/MyDrive/YOUR_DATASET/data.yaml'           # ← CHANGE THIS

if not os.path.exists(BEST_PT):
    print("best.pt not found! Train first!")
else:
    os.system(
        'python val.py'
        ' --weights ' + BEST_PT +
        ' --data ' + DATA_YAML +
        ' --batch 8'
    )
```

---

## Cell 12 — Load Weights for Transfer Learning

```python
import os
import sys
import torch

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

from model.hybridnet import build_model

WEIGHTS     = '/content/drive/MyDrive/HybridNet_runs/run_v1/best.pt'   # ← CHANGE
NEW_CLASSES = 20    # ← number of classes in your NEW dataset
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model(
    num_classes=NEW_CLASSES,
    weights=WEIGHTS,
    device=device
)

model.info()
print("Weights loaded!")
print("Ready for transfer learning!")
```

---

## Important Notes

**If Colab disconnects:**
- Run Cells 1, 2, 3, 4, 5 again
- Then run Cell 9 (resume) — NOT Cell 8

**Things to change in each cell:**
- `YOUR_DATASET` → your dataset folder name in Drive
- `RUN_NAME` → name for your training run (keep same for resume)

**Recommended batch sizes:**
- Free Colab → `BATCH = 8`
- Colab Pro  → `BATCH = 16`

**Weights are auto-saved to Drive after every epoch:**
- `best.pt` → best weights overall
- `last.pt` → last epoch (use this to resume)

---

## Questions or Issues?
Open an issue at: https://github.com/Hakai99/HybridNet/issues
