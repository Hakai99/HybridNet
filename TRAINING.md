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

## Cell 4 — Install Requirements

```python
import os
os.chdir('/content/HybridNet/HybridNet')

os.system('pip install -r requirements.txt -q')
print("Done!")
```

---

## Cell 5 — Check Your Dataset

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

## Cell 6 — Test Model Forward Pass

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

## Cell 7 — Smart Training (10 epochs per unit with popup)

> This is the main training cell.
> It trains 10 epochs at a time, then asks you what to do next.
> You need at least 200 epochs total for good results.
> Each day you can run this cell multiple times to accumulate epochs.

```python
import os
import sys
import time
import torch
import yaml
from IPython.display import display, HTML
import ipywidgets as widgets

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

# ── CHANGE THESE ─────────────────────────────
DATASET_PATH  = '/content/drive/MyDrive/YOUR_DATASET'   # ← your dataset
SAVE_DIR      = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME      = 'run_v1'      # ← keep same name every session!
EPOCHS_PER_UNIT = 10          # train this many epochs per unit
TOTAL_TARGET    = 200         # your total target epochs
BATCH           = 8           # 8 for free Colab
IMG_SIZE        = 640
# ─────────────────────────────────────────────

DATA_YAML = DATASET_PATH + '/data.yaml'
RUN_DIR   = SAVE_DIR + '/' + RUN_NAME
LAST_PT   = RUN_DIR + '/last.pt'
LOG_FILE  = RUN_DIR + '/train_log.txt'

os.makedirs(RUN_DIR, exist_ok=True)

def get_current_epoch():
    """Read last completed epoch from log file"""
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE) as f:
        lines = [l for l in f.read().strip().splitlines() if l and not l.startswith('epoch')]
    if not lines:
        return 0
    last = lines[-1].split(',')
    try:
        return int(last[0])
    except Exception:
        return 0

def train_unit(from_epoch, to_epoch):
    """Train one unit of epochs"""
    print("\n" + "="*50)
    print("Training epochs " + str(from_epoch+1) + " → " + str(to_epoch))
    print("="*50 + "\n")

    if os.path.exists(LAST_PT):
        cmd = (
            'python train.py'
            ' --data ' + DATA_YAML +
            ' --weights ' + LAST_PT +
            ' --epochs ' + str(to_epoch) +
            ' --batch ' + str(BATCH) +
            ' --imgsz ' + str(IMG_SIZE) +
            ' --device cuda' +
            ' --save-dir ' + SAVE_DIR +
            ' --name ' + RUN_NAME
        )
    else:
        cmd = (
            'python train.py'
            ' --data ' + DATA_YAML +
            ' --epochs ' + str(to_epoch) +
            ' --batch ' + str(BATCH) +
            ' --imgsz ' + str(IMG_SIZE) +
            ' --device cuda' +
            ' --save-dir ' + SAVE_DIR +
            ' --name ' + RUN_NAME
        )

    os.system(cmd)

def show_popup(current_epoch):
    """Show popup with 3 options"""
    print("\n" + "="*50)
    print("CHECKPOINT REACHED — Epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))
    print("Weights saved to Drive: " + LAST_PT)
    print("="*50)

    # Create buttons
    btn_continue = widgets.Button(
        description='Continue Training',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )
    btn_stop = widgets.Button(
        description='Stop & Save',
        button_style='danger',
        layout=widgets.Layout(width='200px', height='40px')
    )
    btn_later = widgets.Button(
        description='Resume Later',
        button_style='warning',
        layout=widgets.Layout(width='200px', height='40px')
    )

    choice = {'value': None}

    def on_continue(b):
        choice['value'] = 'continue'
        btn_continue.disabled = True
        btn_stop.disabled     = True
        btn_later.disabled    = True
        print("\nContinuing training...")

    def on_stop(b):
        choice['value'] = 'stop'
        btn_continue.disabled = True
        btn_stop.disabled     = True
        btn_later.disabled    = True
        print("\nStopped! Weights saved to Drive.")
        print("Resume tomorrow with Cell 8 (Resume Later).")

    def on_later(b):
        choice['value'] = 'later'
        btn_continue.disabled = True
        btn_stop.disabled     = True
        btn_later.disabled    = True
        print("\nCheckpoint saved! Resume tomorrow with Cell 8.")
        print("Your progress: epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))

    btn_continue.on_click(on_continue)
    btn_stop.on_click(on_stop)
    btn_later.on_click(on_later)

    display(widgets.HBox([btn_continue, btn_stop, btn_later]))

    # Wait for user to click
    while choice['value'] is None:
        time.sleep(0.5)

    return choice['value']

# ── MAIN TRAINING LOOP ──────────────────────
current_epoch = get_current_epoch()

print("Current progress: epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))

if current_epoch >= TOTAL_TARGET:
    print("Training already complete! " + str(TOTAL_TARGET) + " epochs done.")
    print("Use Cell 9 to validate your model.")
else:
    while current_epoch < TOTAL_TARGET:
        next_epoch = min(current_epoch + EPOCHS_PER_UNIT, TOTAL_TARGET)

        # Train one unit
        train_unit(current_epoch, next_epoch)

        # Update epoch count
        current_epoch = get_current_epoch()

        print("\nProgress: " + str(current_epoch) + "/" + str(TOTAL_TARGET) + " epochs done")

        # Check if reached target
        if current_epoch >= TOTAL_TARGET:
            print("\n" + "="*50)
            print("ALL " + str(TOTAL_TARGET) + " EPOCHS COMPLETE!")
            print("Best weights: " + RUN_DIR + "/best.pt")
            print("="*50)
            break

        # Show popup — what to do next?
        action = show_popup(current_epoch)

        if action == 'continue':
            continue    # loop back and train next unit
        else:
            break       # stop training
```

---

## Cell 8 — Resume Later (Run this next session)

> Run Cells 1–4 first, then run this cell to resume from where you left off

```python
import os
import sys
import time
import torch
import yaml
import ipywidgets as widgets
from IPython.display import display

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

# ── CHANGE THESE ─────────────────────────────
DATASET_PATH    = '/content/drive/MyDrive/YOUR_DATASET'   # ← your dataset
SAVE_DIR        = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME        = 'run_v1'      # ← SAME name as Cell 7!
EPOCHS_PER_UNIT = 10
TOTAL_TARGET    = 200
BATCH           = 8
IMG_SIZE        = 640
# ─────────────────────────────────────────────

DATA_YAML = DATASET_PATH + '/data.yaml'
RUN_DIR   = SAVE_DIR + '/' + RUN_NAME
LAST_PT   = RUN_DIR + '/last.pt'
LOG_FILE  = RUN_DIR + '/train_log.txt'

def get_current_epoch():
    if not os.path.exists(LOG_FILE):
        return 0
    with open(LOG_FILE) as f:
        lines = [l for l in f.read().strip().splitlines() if l and not l.startswith('epoch')]
    if not lines:
        return 0
    try:
        return int(lines[-1].split(',')[0])
    except Exception:
        return 0

def train_unit(to_epoch):
    cmd = (
        'python train.py'
        ' --data ' + DATA_YAML +
        ' --weights ' + LAST_PT +
        ' --epochs ' + str(to_epoch) +
        ' --batch ' + str(BATCH) +
        ' --imgsz ' + str(IMG_SIZE) +
        ' --device cuda' +
        ' --save-dir ' + SAVE_DIR +
        ' --name ' + RUN_NAME
    )
    os.system(cmd)

def show_popup(current_epoch):
    print("\n" + "="*50)
    print("CHECKPOINT — Epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))
    print("="*50)

    btn_continue = widgets.Button(description='Continue Training', button_style='success', layout=widgets.Layout(width='200px', height='40px'))
    btn_stop     = widgets.Button(description='Stop & Save',       button_style='danger',  layout=widgets.Layout(width='200px', height='40px'))
    btn_later    = widgets.Button(description='Resume Later',      button_style='warning', layout=widgets.Layout(width='200px', height='40px'))

    choice = {'value': None}

    def on_continue(b):
        choice['value'] = 'continue'
        btn_continue.disabled = btn_stop.disabled = btn_later.disabled = True
        print("Continuing...")

    def on_stop(b):
        choice['value'] = 'stop'
        btn_continue.disabled = btn_stop.disabled = btn_later.disabled = True
        print("Stopped! Weights saved to Drive.")

    def on_later(b):
        choice['value'] = 'later'
        btn_continue.disabled = btn_stop.disabled = btn_later.disabled = True
        print("Saved! Resume tomorrow with this cell.")

    btn_continue.on_click(on_continue)
    btn_stop.on_click(on_stop)
    btn_later.on_click(on_later)
    display(widgets.HBox([btn_continue, btn_stop, btn_later]))

    while choice['value'] is None:
        time.sleep(0.5)

    return choice['value']

# ── RESUME FROM LAST CHECKPOINT ─────────────
current_epoch = get_current_epoch()

if not os.path.exists(LAST_PT):
    print("No checkpoint found! Run Cell 7 first to start training.")
else:
    print("Resuming from epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))
    print("Checkpoint: " + LAST_PT)

    if current_epoch >= TOTAL_TARGET:
        print("Training already complete!")
    else:
        while current_epoch < TOTAL_TARGET:
            next_epoch = min(current_epoch + EPOCHS_PER_UNIT, TOTAL_TARGET)
            train_unit(next_epoch)
            current_epoch = get_current_epoch()
            print("Progress: " + str(current_epoch) + "/" + str(TOTAL_TARGET))

            if current_epoch >= TOTAL_TARGET:
                print("\nALL " + str(TOTAL_TARGET) + " EPOCHS COMPLETE!")
                print("Best weights: " + RUN_DIR + "/best.pt")
                break

            action = show_popup(current_epoch)
            if action != 'continue':
                break
```

---

## Cell 9 — Check Weights & Progress

```python
import os

SAVE_DIR = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME = 'run_v1'
run_path = SAVE_DIR + '/' + RUN_NAME
log_file = run_path + '/train_log.txt'

print("=" * 45)
print("Training Progress")
print("=" * 45)

# Check weights
for f in ['best.pt', 'last.pt']:
    path = run_path + '/' + f
    if os.path.exists(path):
        size = round(os.path.getsize(path) / 1e6, 1)
        print("FOUND " + f + " — " + str(size) + " MB")
    else:
        print("MISSING " + f)

# Show epoch progress
if os.path.exists(log_file):
    with open(log_file) as f:
        lines = [l for l in f.read().strip().splitlines() if l and not l.startswith('epoch')]
    print("\nEpochs completed: " + str(len(lines)))
    if lines:
        print("Last entry: " + lines[-1])
else:
    print("No training log found yet.")

print("=" * 45)
```

---

## Cell 10 — Validate (mAP Score)

```python
import os
os.chdir('/content/HybridNet/HybridNet')

BEST_PT   = '/content/drive/MyDrive/HybridNet_runs/run_v1/best.pt'
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

## Cell 11 — Load Weights for Transfer Learning

```python
import os
import sys
import torch

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

from model.hybridnet import build_model

WEIGHTS     = '/content/drive/MyDrive/HybridNet_runs/run_v1/best.pt'
NEW_CLASSES = 20
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

**How the 10-epoch unit system works:**
- Each unit trains 10 epochs then pauses
- A popup appears with 3 buttons:
  - **Continue Training** → immediately trains next 10 epochs
  - **Stop & Save** → saves checkpoint and stops
  - **Resume Later** → saves checkpoint, come back tomorrow with Cell 8
- Weights save to Drive automatically after every epoch — you never lose progress!

**Session flow:**
```
Day 1: Cells 1,2,3,4,5,6,7 → train 10 epochs → popup → continue → 20 epochs → Resume Later
Day 2: Cells 1,2,3,4 → Cell 8 → resumes from epoch 20 → train more
...repeat until 200 epochs!
```

**Things to change:**
- `YOUR_DATASET` → your dataset folder name in Drive
- `RUN_NAME` → name for this training run (keep SAME every session!)
- `TOTAL_TARGET` → your target total epochs (default 200)
- `EPOCHS_PER_UNIT` → how many epochs per unit (default 10)

**Recommended batch sizes:**
- Free Colab → `BATCH = 8`
- Colab Pro  → `BATCH = 16`

**Weights saved automatically:**
- `best.pt` → best weights overall
- `last.pt` → last epoch checkpoint (used for resume)

---

## Questions or Issues?
Open an issue at: https://github.com/Hakai99/HybridNet/issues
