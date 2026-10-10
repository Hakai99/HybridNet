# HybridNet — Training Guide

## Before You Start
- Open Google Colab
- Go to **Runtime → Change runtime type → GPU (T4)**
- Make sure your dataset is in **Google Drive** in YOLO format

---

## Dataset Format Required

```
your_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

```yaml
train: images/train
val: images/val
nc: 20
names:
  - aeroplane
  - bicycle
  - bird
  - boat
  - bottle
  - bus
  - car
  - cat
  - chair
  - cow
  - diningtable
  - dog
  - horse
  - motorbike
  - person
  - pottedplant
  - sheep
  - sofa
  - train
  - tvmonitor
```

---

## Cell 1 — Check GPU

```python
import torch

print("=" * 40)
if torch.cuda.is_available():
    print("GPU: " + torch.cuda.get_device_name(0))
    print("CUDA: " + torch.version.cuda)
    print("Status: READY!")
else:
    print("NO GPU — Go to Runtime > Change runtime type > GPU")
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
    print("Mounted!")
else:
    print("Already mounted!")

print(os.listdir('/content/drive/MyDrive'))
```

---

## Cell 3 — Clone HybridNet

```python
import os, shutil

os.chdir('/content')

if os.path.exists('/content/HybridNet'):
    shutil.rmtree('/content/HybridNet')

os.system('git clone https://github.com/Hakai99/HybridNet.git /content/HybridNet')
os.chdir('/content/HybridNet/HybridNet')

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

## Cell 5 — Check Dataset

```python
import os, yaml

DATASET_PATH = '/content/drive/MyDrive/hybrid_test'   # ← CHANGE THIS
DATA_YAML    = DATASET_PATH + '/data.yaml'

print("=" * 45)
if not os.path.exists(DATASET_PATH):
    print("MISSING: " + DATASET_PATH)
else:
    print("FOUND: " + DATASET_PATH)
    if os.path.exists(DATA_YAML):
        with open(DATA_YAML) as f:
            d = yaml.safe_load(f)
        print("Classes: " + str(d['nc']))
        print("Names:   " + str(d['names']))
    for folder in ['images', 'labels']:
        for split in ['train', 'val']:
            p = DATASET_PATH + '/' + folder + '/' + split
            if os.path.exists(p):
                print(folder + '/' + split + ': ' + str(len(os.listdir(p))) + ' files')
            else:
                print("MISSING: " + folder + '/' + split)
print("=" * 45)
```

---

## Cell 6 — Test Model Forward Pass

```python
import os, sys, torch, yaml

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

from model.hybridnet import HybridNet

DATA_YAML = '/content/drive/MyDrive/hybrid_test/data.yaml'   # ← CHANGE THIS

with open(DATA_YAML) as f:
    d = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = HybridNet(num_classes=d['nc']).to(device)
model.info()

preds = model(torch.randn(1, 3, 640, 640).to(device))
print("Forward pass OK!")
```

---

## Cell 7 — Smart Training (First Session)

> Trains 10 epochs at a time → popup after each unit → saves to Drive every epoch.
> Weights: `hybridmain.pt` (best) and `hybridlast.pt` (latest).

```python
import os, sys, time, torch, subprocess, ipywidgets as widgets
from IPython.display import display

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

DATASET_PATH    = '/content/drive/MyDrive/hybrid_test'   # ← CHANGE THIS
SAVE_DIR        = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME        = 'run_v1'
EPOCHS_PER_UNIT = 20
TOTAL_TARGET    = 800
BATCH           = 8
IMG_SIZE        = 640

DATA_YAML  = DATASET_PATH + '/data.yaml'
RUN_DIR    = SAVE_DIR + '/' + RUN_NAME
LAST_PT    = RUN_DIR + '/hybridlast.pt'
LOG_FILE   = RUN_DIR + '/train_log.txt'

os.makedirs(RUN_DIR, exist_ok=True)

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
    print("\n" + "="*50)
    print("Training to epoch " + str(to_epoch))
    print("="*50 + "\n")
    cmd = [
        'python', 'train.py',
        '--data',     DATA_YAML,
        '--epochs',   str(to_epoch),
        '--batch',    str(BATCH),
        '--imgsz',    str(IMG_SIZE),
        '--device',   'cuda',
        '--save-dir', SAVE_DIR,
        '--name',     RUN_NAME
    ]
    if os.path.exists(LAST_PT):
        cmd += ['--weights', LAST_PT]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd='/content/HybridNet/HybridNet')
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

def show_popup(current_epoch):
    print("\n" + "="*50)
    print("CHECKPOINT — Epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))
    print("="*50)

    b1 = widgets.Button(description='Continue Training', button_style='success', layout=widgets.Layout(width='200px', height='40px'))
    b2 = widgets.Button(description='Stop & Save',       button_style='danger',  layout=widgets.Layout(width='200px', height='40px'))
    b3 = widgets.Button(description='Resume Later',      button_style='warning', layout=widgets.Layout(width='200px', height='40px'))

    choice = {'v': None}

    def f1(b): choice['v'] = 'continue'; b1.disabled = b2.disabled = b3.disabled = True; print("Continuing...")
    def f2(b): choice['v'] = 'stop';     b1.disabled = b2.disabled = b3.disabled = True; print("Stopped. Weights saved to Drive.")
    def f3(b): choice['v'] = 'later';    b1.disabled = b2.disabled = b3.disabled = True; print("Saved. Resume with Cell 8.")

    b1.on_click(f1); b2.on_click(f2); b3.on_click(f3)
    display(widgets.HBox([b1, b2, b3]))

    while choice['v'] is None:
        time.sleep(0.5)
    return choice['v']

current_epoch = get_current_epoch()
print("Progress: epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))

if current_epoch >= TOTAL_TARGET:
    print("Already complete! Use Cell 9 to validate.")
else:
    while current_epoch < TOTAL_TARGET:
        next_epoch = min(current_epoch + EPOCHS_PER_UNIT, TOTAL_TARGET)
        train_unit(next_epoch)
        current_epoch = get_current_epoch()
        print("Progress: " + str(current_epoch) + "/" + str(TOTAL_TARGET))

        if current_epoch >= TOTAL_TARGET:
            print("\nALL " + str(TOTAL_TARGET) + " EPOCHS COMPLETE!")
            print("Best weights: " + RUN_DIR + "/hybridmain.pt")
            break

        action = show_popup(current_epoch)
        if action != 'continue':
            break
```

---

## Cell 8 — Resume Later (Next Session)

> Run Cells 1–4 first, then run this cell.

```python
import os, sys, time, torch, subprocess, ipywidgets as widgets
from IPython.display import display

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

DATASET_PATH    = '/content/drive/MyDrive/hybrid_test'   # ← CHANGE THIS
SAVE_DIR        = '/content/drive/MyDrive/HybridNet_runs'
RUN_NAME        = 'run_v1'
EPOCHS_PER_UNIT = 20
TOTAL_TARGET    = 800
BATCH           = 8
IMG_SIZE        = 640

DATA_YAML  = DATASET_PATH + '/data.yaml'
RUN_DIR    = SAVE_DIR + '/' + RUN_NAME
LAST_PT    = RUN_DIR + '/hybridlast.pt'
LOG_FILE   = RUN_DIR + '/train_log.txt'

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
    cmd = [
        'python', 'train.py',
        '--data',     DATA_YAML,
        '--weights',  LAST_PT,
        '--epochs',   str(to_epoch),
        '--batch',    str(BATCH),
        '--imgsz',    str(IMG_SIZE),
        '--device',   'cuda',
        '--save-dir', SAVE_DIR,
        '--name',     RUN_NAME
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd='/content/HybridNet/HybridNet')
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

def show_popup(current_epoch):
    print("\n" + "="*50)
    print("CHECKPOINT — Epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))
    print("="*50)

    b1 = widgets.Button(description='Continue Training', button_style='success', layout=widgets.Layout(width='200px', height='40px'))
    b2 = widgets.Button(description='Stop & Save',       button_style='danger',  layout=widgets.Layout(width='200px', height='40px'))
    b3 = widgets.Button(description='Resume Later',      button_style='warning', layout=widgets.Layout(width='200px', height='40px'))

    choice = {'v': None}

    def f1(b): choice['v'] = 'continue'; b1.disabled = b2.disabled = b3.disabled = True; print("Continuing...")
    def f2(b): choice['v'] = 'stop';     b1.disabled = b2.disabled = b3.disabled = True; print("Stopped. Weights saved.")
    def f3(b): choice['v'] = 'later';    b1.disabled = b2.disabled = b3.disabled = True; print("Saved. Resume tomorrow.")

    b1.on_click(f1); b2.on_click(f2); b3.on_click(f3)
    display(widgets.HBox([b1, b2, b3]))

    while choice['v'] is None:
        time.sleep(0.5)
    return choice['v']

current_epoch = get_current_epoch()

if not os.path.exists(LAST_PT):
    print("No checkpoint found! Run Cell 7 first.")
else:
    print("Resuming from epoch " + str(current_epoch) + "/" + str(TOTAL_TARGET))

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
                print("Best: " + RUN_DIR + "/hybridmain.pt")
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
for f in ['hybridmain.pt', 'hybridlast.pt']:
    path = run_path + '/' + f
    if os.path.exists(path):
        size = round(os.path.getsize(path) / 1e6, 1)
        print("FOUND " + f + " — " + str(size) + " MB")
    else:
        print("MISSING " + f)

if os.path.exists(log_file):
    with open(log_file) as f:
        lines = [l for l in f.read().strip().splitlines() if l and not l.startswith('epoch')]
    print("Epochs completed: " + str(len(lines)))
    if lines:
        print("Last: " + lines[-1])
print("=" * 45)
```

---

## Cell 10 — Validate (mAP Score)

```python
import os, subprocess
os.chdir('/content/HybridNet/HybridNet')

BEST_PT   = '/content/drive/MyDrive/HybridNet_runs/run_v1/hybridmain.pt'
DATA_YAML = '/content/drive/MyDrive/hybrid_test/data.yaml'   # ← CHANGE THIS

if not os.path.exists(BEST_PT):
    print("hybridmain.pt not found!")
else:
    proc = subprocess.Popen(
        ['python', 'val.py', '--weights', BEST_PT, '--data', DATA_YAML, '--batch', '8'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd='/content/HybridNet/HybridNet'
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
```

---

## Cell 11 — Transfer Learning

```python
import os, sys, torch

os.chdir('/content/HybridNet/HybridNet')
sys.path.insert(0, '/content/HybridNet/HybridNet')

from model.hybridnet import build_model

WEIGHTS     = '/content/drive/MyDrive/HybridNet_runs/run_v1/hybridmain.pt'
NEW_CLASSES = 20    # ← number of classes in your new dataset
device      = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_model(num_classes=NEW_CLASSES, weights=WEIGHTS, device=device)
model.info()
print("Ready for transfer learning!")
```

---

## Important Notes

**Session flow:**
```
Day 1: Cells 1,2,3,4,5,6,7 → 10 epochs → popup → continue → 20 epochs → Resume Later
Day 2: Cells 1,2,3,4 → Cell 8 → resume from where you stopped
...repeat until 200 epochs
```

**Keep the same `RUN_NAME` every session — that's how it knows where to resume from.**

**Weights saved after every epoch:**
- `hybridmain.pt` → best overall weights (use this for inference / transfer learning)
- `hybridlast.pt` → last epoch (used internally to resume)

**Batch sizes:**
- Free Colab → `BATCH = 8`
- Colab Pro  → `BATCH = 16`

---

Questions? https://github.com/Hakai99/HybridNet/issues
