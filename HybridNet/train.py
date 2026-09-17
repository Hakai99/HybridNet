"""
HybridNet - train.py
====================
Main Training Script

Usage:
    python train.py --data data.yaml --epochs 100 --batch 16

Supports:
    - Resume from checkpoint
    - Auto-save best weights
    - Training log
    - Any YOLO-format dataset

Author: Your Name
"""

import os
import sys
import time
import argparse
import torch
import torch.optim as optim
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train HybridNet')
    parser.add_argument('--data',     type=str,   required=True,   help='path to data.yaml')
    parser.add_argument('--epochs',   type=int,   default=100,     help='number of epochs')
    parser.add_argument('--batch',    type=int,   default=16,      help='batch size')
    parser.add_argument('--imgsz',    type=int,   default=640,     help='image size')
    parser.add_argument('--lr',       type=float, default=1e-3,    help='learning rate')
    parser.add_argument('--workers',  type=int,   default=4,       help='dataloader workers')
    parser.add_argument('--weights',  type=str,   default=None,    help='resume from .pt file')
    parser.add_argument('--save-dir', type=str,   default='runs',  help='save directory')
    parser.add_argument('--name',     type=str,   default='exp',   help='run name')
    parser.add_argument('--device',   type=str,   default='',      help='cuda or cpu')
    return parser.parse_args()


def get_device(device_str):
    if device_str == '':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False):
    ckpt = {
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'loss':        loss,
    }
    torch.save(ckpt, path / 'last.pt')
    if is_best:
        torch.save(ckpt, path / 'best.pt')
        print(f"  ✅ New best saved → {path}/best.pt")


def train():
    args   = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*50}")
    print(f"  HybridNet Training")
    print(f"{'='*50}")
    print(f"  Device  : {device}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"{'='*50}\n")

    # ── Save directory ──────────────────────────────
    save_dir = Path(args.save_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataloader ──────────────────────────────────
    from utils.dataloader import build_dataloader
    train_loader, num_classes, class_names = build_dataloader(
        args.data, split='train',
        img_size=args.imgsz,
        batch_size=args.batch,
        workers=args.workers
    )

    print(f"Classes ({num_classes}): {class_names}\n")

    # ── Model ───────────────────────────────────────
    from model.hybridnet import build_model
    model = build_model(
        num_classes=num_classes,
        weights=args.weights,
        device=str(device)
    )
    model.info()

    # ── Loss ────────────────────────────────────────
    from utils.loss import HybridLoss
    criterion = HybridLoss(num_classes=num_classes).to(device)

    # ── Optimizer ───────────────────────────────────
    # Include loss uncertainty params in optimizer
    all_params = list(model.parameters()) + list(criterion.uncertainty.parameters())
    optimizer  = optim.AdamW(all_params, lr=args.lr, weight_decay=5e-4)

    # Cosine LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Resume ──────────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if args.weights and Path(args.weights).exists():
        ckpt = torch.load(args.weights, map_location=device)
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            optimizer.load_state_dict(ckpt['optim_state'])
            best_loss = ckpt.get('loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}\n")

    # ── Log file ────────────────────────────────────
    log_file = open(save_dir / 'train_log.txt', 'a')
    log_file.write(f"epoch,loss,box,cls,ctr,lr\n")

    # ── Training Loop ───────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()

        epoch_loss = 0.0
        epoch_box  = 0.0
        epoch_cls  = 0.0
        epoch_ctr  = 0.0
        t0 = time.time()

        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            imgs    = imgs.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(imgs)

            # Loss
            loss, loss_dict = criterion(predictions, targets, device)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)

            optimizer.step()

            # Accumulate
            epoch_loss += loss_dict['total']
            epoch_box  += loss_dict['box']
            epoch_cls  += loss_dict['cls']
            epoch_ctr  += loss_dict['ctr']

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                pct = 100 * (batch_idx + 1) / len(train_loader)
                print(
                    f"  Epoch [{epoch+1}/{args.epochs}] "
                    f"[{batch_idx+1}/{len(train_loader)}] {pct:.0f}% | "
                    f"loss={loss_dict['total']:.4f} "
                    f"box={loss_dict['box']:.4f} "
                    f"cls={loss_dict['cls']:.4f} "
                    f"ctr={loss_dict['ctr']:.4f} | "
                    f"w_box={loss_dict['w_box']:.2f} "
                    f"w_cls={loss_dict['w_cls']:.2f} "
                    f"w_ctr={loss_dict['w_ctr']:.2f}",
                    end='\r'
                )

        # ── End of Epoch ────────────────────────────
        scheduler.step()

        n          = len(train_loader)
        avg_loss   = epoch_loss / n
        avg_box    = epoch_box  / n
        avg_cls    = epoch_cls  / n
        avg_ctr    = epoch_ctr  / n
        cur_lr     = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"loss={avg_loss:.4f} box={avg_box:.4f} "
              f"cls={avg_cls:.4f} ctr={avg_ctr:.4f} | "
              f"lr={cur_lr:.6f} | {elapsed:.1f}s")

        # Log to file
        log_file.write(f"{epoch+1},{avg_loss:.4f},{avg_box:.4f},{avg_cls:.4f},{avg_ctr:.4f},{cur_lr:.6f}\n")
        log_file.flush()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_checkpoint(model, optimizer, epoch, avg_loss, save_dir, is_best)

    log_file.close()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Weights saved to: {save_dir}/best.pt")


if __name__ == '__main__':
    train()
