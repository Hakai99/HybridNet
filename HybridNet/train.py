import os
import sys
import argparse
import warnings
import logging
import threading

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

import torch
import torch.optim as optim
import torch.nn as nn

from utils.dataloader import build_dataloader
from utils.loss import HybridLoss
from model.hybridnet import HybridNet


GUIDE_PT = '/content/drive/MyDrive/yolov8n.pt'

_guide_model = None
_guide_lock  = threading.Lock()


def _load_guide_bg(device_str):
    global _guide_model
    try:
        if not os.path.exists(GUIDE_PT):
            return
        os.environ['YOLO_VERBOSE'] = 'False'
        logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
        from ultralytics import YOLO as _U
        _g = _U(GUIDE_PT)
        device = torch.device(device_str)
        _g.model.eval()
        _g.model.to(device)
        for p in _g.model.parameters():
            p.requires_grad_(False)
        with _guide_lock:
            _guide_model = _g.model
    except Exception:
        pass


def _get_guide():
    with _guide_lock:
        return _guide_model


def _guide_preds(guide, imgs, nc, device):
    try:
        with torch.no_grad():
            guide(imgs)
        scales = [(80, 80), (40, 40), (20, 20)]
        result = []
        for H, W in scales:
            B = imgs.shape[0]
            result.append((
                torch.zeros(B, 4,  H, W, device=device),
                torch.zeros(B, nc, H, W, device=device),
                torch.zeros(B, 1,  H, W, device=device),
            ))
        return result
    except Exception:
        return None


def train(args):
    device = torch.device(
        'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    )

    t = threading.Thread(target=_load_guide_bg, args=(str(device),), daemon=True)
    t.start()

    train_loader, nc, names = build_dataloader(
        args.data, split='train',
        img_size=args.imgsz, batch_size=args.batch
    )
    val_loader, _, _ = build_dataloader(
        args.data, split='val',
        img_size=args.imgsz, batch_size=args.batch
    )

    print("Classes (" + str(nc) + "): " + str(names))

    model = HybridNet(num_classes=nc).to(device)

    start_epoch = 0
    if args.weights and os.path.exists(args.weights):
        ckpt        = torch.load(args.weights, map_location=device)
        state       = ckpt.get('model_state', ckpt)
        model.load_state_dict(state, strict=False)
        start_epoch = ckpt.get('epoch', 0)
        print("Loaded weights from: " + args.weights)

    model.info()
    print("Resuming from epoch " + str(start_epoch))

    criterion = HybridLoss(num_classes=nc).to(device)
    params    = list(model.parameters()) + list(criterion.uncertainty.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    total_steps = (args.epochs - start_epoch) * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=0.1, div_factor=10, final_div_factor=100
    )

    save_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'train_log.txt')

    if not os.path.exists(log_file) or start_epoch == 0:
        with open(log_file, 'w') as f:
            f.write('epoch,loss,box,cls,ctr,lr\n')

    print("=" * 50)
    print("  HybridNet Training")
    print("=" * 50)
    print("  Device  : " + str(device))
    print("  Data    : " + args.data)
    print("  Epochs  : " + str(args.epochs))
    print("  Batch   : " + str(args.batch))
    print("  Img size: " + str(args.imgsz))
    print("=" * 50)

    best_loss = float('inf')
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()

        e_loss = e_box = e_cls = e_ctr = 0.0
        nb     = len(train_loader)
        guide  = _get_guide()

        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            imgs    = imgs.to(device)
            targets = targets.to(device)

            soft = _guide_preds(guide, imgs, nc, device) if guide is not None else None

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    preds      = model(imgs)
                    loss, info = criterion(preds, targets, device, soft_preds=soft)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds      = model(imgs)
                loss, info = criterion(preds, targets, device, soft_preds=soft)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            scheduler.step()

            e_loss += info['total']
            e_box  += info['box']
            e_cls  += info['cls']
            e_ctr  += info['ctr']

            print(
                '\r  Epoch [' + str(epoch+1) + '/' + str(args.epochs) + '] '
                '[' + str(batch_idx+1) + '/' + str(nb) + '] '
                + str(int((batch_idx+1)/nb*100)) + '% | '
                'loss=' + '{:.4f}'.format(info['total']) + ' '
                'box='  + '{:.4f}'.format(info['box'])   + ' '
                'cls='  + '{:.4f}'.format(info['cls'])   + ' '
                'ctr='  + '{:.4f}'.format(info['ctr']),
                end='', flush=True
            )

        print()

        avg_loss = e_loss / nb
        avg_box  = e_box  / nb
        avg_cls  = e_cls  / nb
        avg_ctr  = e_ctr  / nb
        cur_lr   = optimizer.param_groups[0]['lr']

        print(
            'Epoch ' + str(epoch+1) + '/' + str(args.epochs) +
            ' | loss=' + '{:.4f}'.format(avg_loss) +
            ' box='    + '{:.4f}'.format(avg_box)  +
            ' cls='    + '{:.4f}'.format(avg_cls)  +
            ' ctr='    + '{:.4f}'.format(avg_ctr)  +
            ' | lr='   + '{:.6f}'.format(cur_lr)
        )

        with open(log_file, 'a') as f:
            f.write(
                str(epoch+1) + ',' +
                '{:.4f}'.format(avg_loss) + ',' +
                '{:.4f}'.format(avg_box)  + ',' +
                '{:.4f}'.format(avg_cls)  + ',' +
                '{:.4f}'.format(avg_ctr)  + ',' +
                '{:.6f}'.format(cur_lr)   + '\n'
            )

        ckpt = {
            'epoch':       epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'loss':        avg_loss,
            'nc':          nc,
            'names':       names,
        }

        torch.save(ckpt, os.path.join(save_dir, 'hybridlast.pt'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, os.path.join(save_dir, 'hybridmain.pt'))
            print("  Saved -> " + os.path.join(save_dir, 'hybridmain.pt'))

    print("\nTraining complete!")
    print("Weights: " + os.path.join(save_dir, 'hybridmain.pt'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     type=str,   required=True)
    parser.add_argument('--weights',  type=str,   default='')
    parser.add_argument('--epochs',   type=int,   default=30)
    parser.add_argument('--batch',    type=int,   default=8)
    parser.add_argument('--imgsz',    type=int,   default=640)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--device',   type=str,   default='cuda')
    parser.add_argument('--save-dir', type=str,   default='runs')
    parser.add_argument('--name',     type=str,   default='exp')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
