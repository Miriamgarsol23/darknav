"""
train.py
========
Training loop for DarkNav: three experimental conditions.

Author: Miriam Garcia Sollo
Date:   June 2026
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DarkNavUNet
from src.synthetic import SyntheticCraterDataset, RealCraterDataset


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    TP = (preds * targets).sum().item()
    FP = (preds * (1 - targets)).sum().item()
    FN = ((1 - preds) * targets).sum().item()

    iou       = TP / (TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}


def average_metrics(metrics_list: list) -> dict:
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ─────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, device='cpu', phase='train'):
    is_train = (phase == 'train')
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_metrics = []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_metrics.append(compute_metrics(logits.detach(), masks.detach()))

    avg = average_metrics(all_metrics)
    avg['loss'] = total_loss / len(loader)
    return avg


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(model, train_loader, val_loader, n_epochs, lr, pos_weight,
          out_dir, phase_name, device='cpu', patience=10):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clamp pos_weight: a value above ~50 makes training unstable.
    # If the real dataset has very sparse masks, cap at 20 and warn.
    if pos_weight > 50:
        print(f"WARNING: pos_weight={pos_weight:.1f} is very high (sparse masks).")
        print("         Clamping to 20.0 for training stability.")
        print("         Consider revisiting the coverage filter in preprocessing.py.")
        pos_weight = 20.0

    pw        = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # verbose removed in PyTorch 2.x
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    history = {
        'train_loss': [], 'train_iou': [], 'train_f1': [],
        'val_loss':   [], 'val_iou':   [], 'val_f1':   [],
        'val_precision': [], 'val_recall': [],
    }

    best_val_iou      = 0.0
    epochs_no_improve = 0
    best_ckpt_path    = out_dir / f'{phase_name}_best.pth'

    print(f"\nPhase: {phase_name}")
    print(f"  Epochs     : {n_epochs}")
    print(f"  LR         : {lr}")
    print(f"  pos_weight : {pos_weight:.2f}")
    print(f"  Device     : {device}")
    print(f"  Output     : {out_dir}/\n")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, 'train')
        val_stats   = run_epoch(model, val_loader,   criterion, None,      device, 'val')

        scheduler.step(val_stats['iou'])

        history['train_loss'].append(train_stats['loss'])
        history['train_iou'].append(train_stats['iou'])
        history['train_f1'].append(train_stats['f1'])
        history['val_loss'].append(val_stats['loss'])
        history['val_iou'].append(val_stats['iou'])
        history['val_f1'].append(val_stats['f1'])
        history['val_precision'].append(val_stats['precision'])
        history['val_recall'].append(val_stats['recall'])

        current_lr = optimizer.param_groups[0]['lr']
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train loss={train_stats['loss']:.4f} iou={train_stats['iou']:.4f} | "
            f"val loss={val_stats['loss']:.4f} iou={val_stats['iou']:.4f} "
            f"prec={val_stats['precision']:.3f} rec={val_stats['recall']:.3f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_stats['iou'] > best_val_iou:
            best_val_iou = val_stats['iou']
            epochs_no_improve = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_iou':          best_val_iou,
                'val_precision':    val_stats['precision'],
                'val_recall':       val_stats['recall'],
                'val_f1':           val_stats['f1'],
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    with open(out_dir / f'{phase_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val IoU: {best_val_iou:.4f}")
    print(f"Checkpoint  : {best_ckpt_path}")
    return history


# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────

def build_dataloaders(proc_dir, synth_n=1500, batch_size=8,
                      use_synthetic_train=False):
    proc_dir = Path(proc_dir)
    pos_weight_path = proc_dir.parent / 'pos_weight.npy'

    # Recompute pos_weight from actual processed masks rather than
    # trusting the saved value (which may have been computed on synthetic).
    train_mask_path = proc_dir / 'train_masks.npy'
    if train_mask_path.exists():
        masks = np.load(train_mask_path, mmap_mode='r')
        mean_cov   = float(masks.mean())
        pos_weight = (1.0 - mean_cov) / (mean_cov + 1e-6)
        print(f"  Recomputed pos_weight from real masks: "
              f"mean_coverage={mean_cov:.5f}, pos_weight={pos_weight:.2f}")
    elif pos_weight_path.exists():
        pos_weight = float(np.load(pos_weight_path)[0])
    else:
        pos_weight = 9.0

    if use_synthetic_train:
        train_ds = SyntheticCraterDataset(n_samples=synth_n, seed=42, augment=True)
    else:
        train_ds = RealCraterDataset(
            images_path=proc_dir / 'train_images.npy',
            masks_path =proc_dir / 'train_masks.npy',
            augment=True,
        )

    val_ds = RealCraterDataset(
        images_path=proc_dir / 'dev_images.npy',
        masks_path =proc_dir / 'dev_masks.npy',
        augment=False,
    )
    test_ds = RealCraterDataset(
        images_path=proc_dir / 'test_images.npy',
        masks_path =proc_dir / 'test_masks.npy',
        augment=False,
    )

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"DataLoaders built:")
    print(f"  train : {len(train_ds)} samples ({type(train_ds).__name__})")
    print(f"  val   : {len(val_ds)} samples")
    print(f"  test  : {len(test_ds)} samples")
    print(f"  pos_weight: {pos_weight:.2f}")

    return train_loader, val_loader, test_loader, pos_weight


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DarkNav training')
    parser.add_argument('--condition', required=True,
                        choices=['A', 'B', 'C_pretrain', 'C_finetune'])
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--proc_dir',   type=str,   default='data/processed')
    parser.add_argument('--out_dir',    type=str,   default='runs')
    parser.add_argument('--nfw_checkpoint', type=str, default=None)
    args = parser.parse_args()

    proc_dir = Path(args.proc_dir)
    out_dir  = Path(args.out_dir)
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.condition == 'A':
        model = DarkNavUNet(mode='scratch')
        train_loader, val_loader, _, pos_weight = build_dataloaders(
            proc_dir, batch_size=args.batch_size, use_synthetic_train=False
        )
        train(model, train_loader, val_loader,
              n_epochs=args.epochs, lr=args.lr, pos_weight=pos_weight,
              out_dir=out_dir / 'condition_A', phase_name='condition_A',
              device=device)

    elif args.condition == 'B':
        model = DarkNavUNet(mode='imagenet')
        train_loader, val_loader, _, pos_weight = build_dataloaders(
            proc_dir, batch_size=args.batch_size, use_synthetic_train=False
        )
        train(model, train_loader, val_loader,
              n_epochs=args.epochs, lr=args.lr * 0.1, pos_weight=pos_weight,
              out_dir=out_dir / 'condition_B', phase_name='condition_B',
              device=device)

    elif args.condition == 'C_pretrain':
        model = DarkNavUNet(mode='scratch')
        train_loader, val_loader, _, pos_weight = build_dataloaders(
            proc_dir, batch_size=args.batch_size, use_synthetic_train=True
        )
        train(model, train_loader, val_loader,
              n_epochs=args.epochs, lr=args.lr, pos_weight=pos_weight,
              out_dir=out_dir / 'condition_C', phase_name='nfw_pretrain',
              device=device, patience=15)

    elif args.condition == 'C_finetune':
        if args.nfw_checkpoint is None:
            raise ValueError("--nfw_checkpoint required for C_finetune")
        model = DarkNavUNet(mode='nfw', nfw_checkpoint=args.nfw_checkpoint)
        train_loader, val_loader, _, pos_weight = build_dataloaders(
            proc_dir, batch_size=args.batch_size, use_synthetic_train=False
        )
        train(model, train_loader, val_loader,
              n_epochs=args.epochs, lr=args.lr * 0.1, pos_weight=pos_weight,
              out_dir=out_dir / 'condition_C', phase_name='condition_C',
              device=device)


if __name__ == '__main__':
    main()
