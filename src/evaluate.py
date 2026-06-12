"""
evaluate.py
===========
Full evaluation pipeline for DarkNav.
Author: Miriam Garcia Sollo
Date:   June 2026
"""

import torch
import numpy as np
import json
import time
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import DarkNavUNet
from src.synthetic import RealCraterDataset
from src.train import compute_metrics, average_metrics


def detect_mode_from_checkpoint(ckpt_path):
    """Detect whether a checkpoint was saved with scratch or imagenet encoder
    by inspecting the state_dict keys."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = ckpt['model_state_dict']
    keys  = list(state.keys())
    # ResNet-18 encoder keys look like 'encoder.block1.0.conv1.weight'
    # LightEncoder keys look like 'encoder.block1.block.0.weight'
    if any('block.0.conv1' in k or 'downsample' in k for k in keys):
        return 'imagenet'
    return 'scratch'


def load_model(ckpt_path):
    """Load a DarkNavUNet from checkpoint, auto-detecting architecture."""
    mode  = detect_mode_from_checkpoint(ckpt_path)
    model = DarkNavUNet(mode=mode)
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded with mode='{mode}' from {Path(ckpt_path).name}")
    return model.eval(), mode


def evaluate_condition(model, test_loader, condition_name, device='cpu'):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            all_metrics.append(compute_metrics(logits, masks))
    results = average_metrics(all_metrics)
    results['condition'] = condition_name
    print(f"\n{condition_name} Test Results:")
    print(f"  IoU       : {results['iou']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1        : {results['f1']:.4f}")
    return results


def benchmark_cpu_inference(model, n_runs=50):
    model.eval()
    x = torch.zeros(1, 1, 128, 128)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms':  float(np.std(times)),
        'passes':  float(np.mean(times)) < 100.0,
    }


def export_to_onnx(model, out_path):
    import torch as _torch
    model.eval()
    dummy = _torch.zeros(1, 1, 128, 128)
    out_path = Path(out_path)
    try:
        _torch.onnx.export(
            model, dummy, str(out_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['dem_patch'],
            output_names=['crater_logits'],
            dynamic_axes={
                'dem_patch':     {0: 'batch'},
                'crater_logits': {0: 'batch'},
            },
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  ONNX exported: {out_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ONNX export skipped: {e}")
        return False


def benchmark_onnx(onnx_path, n_runs=50):
    try:
        import onnxruntime as ort
    except ImportError:
        return {}
    if not Path(onnx_path).exists():
        return {}
    sess = ort.InferenceSession(str(onnx_path))
    x = np.zeros((1, 1, 128, 128), dtype=np.float32)
    for _ in range(5):
        sess.run(None, {'dem_patch': x})
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {'dem_patch': x})
        times.append((time.perf_counter() - t0) * 1000)
    return {'mean_ms': float(np.mean(times)), 'std_ms': float(np.std(times))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_dir',   default='data/processed')
    parser.add_argument('--runs_dir',   default='runs')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    proc_dir = Path(args.proc_dir)
    runs_dir = Path(args.runs_dir)

    test_ds = RealCraterDataset(
        proc_dir / 'test_images.npy',
        proc_dir / 'test_masks.npy',
        augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} patches\n")

    checkpoints = [
        ('condition_A', runs_dir / 'condition_A' / 'condition_A_best.pth'),
        ('condition_B', runs_dir / 'condition_B' / 'condition_B_best.pth'),
        ('condition_C', runs_dir / 'condition_C' / 'condition_C_best.pth'),
    ]

    all_results = []
    for name, ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"Skipping {name}: checkpoint not found")
            continue

        print(f"Evaluating {name}...")
        model, mode = load_model(ckpt_path)

        results        = evaluate_condition(model, test_loader, name)
        results['mode'] = mode

        timing = benchmark_cpu_inference(model)
        results['cpu_ms']   = timing['mean_ms']
        results['cpu_std']  = timing['std_ms']
        results['cpu_pass'] = timing['passes']
        print(f"  CPU: {timing['mean_ms']:.1f} +/- {timing['std_ms']:.1f} ms "
              f"({'PASS' if timing['passes'] else 'FAIL'})")

        onnx_path = runs_dir / name / f'{name}.onnx'
        ok = export_to_onnx(model, onnx_path)
        if ok:
            onnx_t = benchmark_onnx(onnx_path)
            if onnx_t:
                results['onnx_ms'] = onnx_t['mean_ms']
                print(f"  ONNX: {onnx_t['mean_ms']:.1f} ms")

        all_results.append(results)

    # Save always, even if partial
    out_path = runs_dir / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n=== FINAL COMPARISON TABLE ===")
    print(f"{'Condition':<15} {'Mode':<10} {'IoU':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6} {'CPU ms':>8}")
    print("-" * 62)
    for r in all_results:
        print(f"{r['condition']:<15} {r.get('mode','?'):<10} "
              f"{r['iou']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} "
              f"{r.get('cpu_ms', 0):>8.1f}")


if __name__ == '__main__':
    main()
