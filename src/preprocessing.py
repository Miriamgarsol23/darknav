"""
preprocessing.py
================
Pipeline for loading, filtering, and saving real LRO DEM patches
from the DeepMoon dataset (Silburt et al. 2019, Zenodo 1133969).

Author: Miriam Garcia Sollo
Date:   June 2026

Real HDF5 structure (verified):
    f['input_images']  -> Dataset (30000, 256, 256) uint8
    f['target_masks']  -> Dataset (30000, 256, 256) float32
    f['cll_xy']        -> Group with subkeys img_00000, img_00001, ...
    f['longlat_bounds'], f['pix_bounds'], f['pix_distortion_coefficient'] -> Groups

Usage:
    python src/preprocessing.py --data_dir data --out_dir data/processed
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from scipy.ndimage import zoom
from typing import Tuple


PATCH_SIZE   = 128
MIN_COVERAGE = 0.005
MAX_COVERAGE = 0.60


# ─────────────────────────────────────────────
# HDF5 loading
# ─────────────────────────────────────────────

def load_hdf5_split(
    path: Path,
    target_size: int = PATCH_SIZE,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DEM patches and masks from a DeepMoon HDF5 file.

    input_images and target_masks are both plain 3D Datasets:
        input_images : (N, 256, 256) uint8
        target_masks : (N, 256, 256) float32, values in [0, 1]

    Parameters
    ----------
    path        : Path to .hdf5 file
    target_size : output patch size (downsampled from 256)
    max_samples : if set, only load the first max_samples patches (for testing)

    Returns
    -------
    images : (N, target_size, target_size) float32, normalised to [-1, 1]
    masks  : (N, target_size, target_size) uint8, binary 0/1
    """
    with h5py.File(path, 'r') as f:
        raw_images = f['input_images']  # Dataset (N, 256, 256) uint8
        raw_masks  = f['target_masks']  # Dataset (N, 256, 256) float32

        n_total = raw_images.shape[0]
        n = n_total if max_samples is None else min(max_samples, n_total)

        orig_size = raw_images.shape[1]
        print(f'  {path.name}: {n_total} patches at {orig_size}x{orig_size}')
        print(f'  Loading {n} patches...', end=' ')

        imgs  = raw_images[:n].astype(np.float32)   # load into RAM
        masks = raw_masks[:n]                         # float32 in [0,1]

    print('done')

    # Downsample to target_size
    if orig_size != target_size:
        factor = target_size / orig_size
        print(f'  Downsampling {orig_size} -> {target_size} (factor={factor:.3f})...', end=' ')
        imgs  = np.stack([zoom(imgs[i],  factor, order=1) for i in range(n)])
        masks = np.stack([zoom(masks[i], factor, order=0) for i in range(n)])
        print('done')

    # Normalise images to [-1, 1] per patch
    pmin  = imgs.min(axis=(1, 2), keepdims=True)
    pmax  = imgs.max(axis=(1, 2), keepdims=True)
    denom = np.where(pmax - pmin > 0, pmax - pmin, 1.0)
    imgs  = (2.0 * (imgs - pmin) / denom - 1.0).astype(np.float32)

    # Binarise masks (threshold at 0.5)
    masks = (masks > 0.5).astype(np.uint8)

    return imgs, masks


# ─────────────────────────────────────────────
# Quality filtering
# ─────────────────────────────────────────────

def filter_by_coverage(
    images: np.ndarray,
    masks: np.ndarray,
    min_cov: float = MIN_COVERAGE,
    max_cov: float = MAX_COVERAGE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove patches with crater coverage outside [min_cov, max_cov]."""
    coverages = masks.mean(axis=(1, 2))
    keep = np.where((coverages >= min_cov) & (coverages <= max_cov))[0]
    print(f'  Coverage filter [{min_cov}, {max_cov}]: '
          f'{len(images)} -> {len(keep)} patches kept '
          f'(removed {len(images) - len(keep)})')
    return images[keep], masks[keep], keep


# ─────────────────────────────────────────────
# Full pipeline for one split
# ─────────────────────────────────────────────

def process_split(
    split: str,
    data_dir: Path,
    out_dir: Path,
    max_samples: int = None,
) -> dict:
    """Run the full preprocessing pipeline for one data split.

    Steps:
    1. Load HDF5 (input_images + target_masks)
    2. Downsample to 128x128
    3. Normalise images to [-1, 1] per patch
    4. Binarise masks at threshold 0.5
    5. Filter by mask coverage
    6. Save to out_dir as .npy files

    Parameters
    ----------
    split       : 'train', 'dev', or 'test'
    data_dir    : directory with raw Zenodo HDF5 files
    out_dir     : directory for processed .npy files
    max_samples : limit patches loaded (useful for quick testing)
    """
    print(f'\nProcessing split: {split}')
    print('-' * 40)

    img_path = data_dir / f'{split}_images.hdf5'
    if not img_path.exists():
        raise FileNotFoundError(
            f'Missing: {img_path}\n'
            f'Download with: cd data && zenodo_get 1133969'
        )

    images, masks = load_hdf5_split(img_path, target_size=PATCH_SIZE,
                                    max_samples=max_samples)
    images, masks, kept_idx = filter_by_coverage(images, masks)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{split}_images.npy',  images)
    np.save(out_dir / f'{split}_masks.npy',   masks)
    np.save(out_dir / f'{split}_indices.npy', kept_idx)

    stats = {
        'split':    split,
        'n_kept':   len(images),
        'mean_cov': float(masks.mean()),
        'mb':       images.nbytes / 1e6,
    }

    print(f'  Saved {len(images)} patches to {out_dir}/')
    print(f'  {split}_images.npy : {images.shape}, {stats["mb"]:.1f} MB')
    print(f'  Mean mask coverage : {stats["mean_cov"]:.4f}')
    return stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DarkNav preprocessing: HDF5 -> filtered .npy arrays'
    )
    parser.add_argument('--data_dir',    type=str, default='data')
    parser.add_argument('--out_dir',     type=str, default='data/processed')
    parser.add_argument('--splits',      nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--max_samples', type=int,  default=None,
                        help='Limit patches per split (for quick testing, e.g. 500)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    all_stats = []
    for split in args.splits:
        try:
            stats = process_split(split, data_dir, out_dir,
                                  max_samples=args.max_samples)
            all_stats.append(stats)
        except FileNotFoundError as e:
            print(f'  Skipping {split}: {e}')

    print('\n=== Preprocessing summary ===')
    for s in all_stats:
        print(f"  {s['split']:5s}: {s['n_kept']:5d} patches  "
              f"{s['mb']:.0f} MB  coverage={s['mean_cov']:.4f}")


if __name__ == '__main__':
    main()
