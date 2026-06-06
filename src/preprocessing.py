"""
preprocessing.py  (corrected for real DeepMoon HDF5 structure)
==============================================================
Author: Miriam Garcia Sollo
Date:   June 2026

Real DeepMoon HDF5 structure (verified from file):
    f['input_images']   -> Group with keys '0', '1', ..., 'N-1'
                           each value is a 2D array (H, W) float
    f['target_masks']   -> Group with keys '0', '1', ..., 'N-1'
                           each value is a 2D binary array (H, W)
    f['cll_xy']         -> crater positions (not used, we use target_masks)
    f['longlat_bounds'] -> geographic bounds per patch
    f['pix_bounds']     -> pixel bounds
"""

import numpy as np
import h5py
import argparse
from pathlib import Path
from scipy.ndimage import zoom
from typing import Tuple, Optional


PATCH_SIZE   = 128
MIN_COVERAGE = 0.005
MAX_COVERAGE = 0.60


# ─────────────────────────────────────────────
# HDF5 loading  (CORRECTED)
# ─────────────────────────────────────────────

def load_hdf5_split(
    path: Path,
    target_size: int = PATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DEM patches and masks from a DeepMoon HDF5 file.

    The DeepMoon HDF5 format stores images and masks as Groups
    with integer-string subkeys ('0', '1', ..., 'N-1').

    Parameters
    ----------
    path : Path to .hdf5 file
    target_size : output patch size in pixels (downsampled from original)

    Returns
    -------
    images : np.ndarray, shape (N, target_size, target_size), float32
             Normalised to [-1, 1] per patch.
    masks  : np.ndarray, shape (N, target_size, target_size), uint8
             Binary: 1 = crater pixel, 0 = background.
    """
    with h5py.File(path, 'r') as f:

        # Discover structure: subkeys are '0', '1', ... as strings
        img_group  = f['input_images']
        mask_group = f['target_masks']

        # Sort keys numerically
        keys = sorted(img_group.keys(), key=lambda k: int(k))
        n = len(keys)
        print(f'  Found {n} patches in {path.name}')

        # Load first patch to get original size
        first_img = img_group[keys[0]][:]
        orig_size = first_img.shape[-1]  # works for (H,W) or (1,H,W) or (H,W,1)

        print(f'  Original patch shape: {img_group[keys[0]].shape}')
        factor = target_size / orig_size if orig_size != target_size else 1.0

        images = np.zeros((n, target_size, target_size), dtype=np.float32)
        masks  = np.zeros((n, target_size, target_size), dtype=np.uint8)

        for i, k in enumerate(keys):
            if i % 1000 == 0:
                print(f'  Loading {i}/{n}...', end='\r')

            # Load image
            img = img_group[k][:]
            img = np.squeeze(img)            # remove any singleton dims -> (H, W)
            if img.ndim != 2:
                raise ValueError(f'Unexpected image shape after squeeze: {img.shape}')

            # Load mask
            msk = mask_group[k][:]
            msk = np.squeeze(msk)

            # Downsample if needed
            if factor != 1.0:
                img = zoom(img, factor, order=1)
                msk = zoom(msk, factor, order=0)  # nearest for binary mask

            images[i] = img.astype(np.float32)
            masks[i]  = (msk > 0).astype(np.uint8)

        print(f'  Loaded {n}/{n} patches')

    # Normalise images to [-1, 1] per patch
    pmin = images.min(axis=(1, 2), keepdims=True)
    pmax = images.max(axis=(1, 2), keepdims=True)
    denom = np.where(pmax - pmin > 0, pmax - pmin, 1.0)
    images = (2.0 * (images - pmin) / denom - 1.0).astype(np.float32)

    return images, masks


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
          f'{len(images)} -> {len(keep)} patches kept')
    return images[keep], masks[keep], keep


# ─────────────────────────────────────────────
# Sobel channel (optional)
# ─────────────────────────────────────────────

def apply_sobel_channel(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude normalised to [0, 1]."""
    from scipy.ndimage import sobel
    sx = sobel(image, axis=1)
    sy = sobel(image, axis=0)
    mag = np.sqrt(sx**2 + sy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag.astype(np.float32)


# ─────────────────────────────────────────────
# Full pipeline for one split  (CORRECTED)
# ─────────────────────────────────────────────

def process_split(
    split: str,
    data_dir: Path,
    out_dir: Path,
    add_sobel: bool = False,
) -> dict:
    """Run the full preprocessing pipeline for one data split.

    Steps:
    1. Load HDF5: images from 'input_images', masks from 'target_masks'
    2. Downsample to 128x128
    3. Normalise images to [-1, 1]
    4. Filter by mask coverage
    5. Save to out_dir as .npy files

    Parameters
    ----------
    split    : 'train', 'dev', or 'test'
    data_dir : directory with raw Zenodo HDF5 files
    out_dir  : directory for processed .npy files
    add_sobel: if True, also save Sobel gradient channel
    """
    print(f'\nProcessing split: {split}')
    print('-' * 40)

    img_path = data_dir / f'{split}_images.hdf5'
    if not img_path.exists():
        raise FileNotFoundError(
            f'Missing: {img_path}\nDownload with: zenodo_get 1133969'
        )

    # Step 1-3: load, downsample, normalise
    images, masks = load_hdf5_split(img_path, target_size=PATCH_SIZE)

    # Step 4: filter
    images, masks, kept_idx = filter_by_coverage(images, masks)

    # Step 5: optional Sobel
    if add_sobel:
        print('  Computing Sobel channel...', end=' ')
        sobel_ch = np.stack([apply_sobel_channel(images[i]) for i in range(len(images))])
        np.save(out_dir / f'{split}_sobel.npy', sobel_ch)
        print(f'done ({sobel_ch.nbytes/1e6:.1f} MB)')

    # Step 6: save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{split}_images.npy',  images)
    np.save(out_dir / f'{split}_masks.npy',   masks)
    np.save(out_dir / f'{split}_indices.npy', kept_idx)

    stats = {
        'split':   split,
        'n_kept':  len(images),
        'mean_cov': float(masks.mean()),
        'mb':       images.nbytes / 1e6,
    }

    print(f'  Saved {len(images)} patches -> {out_dir}/')
    print(f'  {split}_images.npy : {images.shape}, {stats["mb"]:.1f} MB')
    print(f'  Mean mask coverage : {stats["mean_cov"]:.4f}')
    return stats


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir',  type=str, default='data/processed')
    parser.add_argument('--sobel', action='store_true')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    for split in args.splits:
        process_split(split, data_dir, out_dir, add_sobel=args.sobel)


if __name__ == '__main__':
    main()
