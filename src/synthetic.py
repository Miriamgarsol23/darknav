"""
synthetic.py
============
NFW-based synthetic crater DEM generator for DarkNav.

Author: Miriam Garcia Sollo
Date:   June 2026

This module implements the physics-informed synthetic data generator described in
the DarkNav design document. It produces 128x128 float32 DEM patches and paired
binary segmentation masks using the analytical projected NFW density profile
(Bartelmann 1996; Wright & Brainerd 2000) combined with a power-law ejecta model
(Melosh 1989).

The morphological analogy:
    NFW projected surface density Sigma(R) -> inverted -> crater bowl
    Power-law ejecta beyond r_rim          -> crater rim elevation
    Gaussian noise                         -> sensor noise

Usage:
    from src.synthetic import SyntheticCraterDataset, generate_synthetic_dataset
    dataset = SyntheticCraterDataset(n_samples=1500, seed=42)
    image, mask = dataset[0]

    # Or generate and save to disk:
    generate_synthetic_dataset(n_samples=1500, out_dir='data/synthetic', seed=42)
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Core NFW profile functions
# ─────────────────────────────────────────────

def nfw_projected_F(x: np.ndarray) -> np.ndarray:
    """Dimensionless profile factor F(x) for the projected NFW surface density.

    Reference: Wright and Brainerd (2000), ApJ 534, 34-40.
    Derived from Bartelmann (1996), A&A 313, 697-702.

    The projected surface density is Sigma(R) = 2 * rho_s * r_s * F(R/r_s).

    Parameters
    ----------
    x : array-like
        Dimensionless projected radius x = R / r_s. Must be positive.

    Returns
    -------
    F : np.ndarray
        Profile factor, always positive and monotonically decreasing in x.

    Notes
    -----
    The x > 1 branch uses arctan(sqrt(x^2-1)) rather than arccos(1/x) for
    numerical stability near x = 1. The two forms are mathematically equivalent
    for x > 1 but arctan avoids precision loss in the transition region.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    F = np.zeros_like(x)

    mask_inner = x < 1.0
    xm = x[mask_inner]
    F[mask_inner] = (1.0 / (xm**2 - 1.0)) * (
        1.0 - np.arccosh(1.0 / xm) / np.sqrt(1.0 - xm**2)
    )

    mask_unity = np.abs(x - 1.0) < 1e-6
    F[mask_unity] = 1.0 / 3.0

    mask_outer = x > 1.0
    xm = x[mask_outer]
    F[mask_outer] = (1.0 / (xm**2 - 1.0)) * (
        1.0 - np.arctan(np.sqrt(xm**2 - 1.0)) / np.sqrt(xm**2 - 1.0)
    )

    return F


def nfw_projected_sigma(
    R: np.ndarray,
    rho_s: float = 1.0,
    r_s: float = 15.0,
) -> np.ndarray:
    """Projected NFW surface mass density Sigma(R).

    Parameters
    ----------
    R     : projected radius in pixels (2D array from ogrid)
    rho_s : characteristic density (amplitude parameter)
    r_s   : scale radius in pixels

    Returns
    -------
    Sigma : np.ndarray, same shape as R
    """
    return 2.0 * rho_s * r_s * nfw_projected_F(R / r_s)


# ─────────────────────────────────────────────
# Synthetic crater generator
# ─────────────────────────────────────────────

def make_single_crater(
    size: int = 128,
    r_s: float = 20.0,
    rho_s: float = 1.0,
    r_rim_factor: float = 1.5,
    t0_ejecta: float = 0.25,
    alpha_ejecta: float = 3.2,
    noise_std: float = 0.03,
    center_offset: Tuple[int, int] = (0, 0),
    ellipticity: float = 0.0,
    ellipticity_angle: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single synthetic crater DEM patch and binary mask.

    The DEM is constructed as:
        bowl  = -Sigma(R) / max(Sigma)       [inverted NFW projection]
        rim   = t0 * (R / r_rim)^(-alpha)    [power-law ejecta, r > r_rim]
        dem   = (bowl + rim) / max(|bowl+rim|) + Gaussian_noise

    The binary mask labels pixels where dem < MASK_THRESHOLD as crater interior.

    Parameters
    ----------
    size : int
        Output patch size in pixels (square).
    r_s : float
        NFW scale radius in pixels. Controls crater width.
        Calibrated: r_s = 1 pixel ~ 0.24 km at LOLA 118 m/px downsampled to 128px.
    rho_s : float
        NFW amplitude. Controls bowl depth relative to rim.
    r_rim_factor : float
        Rim radius as multiple of r_s. Default 1.5 based on calibration in Day 2.
    t0_ejecta : float
        Ejecta thickness amplitude at the rim. Controls rim height.
    alpha_ejecta : float
        Power-law decay exponent (empirically 2.8-3.5, Melosh 1989).
    noise_std : float
        Gaussian noise standard deviation. Simulates sensor noise.
    center_offset : (dy, dx) tuple
        Integer pixel offset of crater centre from patch centre.
        Used to place craters off-centre for realistic training diversity.
    ellipticity : float in [0, 0.5]
        Elliptical distortion to simulate oblique impacts.
        0 = circular. 0.3 = moderately elliptical.
    ellipticity_angle : float
        Rotation angle of ellipse major axis in radians.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    dem  : np.ndarray, shape (size, size), float32, range approximately [-1, 1]
    mask : np.ndarray, shape (size, size), uint8, values 0 or 1
    """
    MASK_THRESHOLD = -0.20  # calibrated in Day 2 notebook

    if rng is None:
        rng = np.random.default_rng()

    cx = size // 2 + center_offset[1]
    cy = size // 2 + center_offset[0]

    yg, xg = np.ogrid[:size, :size]
    dx = xg - cx
    dy = yg - cy

    # Apply elliptical distortion if requested
    if ellipticity > 0:
        cos_a = np.cos(ellipticity_angle)
        sin_a = np.sin(ellipticity_angle)
        dx_rot =  cos_a * dx + sin_a * dy
        dy_rot = -sin_a * dx + cos_a * dy
        a = 1.0
        b = 1.0 - ellipticity
        R = np.sqrt((dx_rot / a)**2 + (dy_rot / b)**2) + 1e-3
    else:
        R = np.sqrt(dx**2 + dy**2) + 1e-3

    # Bowl: inverted projected NFW
    sigma = nfw_projected_sigma(R, rho_s=rho_s, r_s=r_s)
    bowl = -sigma / sigma.max()

    # Rim: power-law ejecta beyond r_rim
    r_rim = r_rim_factor * r_s
    rim = np.zeros_like(bowl)
    outside = R > r_rim
    rim[outside] = t0_ejecta * (R[outside] / r_rim) ** (-alpha_ejecta)

    # Combine and normalise
    dem = bowl + rim
    dem = dem / np.abs(dem).max()

    # Add Gaussian noise
    dem = dem + rng.normal(0.0, noise_std, dem.shape)

    dem = dem.astype(np.float32)
    mask = (dem < MASK_THRESHOLD).astype(np.uint8)

    return dem, mask


def make_multi_crater_patch(
    size: int = 128,
    n_craters: int = 1,
    r_s_range: Tuple[float, float] = (8.0, 30.0),
    noise_std: float = 0.04,
    background_roughness: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a DEM patch with multiple NFW craters and a rough background.

    Multiple craters are placed at random positions and sizes within the patch.
    Their contributions to the DEM are summed, which approximates the appearance
    of overlapping craters of different ages in real lunar terrain.

    Parameters
    ----------
    size : int
    n_craters : int
        Number of craters to place in the patch.
    r_s_range : (min_r_s, max_r_s)
        Uniform sampling range for r_s in pixels.
    noise_std : float
        Gaussian noise on the final DEM.
    background_roughness : float
        Amplitude of low-frequency background terrain variation.
    rng : np.random.Generator or None

    Returns
    -------
    dem  : np.ndarray, shape (size, size), float32
    mask : np.ndarray, shape (size, size), uint8
    """
    MASK_THRESHOLD = -0.20

    if rng is None:
        rng = np.random.default_rng()

    # Low-frequency background roughness (tilted planes, gentle slopes)
    yg, xg = np.mgrid[:size, :size]
    slope_x = rng.uniform(-background_roughness, background_roughness)
    slope_y = rng.uniform(-background_roughness, background_roughness)
    dem = (slope_x * (xg / size) + slope_y * (yg / size)).astype(np.float32)
    combined_mask = np.zeros((size, size), dtype=np.uint8)

    for _ in range(n_craters):
        r_s = rng.uniform(*r_s_range)
        # Place centre randomly but keep the crater mostly inside the patch
        margin = int(r_s * 1.5)
        margin = min(margin, size // 4)
        cx_off = int(rng.integers(-margin, margin + 1))
        cy_off = int(rng.integers(-margin, margin + 1))
        ellip = rng.uniform(0.0, 0.25)
        angle = rng.uniform(0, np.pi)
        noise_i = rng.uniform(0.01, 0.06)
        alpha_i = rng.uniform(2.8, 3.5)

        dem_i, mask_i = make_single_crater(
            size=size,
            r_s=r_s,
            noise_std=noise_i,
            center_offset=(cy_off, cx_off),
            ellipticity=ellip,
            ellipticity_angle=angle,
            alpha_ejecta=alpha_i,
            rng=rng,
        )
        # Accumulate: craters add to the DEM
        dem = dem + 0.8 * dem_i
        combined_mask = np.maximum(combined_mask, mask_i)

    # Final normalisation and noise
    dem = dem / (np.abs(dem).max() + 1e-6)
    dem = dem + rng.normal(0, noise_std, dem.shape).astype(np.float32)
    dem = dem.astype(np.float32)

    # Recompute mask on the final combined DEM
    combined_mask = (dem < MASK_THRESHOLD).astype(np.uint8)

    return dem, combined_mask


# ─────────────────────────────────────────────
# Dataset generation: save to disk
# ─────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 1500,
    out_dir: str = 'data/synthetic',
    size: int = 128,
    seed: int = 42,
    multi_crater_fraction: float = 0.35,
) -> None:
    """Generate and save the full synthetic dataset to disk.

    Saves images and masks as compressed .npz files for fast loading.
    Also saves a CSV log with per-sample generation parameters.

    Parameters
    ----------
    n_samples : int
        Total number of synthetic patches to generate.
    out_dir : str
        Output directory. Created if it does not exist.
    size : int
        Patch size in pixels.
    seed : int
        Master random seed for full reproducibility.
    multi_crater_fraction : float
        Fraction of samples that contain multiple craters (1-3).
        The rest are single-crater patches.

    Output files
    ------------
    out_dir/synthetic_images.npy  : float32 array (N, H, W)
    out_dir/synthetic_masks.npy   : uint8   array (N, H, W)
    out_dir/generation_log.csv    : parameters for each sample
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=seed)
    images = np.zeros((n_samples, size, size), dtype=np.float32)
    masks  = np.zeros((n_samples, size, size), dtype=np.uint8)

    log_rows = []
    n_multi = int(n_samples * multi_crater_fraction)

    print(f'Generating {n_samples} synthetic patches ({size}x{size})')
    print(f'  Single-crater: {n_samples - n_multi}')
    print(f'  Multi-crater:  {n_multi}')

    for i in range(n_samples):
        if i % 100 == 0:
            print(f'  {i}/{n_samples}', end='\r')

        if i < n_samples - n_multi:
            # Single crater patch
            r_s    = rng.uniform(8.0, 30.0)
            noise  = rng.uniform(0.01, 0.07)
            ellip  = rng.uniform(0.0, 0.30)
            angle  = rng.uniform(0.0, np.pi)
            alpha  = rng.uniform(2.8, 3.5)
            cx_off = int(rng.integers(-12, 13))
            cy_off = int(rng.integers(-12, 13))

            dem, mask = make_single_crater(
                size=size, r_s=r_s, noise_std=noise,
                center_offset=(cy_off, cx_off),
                ellipticity=ellip, ellipticity_angle=angle,
                alpha_ejecta=alpha, rng=rng,
            )
            log_rows.append({
                'idx': i, 'type': 'single', 'n_craters': 1,
                'r_s': round(r_s, 2), 'noise': round(noise, 3),
                'ellipticity': round(ellip, 3), 'alpha': round(alpha, 2),
            })
        else:
            # Multi-crater patch
            n_c = int(rng.integers(2, 4))
            noise = rng.uniform(0.02, 0.06)
            dem, mask = make_multi_crater_patch(
                size=size, n_craters=n_c,
                noise_std=noise, rng=rng,
            )
            log_rows.append({
                'idx': i, 'type': 'multi', 'n_craters': n_c,
                'r_s': -1, 'noise': round(noise, 3),
                'ellipticity': -1, 'alpha': -1,
            })

        images[i] = dem
        masks[i]  = mask

    print(f'  {n_samples}/{n_samples} - done')

    # Save arrays
    np.save(out_path / 'synthetic_images.npy', images)
    np.save(out_path / 'synthetic_masks.npy',  masks)

    # Save generation log
    import csv
    with open(out_path / 'generation_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f'Saved to {out_path}/')
    print(f'  synthetic_images.npy : {images.shape}, {images.nbytes / 1e6:.1f} MB')
    print(f'  synthetic_masks.npy  : {masks.shape}')
    print(f'  generation_log.csv   : {len(log_rows)} rows')
    print(f'  Mask coverage stats  : mean={masks.mean():.3f}, '
          f'min={masks.min()}, max={masks.max()}')


# ─────────────────────────────────────────────
# PyTorch Dataset classes
# ─────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SyntheticCraterDataset(Dataset):
        """PyTorch Dataset for synthetic NFW crater DEM patches.

        Generates patches on-the-fly (no disk I/O during training).
        Each call to __getitem__ produces a fresh random patch.

        Parameters
        ----------
        n_samples : int
            Length of the dataset (number of unique patches).
        size : int
            Patch size in pixels.
        seed : int
            Master seed. Each sample uses seed + idx for reproducibility.
        augment : bool
            If True, apply random flips and rotations.
        multi_crater_fraction : float
            Fraction of samples with 2-3 craters.

        Returns
        -------
        image : torch.Tensor, shape (1, H, W), float32
        mask  : torch.Tensor, shape (1, H, W), float32 (values 0.0 or 1.0)
        """

        def __init__(
            self,
            n_samples: int = 1500,
            size: int = 128,
            seed: int = 42,
            augment: bool = True,
            multi_crater_fraction: float = 0.35,
        ):
            self.n_samples = n_samples
            self.size = size
            self.seed = seed
            self.augment = augment
            self.multi_crater_fraction = multi_crater_fraction
            self.n_multi = int(n_samples * multi_crater_fraction)

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, idx: int):
            rng = np.random.default_rng(seed=self.seed + idx)

            if idx < self.n_samples - self.n_multi:
                r_s   = rng.uniform(8.0, 30.0)
                noise = rng.uniform(0.01, 0.07)
                ellip = rng.uniform(0.0, 0.30)
                angle = rng.uniform(0.0, np.pi)
                alpha = rng.uniform(2.8, 3.5)
                cx_off = int(rng.integers(-12, 13))
                cy_off = int(rng.integers(-12, 13))
                dem, mask = make_single_crater(
                    size=self.size, r_s=r_s, noise_std=noise,
                    center_offset=(cy_off, cx_off),
                    ellipticity=ellip, ellipticity_angle=angle,
                    alpha_ejecta=alpha, rng=rng,
                )
            else:
                n_c   = int(rng.integers(2, 4))
                noise = rng.uniform(0.02, 0.06)
                dem, mask = make_multi_crater_patch(
                    size=self.size, n_craters=n_c,
                    noise_std=noise, rng=rng,
                )

            dem  = dem.astype(np.float32)
            mask = mask.astype(np.float32)

            if self.augment:
                dem, mask = self._augment(dem, mask, rng)

            image_t = torch.from_numpy(dem[np.newaxis])   # (1, H, W)
            mask_t  = torch.from_numpy(mask[np.newaxis])  # (1, H, W)

            return image_t, mask_t

        @staticmethod
        def _augment(
            dem: np.ndarray,
            mask: np.ndarray,
            rng: np.random.Generator,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Apply random geometric augmentations consistently to DEM and mask."""
            # Random horizontal flip
            if rng.random() > 0.5:
                dem  = np.fliplr(dem).copy()
                mask = np.fliplr(mask).copy()
            # Random vertical flip
            if rng.random() > 0.5:
                dem  = np.flipud(dem).copy()
                mask = np.flipud(mask).copy()
            # Random 90-degree rotation
            k = int(rng.integers(0, 4))
            if k > 0:
                dem  = np.rot90(dem,  k).copy()
                mask = np.rot90(mask, k).copy()
            return dem, mask


    class RealCraterDataset(Dataset):
        """PyTorch Dataset for real LRO DEM patches from the DeepMoon package.

        Loads pre-processed arrays saved by the Day 2 notebook.

        Parameters
        ----------
        images_path : str or Path
            Path to .npy file with real DEM patches, shape (N, H, W).
        masks_path : str or Path
            Path to .npy file with binary masks, shape (N, H, W).
        indices : np.ndarray or None
            Subset of patch indices to use (from Day 2 quality filter).
        augment : bool
            If True, apply random flips and rotations.

        Returns
        -------
        image : torch.Tensor, shape (1, H, W), float32
        mask  : torch.Tensor, shape (1, H, W), float32
        """

        def __init__(
            self,
            images_path,
            masks_path,
            indices=None,
            augment: bool = True,
        ):
            self.images = np.load(images_path, mmap_mode='r')
            self.masks  = np.load(masks_path,  mmap_mode='r')
            self.indices = indices if indices is not None else np.arange(len(self.images))
            self.augment = augment

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int):
            real_idx = self.indices[idx]
            dem  = self.images[real_idx].astype(np.float32)
            mask = self.masks[real_idx].astype(np.float32)

            if self.augment:
                rng = np.random.default_rng()
                dem, mask = SyntheticCraterDataset._augment(dem, mask, rng)

            image_t = torch.from_numpy(dem[np.newaxis])
            mask_t  = torch.from_numpy(mask[np.newaxis])

            return image_t, mask_t


    class CombinedDataset(Dataset):
        """Interleaves synthetic and real patches during training.

        Returns one synthetic patch followed by one real patch alternately,
        so the model sees both domains every batch regardless of DataLoader order.

        Parameters
        ----------
        synthetic_dataset : SyntheticCraterDataset
        real_dataset      : RealCraterDataset
        """

        def __init__(self, synthetic_dataset, real_dataset):
            self.synth = synthetic_dataset
            self.real  = real_dataset
            self._len  = len(synthetic_dataset) + len(real_dataset)

        def __len__(self) -> int:
            return self._len

        def __getitem__(self, idx: int):
            if idx % 2 == 0:
                return self.synth[idx // 2 % len(self.synth)]
            else:
                return self.real[idx  // 2 % len(self.real)]
