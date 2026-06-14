"""
dataset.py
==========
PyTorch Dataset classes for DarkNav.

Author: Miriam Garcia Sollo
Date:   June 2026

This module re-exports the Dataset classes from synthetic.py for
convenience and backwards compatibility with the project structure.

Classes
-------
SyntheticCraterDataset : on-the-fly NFW synthetic crater generator
RealCraterDataset      : loader for preprocessed LRO DEM patches
CombinedDataset        : interleaves synthetic and real patches
"""

from src.synthetic import (
    SyntheticCraterDataset,
    RealCraterDataset,
    CombinedDataset,
)

__all__ = [
    'SyntheticCraterDataset',
    'RealCraterDataset',
    'CombinedDataset',
]
