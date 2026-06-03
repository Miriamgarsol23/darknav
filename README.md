# DarkNav

**Autonomous Crater Detection for Deep-Space Navigation via Physics-Informed Synthetic Pretraining**

Author: Miriam Garcia Sollo
Repository: personal research project
Started: June 2026

---

## Overview

Spacecraft navigating in deep space cannot rely on GPS. The standard approach for autonomous
position estimation is Terrain Relative Navigation (TRN): the onboard computer identifies
surface features such as impact craters in camera images and matches them against a known
catalogue to estimate position without ground-station contact.

This project trains a lightweight U-Net CNN to detect craters in Digital Elevation Map (DEM)
patches from the Lunar Reconnaissance Orbiter (LRO), with one original contribution: the
network is pretrained on synthetic images derived from the projected Navarro-Frenk-White (NFW)
dark-matter halo density profile. The NFW profile, when projected in 2D, produces a radially
symmetric bowl shape that is morphologically analogous to an impact crater in a DEM. This
provides a physics-informed weight initialisation that is geometrically superior to generic
ImageNet pretraining, and addresses the core bottleneck of labelled data scarcity in planetary
science.

The model is designed to run on CPU-only hardware, targeting the constraints of embedded
spaceborne compute.

---

## Research questions

1. Does pretraining on NFW-profile synthetic DEMs improve IoU on real lunar crater detection
   compared to random initialisation and ImageNet pretraining?

2. What range of concentration parameter c and scale radius r_s produces synthetic craters
   that best approximate the morphology of real LRO DEM craters?

3. What is the minimum model size (number of parameters) that achieves IoU above 0.70 on
   the real test set while remaining viable for CPU inference?

---

## Project structure

```
darknav/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── design_doc.md        research design, architecture decisions, open questions
│   ├── research_log.md      dated notes on every paper read
│   └── bibliography.md      all sources with DOI and URL
├── notebooks/
│   ├── 00_scratch_day1.ipynb     NFW analogy visualisation and parameter sweep
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_gen.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_gradcam.ipynb
├── src/
│   ├── synthetic.py     NFW synthetic crater generator
│   ├── dataset.py       PyTorch Dataset class
│   ├── model.py         U-Net implementation
│   ├── train.py         training loop
│   └── evaluate.py      metrics and evaluation utilities
└── data/
    └── .gitkeep         data directory tracked but contents gitignored
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/darknav.git
cd darknav
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

The primary dataset is the DeepMoon package (Silburt et al. 2019), publicly available at:
https://doi.org/10.5281/zenodo.1133969

Download instructions:

```bash
pip install zenodo_get
cd data
zenodo_get 1133969
```

Real crater ground truth uses the Robbins (2019) lunar crater catalogue filtered to diameters
between 2 km and 16 km and eccentricity below 0.3.

---

## Architecture summary

Input: 128 x 128 DEM patch, single channel, float32, normalised to [-1, 1]
Model: U-Net with ResNet-18 encoder, approximately 1.2M parameters
Output: 128 x 128 binary segmentation mask (crater pixel / background pixel)
Post-processing: circle fitting to extract (x, y, radius) per detected crater
Primary metric: pixel-level Intersection over Union (IoU)

---

## References

Full annotated bibliography in docs/bibliography.md.

Core papers:
- Silburt et al. 2019, Icarus 317, 27-38
- Downes, Steiner and How 2020, arXiv:2007.07702
- Navarro, Frenk and White 1997, ApJ 490, 493
- Kim and Singh 2024, Scientific Reports, doi:10.1038/s41598-024-81377-z
- Rijlaarsdam et al. 2025, Scientific Reports, doi:10.1038/s41598-025-21467-8
