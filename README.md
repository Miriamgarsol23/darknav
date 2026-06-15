# DarkNav

**Autonomous Crater Detection for Deep-Space Terrain Relative Navigation using Physics-Informed Synthetic Pretraining**

*Miriam Garcia Sollo. June 2026*

---

## Why I built this

When I did the International Baccalaureate, I did an Extended Essay on the subject of Physics HL and I worked on a project
about dark matter halo morphology using the MultiDark N-body simulation. The core of that project was the
Navarro-Frenk-White profile (a formula that describes how dark matter density
distributes radially around a halo centre, forming a characteristic bowl shape that
peaks at the centre and decays outward).

Some months ago, while reading about autonomous spacecraft navigation, I came across the
problem of crater detection for Terrain Relative Navigation: how does a spacecraft
know where it is when there is no GPS and the round-trip communication delay to Earth
is measured in minutes? The answer is that it looks down at the surface, finds craters,
and matches them against a catalogue. And the standard input for this is a Digital
Elevation Map, which represents terrain as a height field.

When I saw what a crater looks like in a DEM for the first time, I recognised the
shape immediately. It is the same bowl (entral depression, smooth walls, elevated
rim, radially symmetric). The NFW profile projected in 2D and inverted is
morphologically the same structure as an impact crater.

That connection is what became this project. If the profiles are analogous, I could
use the NFW formula to generate unlimited synthetic training data for crater detection
analytically, in microseconds, with no expensive physics simulation. I wanted to
test whether a model pretrained on these synthetic NFW craters would learn something
geometrically useful that transfers to real lunar DEMs.

---

## What the project does

DarkNav trains a U-Net segmentation model to detect craters in 128x128 pixel patches
from the Lunar Reconnaissance Orbiter Digital Elevation Map. The model outputs a
binary mask labelling each pixel as crater or background. Three conditions are
compared:

- **Condition A**: trained from scratch on real LRO data only (baseline)
- **Condition B**: ResNet-18 encoder pretrained on ImageNet, fine-tuned on real data
- **Condition C**: pretrained on NFW synthetic craters, fine-tuned on real data

The hypothesis is that NFW pretraining teaches the encoder to detect radially symmetric
cavities before it ever sees real planetary data. This is especially relevant for
missions to unexplored bodies where no labelled data exists at all.

---

## How I approached it

I spent the first two days reading the literature carefully before writing a single
line of training code. The papers that shaped the project most were Silburt, who 
built the DeepMoon dataset and established that DEMs are better than
optical images for crater detection because they are illumination-invariant; Downes,
who showed how a crater detector feeds into a full TRN system with an
Extended Kalman Filter; and the original NFW paper,
which I already knew from the MultiDark project.

The analytical projection formula (the 2D integral of the NFW density along the line
of sight) came from Wright and Brainerd. Implementing it correctly took longer
than I expected, there is a numerical stability issue near x=1 where arccos(1/x) and
arctan(sqrt(x^2-1)) are mathematically equivalent but the arctan form is far more
stable in floating point. I found this by testing the output visually and noticing that
the cross-section profile was collapsing to near-zero. Fixing it was the first real
result of the project.

The synthetic generator also adds a power-law ejecta rim beyond the crater edge, based
on empirical decay exponents from impact geology (alpha between 2.8 and 3.5). Without
this the synthetic craters are too smooth, they have the bowl but not the raised rim
that real craters show in DEMs.

I built the training pipeline from scratch in PyTorch with three conditions, BCEWithLogitsLoss
with pos_weight to handle the class imbalance (crater pixels are around 0.4% of all
pixels in the real dataset), and an early stopping loop that saves the best checkpoint
by validation IoU. The full training for all three conditions runs from the command
line and took a few hours on CPU.

---

## Results

| Condition | IoU | Precision | Recall | F1 | CPU ms | ONNX ms |
|---|---|---|---|---|---|---|
| A: scratch | 0.1333 | 0.1398 | 0.7420 | 0.2351 | 30.8 | 20.1 |
| B: imagenet | 0.0883 | 0.0904 | 0.7946 | 0.1622 | 15.6 | 12.5 |
| C: DarkNav | 0.0913 | 0.0955 | 0.6856 | 0.1673 | 12.7 | 9.97 |

All three conditions pass the 100ms on-orbit inference target comfortably. The ONNX
export of condition C runs at under 10ms per patch on CPU, which is the most relevant
number for any real deployment scenario.

The IoU numbers are low across all conditions, and I think the honest explanation is
dataset size: after quality filtering, the real training set has 414 patches, which is
not enough to cleanly demonstrate the advantage of NFW pretraining over random
initialisation in a single scalar metric. The Grad-CAM analysis tells a more
interesting story, condition C concentrates activation along the crater rim more
consistently than the other two, which is exactly what you would expect if the NFW
geometric prior is being transferred. That qualitative result is in figure 23 in docs/.

The primary bottleneck is labelled data, not the method. With a larger real dataset
the NFW pretraining hypothesis could be tested more cleanly. That is the natural
next step.

---

## Project structure

```
darknav/
├── README.md
├── requirements.txt
├── docs/
│   ├── design_doc.md        problem statement, decisions, open questions
│   ├── research_log.md      notes on every paper read, day by day
│   ├── bibliography.md      9 sources I actually read, with annotations
│   ├── technical_report.md  full write-up with results and discussion
│   └── fig*.png             figures generated by the notebooks
├── src/
│   ├── synthetic.py         NFW generator and PyTorch Dataset classes
│   ├── preprocessing.py     HDF5 to numpy pipeline for real LRO data
│   ├── model.py             DarkNavUNet: scratch, imagenet, nfw modes
│   ├── train.py             training loop with CLI
│   └── evaluate.py          test evaluation, ONNX export, CPU benchmark
├── notebooks/
│   ├── 00_scratch_day1.ipynb      NFW profile visualisation and analogy validation
│   ├── 01_data_exploration.ipynb  real DEM dataset inspection and mask generation
│   ├── 02_synthetic_gen.ipynb     generator ablation and dataset creation
│   ├── 03_preprocessing.ipynb     preprocessing pipeline verification
│   ├── 04_training.ipynb          model architecture and training runs
│   ├── 05_evaluation.ipynb        test metrics and qualitative results
│   └── 06_gradcam.ipynb           Grad-CAM activation analysis
└── runs/
    ├── condition_A/
    ├── condition_B/
    ├── condition_C/
    └── evaluation_results.json
```

---

## Setup

```bash
git clone https://github.com/Miriamgarsol23/darknav.git
cd darknav
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The data directory is empty by design. Download the DeepMoon dataset:

```bash
pip install zenodo_get
cd data && zenodo_get 1133969 && cd ..
python src/preprocessing.py --data_dir data --out_dir data/processed
```

Generate the synthetic dataset:

```bash
python -c "
from src.synthetic import generate_synthetic_dataset
generate_synthetic_dataset(n_samples=1500, out_dir='data/synthetic', seed=42)
"
```

---

## Training from scratch

```bash
# Condition A: scratch baseline
python src/train.py --condition A --epochs 50 --out_dir runs

# Condition B: ImageNet pretrained encoder
python src/train.py --condition B --epochs 50 --out_dir runs

# Condition C: NFW pretraining then fine-tuning
python src/train.py --condition C_pretrain --epochs 30 --out_dir runs
python src/train.py --condition C_finetune --epochs 50 --out_dir runs \
    --nfw_checkpoint runs/condition_C/nfw_pretrain_best.pth
```

## Evaluation

```bash
python src/evaluate.py --proc_dir data/processed --runs_dir runs
```

---

## References

The nine sources I read for this project, with notes on how each shaped the work:

1. Silburt et al. 2019 . DeepMoon. Dataset and illumination-invariance argument.
2. Downes, Steiner and How 2020 . The TRN pipeline that this detector feeds into.
3. Robbins 2019 . Lunar crater catalogue used for ground-truth masks.
4. Navarro, Frenk and White 1997 . The NFW profile. The starting point of the idea.
5. Wright and Brainerd 2000 . Analytical 2D projection formula, implemented in synthetic.py.
6. Ronneberger, Fischer and Brox 2015 . U-Net architecture.
7. Rijlaarsdam et al. 2025. On-orbit model size and inference time benchmarks.
8. Tobin et al. 2017 . Domain randomisation, theoretical basis for the parameter sweep.
9. Selvaraju et al. 2017 . Grad-CAM, implemented in notebooks/06_gradcam.ipynb.

Full annotations: docs/bibliography.md
