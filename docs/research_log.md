# Research Log: DarkNav

Author: Miriam Garcia Sollo
Format: one entry per source, dated, written in my own words as I read.

---

## June 2026, Day 1

### Silburt et al. 2019 - Lunar Crater Identification via Deep Learning
Source: Icarus 317, 27-38. arXiv:1803.02192
Code and data: github.com/silburt/DeepMoon, zenodo 1133969

Read completely. This is the paper the entire project builds on.

Their main decision that I want to understand fully: they chose DEMs instead of optical
images. The reason is that optical crater appearance depends on the sun angle, and the
sun angle changes constantly. A crater lit from the left looks very different from the
same crater lit from above. A DEM records physical elevation, so it does not have this
problem. This matters enormously for any real deployment scenario, because you cannot
control illumination when you are orbiting a body.

Their CNN outputs a target image (not a binary mask) where each crater centre has a
Gaussian blob scaled by crater radius. I might want to replace this with a proper
binary segmentation mask using U-Net, which is more standard and gives cleaner outputs
for the downstream EKF matching step.

The Zenodo data package contains everything pre-processed: 30,000 training images at
256x256, with corresponding crater CSV files giving positions and radii. I plan to
downsample to 128x128 to reduce memory and training time, since my machine has no GPU.

Key number to remember: they recover 92 percent of the human-labelled test craters and
almost double the total number of detections. My baseline to beat.

The Moon-trained model also works on Mercury DEMs. This cross-body generalisation is
exactly what the NFW pretraining hypothesis predicts: if the model learns geometric
structure rather than Moon-specific texture, it should generalise.

Open question from this paper: their model uses a custom CNN output format. I want to
check whether a standard U-Net with binary cross-entropy loss performs comparably.

---

### Downes, Steiner and How 2020 - Lunar TRN Using CNN for Visual Crater Detection
Source: arXiv:2007.07702

This paper defines the full system that my crater detector would be part of. Their
pipeline is: CNN detects craters in camera frame, detected craters are matched to a
catalogue of known craters in the estimated spacecraft position region, matched craters
are fed as measurements into an Extended Kalman Filter that propagates spacecraft position.

LunaNet is their CNN. It outputs bounding boxes with confidence scores. The EKF takes
the (u, v, r) of each detected crater in pixel coordinates and converts it to a
line-of-sight measurement in inertial space.

What I learn from this: precision of the centre coordinate matters more than recall.
A missed crater is fine; a wrong crater coordinate corrupts the filter state. This
suggests I should tune my model to be precise rather than just to maximise IoU.

This also tells me what the output format should be: (x_centre, y_centre, radius) per
crater, not just a pixel mask. The mask is an intermediate step; circle fitting gives
the final navigation input.

---

### Navarro, Frenk and White 1997 - A Universal Density Profile from Hierarchical Clustering
Source: ApJ 490, 493-508

I know this paper from my MultiDark simulation project in physics. The NFW profile is
the standard model for dark matter halo density. The key formula is:

    rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]

At small r, the profile goes as 1/r (cusp). At large r, it goes as 1/r^3 (fast decay).
The transition happens around r = r_s, the scale radius.

The concentration parameter c = R_vir / r_s. A high-c halo is more concentrated (more
mass near the centre). A low-c halo is more diffuse.

The morphological argument for my project: if I project this 3D density distribution
along the line of sight z, I get a 2D surface density Sigma(R) that is radially symmetric
with a central peak decaying outward. When I invert this (high density becomes low
elevation), I get a bowl shape: central depression, smooth walls, gradual rise at large R.
This is what a simple circular crater looks like in a DEM.

The analytical projected formula comes from Bartelmann (1996) and Wright and Brainerd (2000).
I need to implement this in Python and verify it visually before moving to the generator.

Interesting question I want to test: does the concentration parameter c control something
analogous to the crater depth-to-diameter ratio? If c is high (peaked halo), the synthetic
crater is deep and narrow. If c is low, the crater is shallow and wide. I can sample c
from the empirical distribution of halo concentrations to get realistic variety.

---

### Kim and Singh 2024 - Probabilistic Regression for Autonomous TRN (Scientific Reports)
Source: doi:10.1038/s41598-024-81377-z

This is 2024 state of the art. They use a cascading CNN architecture that takes both
intensity images and depth maps as input, and outputs a probability distribution over
spacecraft position rather than a point estimate.

The probabilistic framing is interesting but beyond what I can implement in 10 days.
What matters for me is their result: they validate with Monte Carlo simulations and show
robust performance across multiple simulated scenarios. This confirms the problem is
unsolved and active.

One specific thing I note: they say depth maps from a stereo camera pair are much more
informative than monocular images for TRN. DEMs from orbit are effectively depth maps.
This retroactively justifies using LOLA DEM data rather than optical LRO imagery.

---

### Rijlaarsdam et al. 2025 - Optimizing DL Models for On-Orbit Deployment (Sci. Reports)
Source: doi:10.1038/s41598-025-21467-8

This is directly relevant to the CPU constraint. They apply Neural Architecture Search
to find the smallest model that meets accuracy requirements for on-orbit inference.
Their conclusion: EfficientNet-style models outperform ResNet at equivalent parameter
count for this task. Models with 1 to 2 million parameters are feasible on current
spaceborne hardware.

They also show that 128x128 input resolution is sufficient for detecting objects at the
scales relevant to TRN. This validates my choice of input size.

Key practical note: they benchmark inference time on an ARM Cortex-A9, which is a
representative embedded processor. A model that runs in under 100 ms on a modern laptop
CPU is roughly within range of what spaceborne hardware could achieve with an optimised
ONNX runtime.

I should export my final model to ONNX and benchmark inference time as part of the
evaluation. This makes the work relevant to deployment, not just to a benchmark dataset.

---

### Bartelmann 1996 and Wright and Brainerd 2000 - Analytical NFW projection
Sources: A&A 313, 697-702 and ApJ 534, 34-40

These two papers contain the analytical formula for the projected NFW surface density.
Bartelmann derived the formula in the context of gravitational lensing arc statistics.
Wright and Brainerd extended and corrected it.

The formula I will implement in synthetic.py:

    Sigma(R) = 2 * rho_s * r_s * F(x)
    where x = R / r_s

    F(x) is defined piecewise:
    - x < 1:  (1/(x^2 - 1)) * (1 - (1/sqrt(1 - x^2)) * arccosh(1/x))
    - x = 1:  1/3
    - x > 1:  (1/(x^2 - 1)) * (1 - (1/sqrt(x^2 - 1)) * arccos(1/x))

This is a smooth, continuous, analytically tractable function. I can implement it in
pure NumPy without any special libraries. The inversion (negate and normalise) gives me
the crater depth profile.

The rim will be added separately using the power-law ejecta model from the monograph.

---

### DeepMoon GitHub README and code
Source: github.com/silburt/DeepMoon

I went through the code structure carefully. The pipeline has three parts:
1. input_data_gen.py: crops random patches from the global DEM, generates target images
2. model_train.py: trains the CNN with their custom loss function
3. model_test.py: evaluates and extracts crater distributions

For my project I will not use their model_train.py directly. I will use only their
data generation concept and the Zenodo dataset, and build my own U-Net training pipeline
in PyTorch. Their code uses Keras/TensorFlow; mine will be PyTorch throughout.

The data format from Zenodo: hdf5 files with image arrays and CSV files with crater
(longitude, latitude, radius) for each patch. I need to convert the CSV coordinates
to pixel (x, y, r) for mask generation. This is the main preprocessing task in Day 3.

---

## Observations after Day 1 reading

The NFW analogy holds visually: the projected profile is a radially symmetric bowl with
a smooth rim, which is exactly what a fresh circular crater looks like in a DEM. The
code in the Day 1 notebook confirms this.

The key open question is whether the morphological similarity is deep enough to transfer
through learning. The halo has no sharp rim; the crater does. The ejecta power-law model
in the monograph addresses this by adding a raised rim on top of the NFW inversion.

Tomorrow I will implement the synthetic generator properly with both components and
run a visual comparison against real DEM patches from the DeepMoon dataset.

---

## Day 2

### DeepMoon dataset exploration (Silburt et al. Zenodo 1133969)

Loaded and inspected the full dataset. Key numbers:

Training split: the HDF5 contains patches at 256x256. After downsampling to 128x128
the memory footprint drops from ~7 GB to ~1.7 GB for the full training set, which is
manageable on a standard laptop with 8 GB RAM.

The crater CSV uses normalised coordinates: long and lat are in [0, 1] relative to the
patch, not geographic coordinates. radius is in km. To convert radius to pixels I need
km_per_px, which at 118 m/pixel original resolution and factor-2 downsampling is
approximately 0.24 km/px.

Coverage distribution: many patches have zero craters. These are useless for training
the positive class. After filtering to coverage in [0.005, 0.60] the valid set shrinks
but becomes much cleaner for training.

### Real vs synthetic comparison

The key figure of the project (fig9_real_vs_synthetic.png). The cross-sections show
the same qualitative bowl shape. The NFW synthetic is smoother because it has no
geological noise, overlapping structures, or asymmetric impacts. This is expected and
desirable: the synthetic provides clean geometric priors, the real data adds texture.

### Profile analysis

Extracted radial profiles from ~300 training patches. The mean real DEM profile and the
inverted NFW projected profile overlap well in the range r/r_crater = [0, 1.5].
The main discrepancy is at r/r_crater > 1.0 (the ejecta rim region), which the
power-law ejecta model in the synthetic generator is designed to address.

This is the empirical validation of the central hypothesis. The figure goes in the paper.

### Threshold calibration

Swept thresholds from -0.05 to -0.50. The mask radius tracks r_s best at threshold = -0.20.
Below -0.30 the mask becomes too small (only the very deepest pixels). Above -0.10 the mask
bleeds into the noise floor.

Decision: use threshold = -0.20 for the synthetic generator mask creation.

### Decisions made

1. Mask threshold: -0.20
2. r_s sampling: uniform U(8, 30) pixels
3. Training set: 1500 real + 1500 synthetic
4. km_per_px: 0.24 (used consistently in mask generation from real annotations)
