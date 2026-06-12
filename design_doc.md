# Design Document: DarkNav

Author: Miriam Garcia Sollo
Date: June 2026
Status: Working draft

---

## 1. Problem statement

Spacecraft navigating in deep space have no access to GPS. The dominant paradigm for
autonomous position estimation is Terrain Relative Navigation (TRN), in which the onboard
computer detects surface features in camera images and matches them against a known catalogue
to triangulate position without ground-station contact. Among all geological features on
airless bodies, impact craters are the most stable, geometrically defined, and catalogued
landmarks available for this purpose.

A crater detector suitable for real deployment must:
- Process 128 x 128 pixel DEM patches in under 100 ms on CPU-only hardware
- Achieve pixel-level Precision above 0.80 and Recall above 0.75
- Generalise across illumination conditions and noise levels
- Remain robust to partially degraded or overlapping craters

The core bottleneck is data scarcity. Real annotated DEM data from unexplored bodies does
not exist at training scale. This project addresses that gap.

---

## 2. Why Digital Elevation Maps and not optical images

Silburt et al. (2019) established the key insight: a crater's appearance in a DEM is
independent of solar illumination angle, because elevation is a physical property of the
terrain rather than a reflectance measurement. This removes a major source of domain shift
between training and deployment, which is particularly important for on-orbit inference
where the solar angle changes continuously with orbital position.

All training and evaluation in this project uses DEM patches derived from the LRO LOLA
instrument, not optical imagery.

---

## 3. The NFW-crater morphological analogy

The original contribution of this project is the use of projected NFW dark-matter halo
density profiles as synthetic training data for crater detection.

The Navarro-Frenk-White profile:

    rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]

where rho_s is the characteristic density and r_s is the scale radius. The concentration
parameter c = R_vir / r_s controls how peaked the profile is.

When this three-dimensional profile is projected along the line of sight, the resulting
surface mass density Sigma(R) produces a radially symmetric bowl shape with a smooth
transition between a central depression and a surrounding elevated rim. This is
morphologically equivalent to an impact crater in a DEM.

Both structures share the following properties:
- Approximate radial symmetry
- A characteristic scale radius
- A smooth transition between central depression and surrounding terrain
- Non-spherical perturbations in the real case (triaxiality for halos, oblique impact
  for craters)
- Embedding in a noisy background field

Hypothesis: pretraining a U-Net on synthetic NFW-derived images teaches the early
convolutional layers to detect radial symmetry and edge gradients characteristic of
circular cavities. This provides better weight initialisation for real crater detection
than ImageNet pretraining, which encodes object semantics irrelevant to DEM analysis.

Ejecta model: the synthetic generator also incorporates a power-law ejecta thickness
decay beyond the crater rim:

    t(r) = t_0 * (r / R_rim)^(-alpha)

where alpha is empirically between 2.8 and 3.5 for continuous ejecta deposits on
airless bodies. This adds the raised rim morphology missing from the pure NFW inversion.

---

## 4. Experiment design

Three conditions are compared on an identical held-out real DEM test set:

Condition A (baseline): U-Net trained from scratch using only real LRO DEM patches
Condition B (standard transfer): U-Net with ImageNet-pretrained ResNet-18 encoder, fine-tuned on real data
Condition C (DarkNav): U-Net pretrained on NFW synthetic data, fine-tuned on real LRO data

Primary metric: mean pixel-level IoU on the real test set.
Secondary metrics: Precision, Recall, F1, and CPU inference time per patch.

---

## 5. Architecture

U-Net with ResNet-18 encoder (Ronneberger et al. 2015 architecture, encoder from He et al. 2016).

Chosen for:
- Skip connections preserve spatial detail essential for accurate rim localisation
- Approximately 1.2M parameters, fits comfortably within 200 MB RAM
- Established baseline for crater segmentation in the literature
- CPU inference below 100 ms per 128 x 128 patch on a standard laptop

Architecture detail:
- Encoder: 4 ResNet-18 blocks with pretrained or NFW-pretrained weights
- Bottleneck: 512-channel feature map at 8 x 8 spatial resolution
- Decoder: 4 upsampling stages with skip connections from encoder
- Output: sigmoid activation, single-channel binary mask

Alternative considered and rejected: YOLOv8 with bounding box detection. Rejected because
pixel-level segmentation masks provide more precise rim coordinates for the downstream
Extended Kalman Filter integration than axis-aligned bounding boxes do.

---

## 6. Data pipeline

Training set size: approximately 1500 real DEM patches and 1500 synthetic NFW patches
Patch resolution: 128 x 128 pixels, single channel, float32
Normalisation: global min-max to [-1, 1]
Train / validation / test split: 70 / 15 / 15 percent with geographic separation
  (training patches drawn from latitudes -60 to +60, test patches from polar regions)

Augmentation applied during training:
- Random horizontal and vertical flip
- Random 90-degree rotation (0, 90, 180, 270)
- Gaussian noise with sigma drawn from U(0, 0.02)
- Random brightness shift of +/- 10 percent

Crater catalogue: Robbins (2019), filtered to:
- Diameter 2 km to 16 km
- Eccentricity below or equal to 0.3
- Depth-to-diameter ratio above 0.1

---

## 7. Metrics

IoU (pixel-level): TP / (TP + FP + FN), target above 0.70
Precision: TP / (TP + FP), target above 0.80
Recall: TP / (TP + FN), target above 0.75
F1: harmonic mean of Precision and Recall
CPU inference time: ms per 128 x 128 patch on a single core

---

## 8. Open questions to be resolved during development

- Does NFW pretraining actually improve IoU, or is the effect marginal?
  Answer expected: end of training phase.

- What concentration parameter c range produces synthetic craters most similar
  to real LRO craters? Answer expected: synthetic generation phase.

- Can the model be exported to ONNX and run with onnxruntime in under 100 ms?
  Answer expected: evaluation phase.

- Does adding the ejecta power-law rim model improve downstream fine-tuning
  compared to pure NFW inversion? Answer expected: ablation study in evaluation.
