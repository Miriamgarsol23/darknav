## Day 3

### Building src/synthetic.py

The production generator is a major upgrade over the scratch notebook code.
Key additions:

1. Elliptical distortion via axis rotation. This is important because real craters
   from oblique impacts are not circular. I sample ellipticity uniformly in [0, 0.3]
   and angle uniformly in [0, pi]. This directly maps to the triaxiality in dark matter
   halos, which was another connection I had not originally planned but emerged naturally.

2. Multi-crater patches (35% of the dataset). Real lunar terrain has overlapping craters
   from different epochs. The multi-crater generator places 2-3 NFW bowls at random
   positions and sums their contributions. The mask is recomputed on the final combined
   DEM rather than unioned from individual masks, which is more physically correct
   because the summed DEM changes the threshold crossings.

3. CombinedDataset interleaves synthetic and real patches within each batch. This ensures
   the model always sees both domains during every forward pass, which should reduce
   domain shift compared to sequential synthetic-then-real training.

### Building src/preprocessing.py

The preprocessing pipeline is a standalone Python script with a CLI interface
(python src/preprocessing.py --data_dir data --out_dir data/processed).
This is important for reproducibility: anyone can rerun the pipeline from scratch.

Key decision: I add an optional Sobel gradient channel (--sobel flag) that could be
used as a second input channel to the U-Net. I will not use it in the main experiment
but it is available for an ablation study if time permits.

The filter thresholds (MIN_COVERAGE=0.005, MAX_COVERAGE=0.60) were calibrated in Day 2.

### Class balance

The synthetic dataset has mean crater pixel coverage of approximately 0.08 to 0.15
depending on r_s distribution. This means roughly 85-92% of pixels are background.
Without correction the model will learn to predict all-background and still achieve
85-92% pixel accuracy. The fix is pos_weight in BCEWithLogitsLoss.

Formula: pos_weight = (1 - mean_coverage) / mean_coverage
At mean_coverage=0.10: pos_weight = 9.0

This means each crater pixel contributes 9x more to the loss than each background pixel.
Verified that this is the correct interpretation of PyTorch BCEWithLogitsLoss documentation.

### Ablation observations

The ablation figure (fig13) is the clearest demonstration of why each component matters:
- Pure NFW: bowl shape visible, no rim, no noise. Too clean to be realistic.
- + Ejecta rim: the raised rim is now visible in the cross-section. This is the key
  morphological feature for distinguishing crater from random bowl.
- + Noise: masks are no longer perfect circles because noise crosses the threshold
  at varying radii. This actually improves training because it teaches the model
  to be robust to imperfect boundaries.
- + Ellipticity: the masks become ellipses. Necessary for generalising to real craters.
