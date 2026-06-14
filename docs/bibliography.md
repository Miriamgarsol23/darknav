# Bibliography: DarkNav

Author: Miriam Garcia Sollo
Last updated: June 2026

All entries include DOI or URL and access notes. Format: APA 7th edition.

---

## Section 1: Crater detection and terrain relative navigation

[1] Silburt, A., Ali-Dib, M., Zhu, C., Jackson, A., Valencia, D., Kissin, Y., Tamayo, D.,
    and Menou, K. (2019). Lunar crater identification via deep learning. Icarus, 317, 27-38.
    https://doi.org/10.1016/j.icarus.2018.06.022
    Code: https://github.com/silburt/DeepMoon
    Data: https://doi.org/10.5281/zenodo.1133969
    Notes: Primary baseline and dataset source. CNN trained on LRO-Kaguya merged DEM patches.
    Recovers 92 percent of human-labelled craters. Key insight: DEMs are illumination-invariant.

[2] Downes, L. M., Steiner, T. J., and How, J. P. (2020). Lunar terrain relative navigation
    using a convolutional neural network for visual crater detection. AIAA SciTech 2020.
    arXiv:2007.07702. https://arxiv.org/abs/2007.07702
    Notes: LunaNet architecture. Integrates CNN crater detections with Extended Kalman Filter
    for spacecraft state estimation. Defines the full TRN pipeline that this project serves.

[3] Kim, I., and Singh, S. (2024). Probabilistic regression for autonomous terrain relative
    navigation via multi-modal feature learning. Scientific Reports, 14, 29118.
    https://doi.org/10.1038/s41598-024-81377-z
    Notes: 2024 state of the art for TRN. Multi-modal CNN (intensity and depth) with cascading
    architecture. Demonstrates the active research status of the problem.

[4] Rijlaarsdam, D., et al. (2025). Optimizing deep learning models for on-orbit deployment
    through neural architecture search. Scientific Reports, 15.
    https://doi.org/10.1038/s41598-025-21467-8
    Notes: Directly addresses on-orbit CPU constraints. Neural Architecture Search to find
    smallest viable model. Shows 128x128 inputs are sufficient for small object detection.
    Key reference for the CPU-only deployment goal.

[5] Kechagias-Stamatis, O., and Aouf, N. (2021). Deep learning-based spacecraft relative
    navigation methods: a survey. Acta Astronautica, 191, 399-418.
    https://doi.org/10.1016/j.actaastro.2021.11.023
    arXiv:2108.08876
    Notes: Survey covering pose estimation, crater and hazard detection, asteroid navigation.
    Useful for framing the problem space and citing competing approaches.

[6] Rodda, M., McLeod, S., Pham, K. C., and Chin, T. J. (2024). Camera-pose robust crater
    detection from Chang'e 5. arXiv:2406.04569.
    https://arxiv.org/abs/2406.04569
    Notes: Evaluates Mask R-CNN for crater detection under off-nadir view angles. Relevant
    for understanding the limits of the segmentation approach.

[7] Bird, J., Colburn, K., Petzold, L., and Lubin, P. (2020). Model optimization for deep
    space exploration via simulators and deep learning. arXiv:2012.14092.
    https://arxiv.org/abs/2012.14092
    Notes: Deep learning for detecting exoplanets in bandwidth-limited deep space context.
    Establishes the principle that synthetic simulator data can replace scarce real data.

---

## Section 2: NFW profile and dark matter halo morphology

[8] Navarro, J. F., Frenk, C. S., and White, S. D. M. (1997). A universal density profile
    from hierarchical clustering. The Astrophysical Journal, 490, 493-508.
    https://doi.org/10.1086/304888
    Notes: Original NFW paper. Defines rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2].
    Universal across four orders of magnitude in halo mass. Foundation of the analogy.

[9] Navarro, J. F., Frenk, C. S., and White, S. D. M. (1996). The structure of cold dark
    matter halos. The Astrophysical Journal, 462, 563-575.
    https://doi.org/10.1086/177173
    Notes: Earlier companion paper. First derivation of the profile shape.

[10] Lucie-Smith, L., Peiris, H. V., Pontzen, A., and Lochner, M. (2023). Explaining dark
     matter halo density profiles with neural networks. arXiv:2305.03077.
     https://arxiv.org/abs/2305.03077
     Notes: Trains a CNN to learn latent representations of halo density profiles from N-body
     simulations. Demonstrates that CNNs can capture meaningful morphological structure from
     halo density fields. Validates the premise that halo morphology is learnable.

[11] Lucie-Smith, L., Peiris, H. V., and Pontzen, A. (2022). An interpretable machine-learning
     framework for dark matter halo formation. Monthly Notices of the Royal Astronomical
     Society, 515(2), 2164-2180.
     https://doi.org/10.1093/mnras/stac1833
     Notes: Follows up on [10]. Connects formation history to morphological features via
     interpretable ML. Good methodological reference.

[12] Bartelmann, M. (1996). Arcs from a universal dark-matter halo profile. Astronomy and
     Astrophysics, 313, 697-702.
     Notes: Derives the analytical projected surface density Sigma(R) from the NFW profile.
     Contains the piecewise F(x) formula used in the synthetic generator.

[13] Wright, C. O., and Brainerd, T. G. (2000). Gravitational lensing by NFW halos.
     The Astrophysical Journal, 534, 34-40.
     https://doi.org/10.1086/308744
     Notes: Full analytical derivation of Sigma(R) for the NFW profile. Reference for
     the implementation of the 2D projection in synthetic.py.

---

## Section 3: U-Net and segmentation architectures

[14] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: convolutional networks for
     biomedical image segmentation. Medical Image Computing and Computer-Assisted Intervention
     (MICCAI) 2015. arXiv:1505.04597.
     https://arxiv.org/abs/1505.04597
     Notes: Original U-Net paper. The encoder-decoder architecture with skip connections
     designed for segmentation with limited training data.

[15] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image
     recognition. CVPR 2016. arXiv:1512.03385.
     https://arxiv.org/abs/1512.03385
     Notes: ResNet-18 encoder used in U-Net backbone. Key reference for the encoder choice.

[16] Hashimoto, T., and Mori, M. (2019). Grid-based crater detection on LROC images using
     U-Net and Pix2Pix. Referenced in Silburt et al. review context.
     Notes: Established U-Net as a viable architecture specifically for DEM crater segmentation.
     Justifies the architecture choice over alternatives such as YOLO.

---

## Section 4: Datasets and planetary data

[17] Silburt et al. Zenodo dataset (2018). DeepMoon training data: DEM patches and crater CSV.
     https://doi.org/10.5281/zenodo.1133969
     Notes: 30,000 training images at 256x256, downsampled to 128x128 for this project.
     Includes crater positions and radii from two human-labelled catalogues.

[18] LOLA Team and Kaguya Team (2015). LRO-Kaguya merged DEM, 59 m per pixel.
     Lunar Planetary Data System.
     https://pds-geosciences.wustl.edu/missions/lro/lola.htm
     Notes: Global DEM used by Silburt et al. and as the source for the DeepMoon patches.

[19] Robbins, S. J. (2019). A new global database of lunar impact craters greater than
     1 km diameter: 1. Crater locations and sizes, comparisons with published databases,
     and global analysis. Journal of Geophysical Research: Planets, 124(4), 871-892.
     https://doi.org/10.1029/2018JE005592
     Notes: Ground-truth crater catalogue used for filtering and mask generation.
     Filtered in this project to diameters 2-16 km and eccentricity below 0.3.

[20] Roboflow crater_segm dataset (2022). 106 annotated crater images with instance masks.
     https://universe.roboflow.com/lunar-craters/crater_segm
     Notes: Small supplementary dataset useful for qualitative visualisation and sanity checks.

---

## Section 5: Domain adaptation and transfer learning

[21] Wilson, G., and Cook, D. J. (2020). A survey of unsupervised deep domain adaptation.
     ACM Transactions on Intelligent Systems and Technology, 11(5), 1-46.
     https://doi.org/10.1145/3400066
     Notes: Background reference for synthetic-to-real domain adaptation theory.

[22] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., and Abbeel, P. (2017).
     Domain randomisation for transferring deep neural networks from simulation to the real
     world. IEEE/RSJ IROS 2017. arXiv:1703.06907.
     https://arxiv.org/abs/1703.06907
     Notes: Foundation paper for using randomised synthetic data to bridge the sim-to-real gap.
     The NFW parameter sweep in this project is a form of domain randomisation.

---

## Section 6: Tools and software

[23] Bovy, J., et al. (2015). Galpy: a Python library for galactic dynamics. The
     Astrophysical Journal Supplement Series, 216(2), 29.
     Notes: halotools package used for NFWProfile() density calculations.
     https://halotools.readthedocs.io/

[24] van der Walt, S., et al. (2014). scikit-image: image processing in Python.
     PeerJ, 2, e453. https://doi.org/10.7717/peerj.453
     Notes: Circle fitting and morphological operations for post-processing.

[25] Selvaraju, R. R., et al. (2017). Grad-CAM: visual explanations from deep networks via
     gradient-based localisation. ICCV 2017. arXiv:1610.02391.
     https://arxiv.org/abs/1610.02391
     Notes: Method used for qualitative activation analysis in the evaluation phase.
