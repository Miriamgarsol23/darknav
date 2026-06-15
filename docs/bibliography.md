# Bibliography: DarkNav

Author: Miriam Garcia Sollo
Last updated: June 2026

These are the sources I actually read and used during the project.
Notes reflect my own understanding after reading each paper.

---

## Core papers: crater detection

[1] Silburt, A., Ali-Dib, M., Zhu, C., Jackson, A., Valencia, D., Kissin, Y.,
    Tamayo, D., and Menou, K. (2019). Lunar crater identification via deep learning.
    Icarus, 317, 27-38.
    https://doi.org/10.1016/j.icarus.2018.06.022
    Code: https://github.com/silburt/DeepMoon
    Data: https://doi.org/10.5281/zenodo.1133969

    This is the baseline the whole project builds on. Their key decision of using
    DEMs instead of optical images is what makes the detector robust to solar angle
    variation. The dataset (30,000 LRO patches with Robbins catalogue annotations)
    is the training data for the real conditions. They recover 92% of human-labelled
    craters and almost double total detections. Their Moon-trained model transfers
    to Mercury, which suggests the model is learning geometry, not Moon-specific
    texture. That generalisation is exactly what the NFW pretraining is designed
    to reinforce.

[2] Downes, L. M., Steiner, T. J., and How, J. P. (2020). Lunar terrain relative
    navigation using a convolutional neural network for visual crater detection.
    AIAA SciTech 2020. arXiv:2007.07702.
    https://arxiv.org/abs/2007.07702

    This paper defines the full TRN pipeline that my crater detector would feed into.
    The CNN detects craters in pixel coordinates, those are matched to a catalogue
    within the estimated spacecraft position region, and the matches are fed as
    measurements into an Extended Kalman Filter. Reading this made me understand
    that precision of the crater centre coordinate matters more than recall for
    the downstream filter — a missed crater is recoverable, a wrong coordinate
    corrupts the state estimate.

[3] Robbins, S. J. (2019). A new global database of lunar impact craters greater
    than 1 km diameter. Journal of Geophysical Research: Planets, 124(4), 871-892.
    https://doi.org/10.1029/2018JE005592

    The ground-truth catalogue used for mask generation. I filtered to diameters
    2-16 km and eccentricity below 0.3 to keep the training structures circular
    and within the resolution of the downsampled DEM patches.

---

## NFW profile and dark matter morphology

[4] Navarro, J. F., Frenk, C. S., and White, S. D. M. (1997). A universal density
    profile from hierarchical clustering. The Astrophysical Journal, 490, 493-508.
    https://doi.org/10.1086/304888

    The original NFW paper. I already knew this profile from my MultiDark project
    on dark matter halo morphology. The key formula is:
        rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]
    Universal across four orders of magnitude in halo mass. The concentration
    parameter c = R_vir / r_s controls how peaked the profile is, which I mapped
    to the depth-to-diameter ratio of craters in the synthetic generator.

[5] Wright, C. O., and Brainerd, T. G. (2000). Gravitational lensing by NFW halos.
    The Astrophysical Journal, 534, 34-40.
    https://doi.org/10.1086/308744

    Contains the analytical formula for the projected NFW surface density Sigma(R),
    which is the core of the synthetic generator. I implemented the piecewise F(x)
    function from this paper in src/synthetic.py. The critical numerical detail is
    that the x > 1 branch must use arctan(sqrt(x^2-1)) rather than arccos(1/x)
    for stability near x = 1 — I found this bug during Day 1 exploration.

---

## Architecture and deployment

[6] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: convolutional networks
    for biomedical image segmentation. MICCAI 2015. arXiv:1505.04597.
    https://arxiv.org/abs/1505.04597

    The architecture choice. U-Net's skip connections preserve spatial detail
    essential for accurate rim localisation at pixel level. The encoder-decoder
    design with skip connections was originally designed for segmentation with
    limited training data, which is exactly the constraint here.

[7] Rijlaarsdam, D., et al. (2025). Optimizing deep learning models for on-orbit
    deployment through neural architecture search. Scientific Reports, 15.
    https://doi.org/10.1038/s41598-025-21467-8

    Directly relevant to the deployment constraint. They show that models with
    1-2 million parameters are feasible on current spaceborne hardware, and that
    128x128 inputs are sufficient for small object detection. Their benchmark on
    ARM Cortex-A9 is the reference point for the 100ms inference target.

---

## Domain adaptation context

[8] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., and Abbeel, P. (2017).
    Domain randomisation for transferring deep neural networks from simulation to
    the real world. IEEE/RSJ IROS 2017. arXiv:1703.06907.
    https://arxiv.org/abs/1703.06907

    The theoretical foundation for using randomised synthetic data to bridge the
    sim-to-real gap. The sweep of NFW parameters (r_s, ellipticity, noise, alpha)
    in the synthetic generator is a form of domain randomisation. This paper
    justifies why variety in the synthetic distribution helps generalisation.

---

## Explainability

[9] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D.
    (2017). Grad-CAM: visual explanations from deep networks via gradient-based
    localisation. ICCV 2017. arXiv:1610.02391.
    https://arxiv.org/abs/1610.02391

    Method used for the activation analysis in notebooks/05_evaluation.ipynb.
    The Grad-CAM maps show that NFW pretraining (condition C) concentrates model
    attention on the crater rim more consistently than random initialisation,
    which is the qualitative evidence that the geometric prior transfers.
