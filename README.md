# LatentGait

[![Python](https://img.shields.io/badge/python-3.10-blue)]()
[![Bayesian Model](https://img.shields.io/badge/model-Bayesian-orange)]()
[![Wearable Biomechanics](https://img.shields.io/badge/domain-wearable%20biomechanics-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18961895.svg)](https://doi.org/10.5281/zenodo.18961895)

Analytical code and revised manuscript figures for:

**Latent gait instability underlying retrospective fall occurrence in Parkinson’s disease**

This repository contains the computational workflow supporting the revised manuscript. The analyses estimate a wearable-derived latent gait-instability construct from trunk biomechanics and examine its association with retrospective fall occurrence in Parkinson’s disease.

---

## Overview

Falls in Parkinson’s disease are multifactorial and are not expected to be fully explained by any single gait metric. This project evaluates whether trunk-derived wearable gait features can be integrated into a latent biomechanical construct representing gait instability, and whether this construct is associated with retrospective fall occurrence.

The workflow includes:

1. definition of the analytical variables and conceptual roles;
2. derivation of domain-specific trunk biomechanical axes;
3. Bayesian latent-variable modeling of gait instability;
4. comparison with gait speed as a conventional spatiotemporal descriptor;
5. Bayesian modeling of retrospective fall occurrence;
6. model-based threshold-region and hypothetical instability-reduction contrasts;
7. peer-review sensitivity analyses for additional covariate adjustment;
8. supplementary PCA summaries for domain-axis interpretation.

The repository is intended to document the analytical workflow used for the manuscript, not to provide a deployed clinical prediction tool.

---

## Repository Structure

```text
notebooks/
├── 01_exposure_definition_and_conceptual_roles.ipynb
├── 02_latent_trunk_axes.ipynb
├── 03_spatiotemporal_latent gait instability.ipynb
├── 04_contrast_gait_speed.ipynb
├── 05_bayesian_latent_gait_instability_and_falls_occurrence.ipynb
└── 06_threshold_hypothetical_contrasts.ipynb

scripts/
├── 07_sensitivity_covariate_adjustment_bayesian.py
└── 08_pca_domain_axes_summary.py

figures/
├── Figure1_revised.pdf
├── Figure2_revised.pdf
├── Figure3_revised.pdf
├── Figure4_revised.pdf
├── FigureS1_revised.pdf
└── FigureS2_revised.pdf
```

---

## Analytical Workflow

### 1. Conceptual and analytical variable definition

`01_exposure_definition_and_conceptual_roles.ipynb`

Defines the analytical dataset structure, retrospective fall-occurrence outcome, candidate biomechanical descriptors, and the conceptual roles of clinical and gait variables used in the manuscript.

### 2. Domain-specific trunk biomechanical axes

`02_latent_trunk_axes.ipynb`

Groups trunk-derived wearable features into biomechanical domains and derives domain-specific axes using principal component analysis.

The main domains include:

- rhythmicity and recurrence;
- neuromotor complexity;
- lower trunk kinematics.

### 3. Bayesian latent gait-instability model

`03_spatiotemporal_latent gait instability.ipynb`

Fits the Bayesian latent-variable model used to estimate participant-level latent gait instability from the domain-specific trunk biomechanical axes.

### 4. Contrast analysis with gait speed

`04_contrast_gait_speed.ipynb`

Evaluates gait speed as a conventional spatiotemporal comparator and examines whether the latent gait-instability construct captures information not reducible to gait speed alone.

### 5. Bayesian fall-occurrence model

`05_bayesian_latent_gait_instability_and_falls_occurrence.ipynb`

Models the association between latent gait instability and retrospective fall occurrence using Bayesian regression.

### 6. Threshold-region and hypothetical contrasts

`06_threshold_hypothetical_contrasts.ipynb`

Evaluates a model-estimated transition region along the latent gait-instability continuum and estimates model-based hypothetical changes in fall-occurrence probability under reduced latent instability.

### 7. Peer-review sensitivity analyses

`scripts/07_sensitivity_covariate_adjustment_bayesian.py`

Runs additional Bayesian covariate-adjustment sensitivity analyses requested during peer review, including adjustment for clinical severity and gait-related covariates.

### 8. PCA domain-axis summaries

`scripts/08_pca_domain_axes_summary.py`

Generates supplementary PCA summaries for the trunk biomechanical domains, including additional rhythmicity-recurrence components.

---

## Figures

Final revised manuscript figures are stored in `figures/`.

- `Figure1_revised.pdf`: revised analytical/conceptual workflow.
- `Figure2_revised.pdf`: revised latent gait-instability modeling framework.
- `Figure3_revised.pdf`: revised association between latent gait instability and retrospective fall occurrence.
- `Figure4_revised.pdf`: revised threshold-region and hypothetical contrast analysis.
- `FigureS1_revised.pdf`: revised supplementary figure S1.
- `FigureS2_revised.pdf`: revised supplementary figure S2.

---

## Data Availability

The clinical and wearable datasets used in the manuscript are not distributed in this repository because of ethical, privacy, and institutional restrictions.

This repository provides the analytical code and revised manuscript figures. To reproduce the full analysis, users must supply an equivalent dataset with the same analytical structure used by the notebooks and scripts.

---

## Reproducibility

The workflow was developed in Python and uses standard scientific-computing and Bayesian-analysis libraries, including:

- PyMC;
- ArviZ;
- pandas;
- NumPy;
- scikit-learn;
- statsmodels;
- matplotlib;
- seaborn.

Random seeds are fixed where appropriate to improve reproducibility.

Notebook cell outputs and generated result files are intentionally not stored in the repository. The repository is organized to document the computational workflow rather than to archive intermediate generated outputs.

---

## Citation

If you use this repository or build upon this analytical framework, please cite the associated manuscript and Zenodo release.

**Manuscript**

Trabassi D. et al.  
*Latent gait instability underlying retrospective fall occurrence in Parkinson’s disease.*

Bibliographic details will be updated upon publication.

**Repository DOI**

https://doi.org/10.5281/zenodo.18961895

---

## License

This repository is released under the MIT License.