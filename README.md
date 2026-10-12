# LatentGait

[![Python](https://img.shields.io/badge/python-3.10-blue)]()
[![Bayesian Model](https://img.shields.io/badge/model-Bayesian-orange)]()
[![Wearable Biomechanics](https://img.shields.io/badge/domain-wearable%20biomechanics-purple)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18961895.svg)](https://doi.org/10.5281/zenodo.18961895)

Analytical code for:

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
├── 06_threshold_hypothetical_contrasts.ipynb
├── 08_model_form_sensitivity.ipynb
├── 09_pca_component_sensitivity.ipynb
└── 10_post_second_round_review_simulation.ipynb

scripts/
├── 07_sensitivity_covariate_adjustment_bayesian.py
├── 08_pca_domain_axes_summary.py
└── check_code_only.py

simulation/
├── README.md
├── __init__.py
└── pipeline.py

tests/
└── test_simulation.py
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

## Post second-round review: methodological analyses

The `simulation/` folder and notebook 10 are separate from the clinical analyses.
They generate independent synthetic individuals (default N=269), fit measurement,
association and functional-form models, and evaluate model-defined derivative
regions. They do not relabel clinical records or reproduce empirical manuscript
estimates. See
[simulation/README.md](simulation/README.md) for commands, assumptions and limits.

Notebook 08 compares functional forms on clinical data; notebook 09 examines
component retention. Their source code is retained. The existing scripts 07 and
08 are also retained because covariate-adjustment analysis and PCA summary
generation are distinct tasks, not redundant replacements for those notebooks.

## Code-only distribution

Generated figures, tables, datasets and model outputs are not part of the current
tracked tree. Existing local figures can remain in the ignored `figures/` folder.
Older Git commits may still contain previously published figures. Synthetic-run
outputs must be saved outside this repository.

Before committing, run:

```sh
python scripts/check_code_only.py --strip
python scripts/check_code_only.py
```

This preserves notebook source cells while clearing saved outputs and execution
counts. A GitHub Actions check rejects saved cell outputs and tracked artifacts.

---

## Data Availability

The clinical and wearable datasets used in the manuscript are not distributed in this repository because of ethical, privacy, and institutional restrictions.

This repository provides analytical code. To reproduce the clinical analysis,
users must supply an authorized dataset with the analytical structure required
by the clinical notebooks and scripts. Synthetic experiments need no clinical data.

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
