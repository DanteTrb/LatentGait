# LatentGait

Privacy-preserving replication code for *Latent gait instability underlying retrospective fall occurrence in Parkinson's disease*. It is an analytical workflow, **not** a clinical prediction tool, causal analysis, or source of patient data. No participant records, identifiers, cohort-derived samples, posterior draws, logs, numerical results, or manuscript figures are included.

## Scope and cohort lock

The final cohort is **269 unique participants**: 105 with and 164 without retrospective fall occurrence. The workflow rejects duplicate IDs and does not average or silently collapse repeated trials. Select one eligible, QC-passed trial per person upstream using a documented record-level protocol kept outside this repository.

Workflow: approved local source → record-level QC → subject-level lock → documented features → separate, oriented PCA within 9 rhythmicity–recurrence, 3 complexity, and 3 lower-trunk-kinematic inputs → 500-resample PC1 bootstrap → Bayesian measurement model → association/functional-form sensitivity. Fall outcome is never used to orient components or select features.

## Install and run

```bash
conda env create -f environment.yml
conda activate latent-gait
pip install -e '.[dev]'
pytest
cp configs/analysis.example.yaml /secure/location/analysis.yaml
# edit only the secure copy; write generated artifacts outside this repository
python -m latentgait.workflow --config /secure/location/analysis.yaml --output /secure/location/results
```

Tests generate data from scratch. Never commit authorised data, IDs, intermediate tables, figures, console logs, NetCDF draws, or outputs.

## Input schema

The local CSV needs one QC-passed row per subject, a nonblank local identifier, binary `fall_last_year_binary`, and configured features. Names are configurable.

| Domain | Inputs | Analysis role |
| --- | --- | --- |
| Rhythmicity–recurrence | HR, iHR, %DET in V/ML/AP (9) | independent standardized PCA |
| Neuromotor complexity | MSE in V/ML/AP (3) | independent standardized PCA |
| Lower-trunk kinematics | pelvis tilt, obliquity, rotation ranges (3) | independent PCA; anchored indicator |

Set PCA signs through a pre-outcome impairment-direction convention. The PCA routine returns PC1 variance, oriented loadings and 500-bootstrap percentile intervals without hard-coding a study result. Sensitivity analyses should add complexity PC2, all-domain PC2, and enough within-domain PCs for 80% variance, then compare scores and association estimates.

## Signal-processing and vendor boundaries

Supplementary S1 documents: at least 20 consecutive valid strides; removal of the initial/final two strides; vertical-acceleration events; 3.2-Hz FIR/wavelet event work; HR/iHR with demeaned, 20-Hz zero-lag fourth-order Butterworth filtering and 20 harmonics; %DET after 2,000-point RBF resampling and AMI/FNN/radius selection; and MSE after 2,000-point resampling, normalization, `m=2`, `tau=1`, `r=0.10 SD`, scale 4.

The manuscript leaves some selections as ranges or refers to an original custom routine. They are deliberate TODOs, not guessed defaults: iHR phase/intrinsic-harmonic convention; selected AMI/FNN/radius values; and numerical validation of MSE. Pelvis and spatiotemporal variables are BTS G-Walk exports. This project does not reverse-engineer proprietary BTS algorithms: retain vendor version, export fields and units in a local data dictionary.

## Models, diagnostics and interpretation

`models.py` implements the standard-normal, three-indicator latent model with lower-trunk PC1 as fixed-loading anchor. Posterior score SD is propagated in outcome models as a normal measurement-error approximation. Primary Bayesian logistic association: standardized latent score, age and disease duration. UPDRS-III, H&Y, LEDD, gait speed, and expanded adjustment are sensitivity models only; all associations are conditional and non-causal.

Use four NUTS chains, 3,000 warm-up + 3,000 retained draws each, `target_accept=.95`, and inspect rank R-hat, bulk/tail ESS and divergences. Carry out posterior predictive checks for measurement and outcome models.

`functional_form.py` provides fixed-slope probit, free-slope probit, and 4-knot restricted-cubic-spline sensitivity (5th/35th/65th/95th latent-score percentiles). Evaluate derivatives on a dense grid. The >=80%-of-maximum derivative region is descriptive and model-dependent, never a biological threshold. Compare models with PSIS-LOO, inspect Pareto-k, and use exact LOO refits when `k > .7`. Hypothetical 0.1/0.2/0.3/0.5 SD shifts are fitted-model contrasts, not interventions. The gait-speed reference model is descriptive only.

## Limitations and reference

The study is retrospective and cross-sectional: it cannot establish causal effects, prospective prediction, clinical cut-points, external validity, or transportability. See the manuscript and Supplementary S1–S8 for equations, priors, QC rationale and interpretation. Where implementation detail is omitted, create and preserve a protocol rather than guessing.

Please cite the manuscript and repository DOI in `CITATION.cff`.
