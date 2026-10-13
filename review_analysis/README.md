# Post second-round review: additional analyses

This module implements additional analyses on a **user-supplied external
participant table**. It generates no input data and has no automatic fallback.
Its presence in the repository does not mean these analyses were executed on a
study cohort, validated clinically, or produced any particular manuscript result.

## Run

Activate the environment in `environment.yml`, then from the repository root:

```sh
python -m review_analysis.pipeline --input-csv /external/path/participants.csv --output /external/path/review_run
```

The notebook `10_post_second_round_review_additional_analyses.ipynb` exposes the
same workflow and requires an explicit path before execution. Both inputs and
outputs must be outside this repository. If `--output` is omitted, an external
temporary directory is created. Patient IDs are not exported in result tables.

## Required CSV schema

One row per independently verified participant; unique, nonempty `ID` values.
No aggregation or relabeling is performed. The data owner must resolve repeated
assessments and document inclusion rules beforehand: unique ID strings alone
cannot prove independence. Sample size is read from the file, never fixed to 269.

- `target_bin`: 0/1, retrospective occurrence of at least one fall in the preceding year.
- `Age`, `Duration_years`: primary adjustment covariates.
- `UPDRS_III`, `LEDD`, `Gait_Speed`: additional adjustment covariates.
- Kinematics: `Tilt`, `Obliquity`, `Rotation (range)`.
- Rhythmicity/recurrence: `HR V`, `HR ML`, `HR AP`, `iHR V`, `iHR ML`, `iHR AP`,
  `%det V`, `%det ML`, `%det AP`.
- Complexity: `MSE V`, `MSE ML`, `MSE AP`.

All required values must be finite. At least 30 complete rows and both outcome
classes are required. Missing values, duplicate IDs, constant columns and
rank-deficient feature sets produce errors. No clinical file is bundled.

This is a **new participant-table entry point**, not an exact replay of the
original assessment-row pipeline. It standardizes complete participant features
and covariates using population SD, then computes separate domain PCAs. It does
not silently reproduce the original median imputation or assessment aggregation.
Do not combine its outputs with older analyses without reconciling preprocessing,
cohort membership, orientation and latent scale.

## Analyses and assumptions

1. Outcome-independent domain PCA with deterministic signs; PC1, complexity-PC2,
   all-PC2 and >=80% within-domain variance retention.
2. Tight and relaxed anchoring; loading of kinematic PC1 fixed to 1. Tight-anchor
   residual SD is 0.1; relaxed residual SD is estimated. The downstream default
   uses the relaxed model. Fixed anchor loadings are not empirical contributions.
3. Primary/expanded logistic association models, reporting ORs, equal-tail 95%
   credible intervals and posterior directional probabilities.
4. Fixed/free probit and a natural cubic spline with fixed knots (-4,-1,1,4) on the
   score scale; all models use identical rows and primary covariates. These knots
   differ from the quantile-based clinical notebook 08 specification.
5. A modular approximation integrates 31 marginal score-posterior quantiles in
   the Bernoulli likelihood. Outcomes do not update the measurement model. Shared
   posterior dependence and uncertainty in standardization constants are not
   propagated. Check integration resolution for substantive use.
6. Conditional outcome PSIS-LOO and paired ELPD differences. PCA/measurement fits
   are held fixed: this is not nested validation of the complete pipeline.
   Flagged Pareto diagnostics require exact refits or K-fold, not implemented here.
7. Overall and score-quartile posterior predictive checks, not external or
   cross-validated calibration.
8. Derivative regions at 70%, 80%, 90% of each draw's positive maximum on central
   95% score support. Absent, disconnected and boundary-touching regions remain
   explicit. A tau interval is not a region width. No treatment window is inferred.
9. Hypothetical score-shift contrasts, with outputs outside the repository only.

Use `--smoke` only for execution checks. Defaults otherwise use four chains with
1500 tuning and 1500 retained draws each. Comparisons are marked reliable only
when not in smoke mode, relevant R-hat <=1.01, bulk ESS >=400, no divergences and
no PSIS warnings. This gate is a minimum diagnostic check, not scientific proof.

## Figures

With R packages `ggplot2` and `patchwork` installed:

```sh
Rscript review_analysis/plot_results.R /external/path/review_run
```

The plotting code requires files produced by this entry point and reads its
provenance. It does not reuse previously drawn clinical or illustrative curves.
Short-run plots are marked as execution tests. No fitted results are included.

## Validation status

Unit tests validate the input contract and numerical utilities using small
explicit software fixtures, not clinical data. No new full clinical analysis or
clinical validation is claimed. External validation, calibration slope/intercept,
nested added-value comparisons, physiological validation and treatment-response
heterogeneity are outside this implementation.

```sh
python -m unittest discover -s tests -p 'test_review_analysis.py'
python scripts/check_code_only.py
```
