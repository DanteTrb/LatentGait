# Post second-round review: methodological analyses

This folder is separate from the clinical notebooks. It does not read, relabel,
duplicate or aggregate patient records. By default it generates **269 genuinely
independent synthetic individuals**, with synthetic IDs. Fall counts are random,
not forced to match the paper. No seed search or selection of favorable runs is
performed.

This is an executable simulation experiment, not a reconstruction of empirical
manuscript estimates. It cannot establish that an observed clinical transition
exists. Generating assumptions below are not fitted clinical results.

## Run

Use the repository's `environment.yml` (PyMC 5 / PyTensor), then from its root:

```sh
python -m simulation.pipeline --scenario gradual
python -m simulation.pipeline --scenario transition
python -m simulation.pipeline --scenario null
```

Use `--output /absolute/path/outside/LatentGait` to choose a persistent output
directory. Otherwise a new temporary directory is printed. Output inside the
repository is rejected. No generated datasets, figures, tables, traces or cell
outputs are committed. The notebook `10_post_second_round_review_simulation.ipynb`
provides the same entry point.

Optional scientific plots use R with `ggplot2` and `patchwork`:

```sh
Rscript simulation/plot_synthetic.R /absolute/external/run-directory
```

These plots preserve the loading/distribution, association and multipanel-curve
roles of Figures 1-3, using computed synthetic results only. They are not edits of
clinical figures and do not create a spatiotemporal PPC or cumulative-benefit
figure. All files carry a synthetic label. No favorable estimates are hard-coded.

```sh
python -m unittest discover -s tests -p 'test_simulation.py'
python -m simulation.pipeline --scenario gradual --n 40 --smoke
python scripts/check_code_only.py
```

`--smoke` uses two very short chains: it validates execution, **not inference**.
Production defaults use four chains with 1500 tuning and 1500 retained draws
each. Inspect R-hat, ESS, divergences and PSIS diagnostics before interpretation.
Do not interpret a nonconverged or unreliable run. Runtime depends on hardware.

## Generating assumptions (not results)

- Seed defaults to 20261012; sample size defaults to 269.
- A standard-normal latent variable contributes to domain axes with generating
  loadings 0.8, 0.6 and 0.35. Feature counts are 3, 9 and 3.
- Age/duration and three synthetic clinical covariates are generated, not copied
  from the clinical data. All simulated features are complete.
- Fall probability is `Phi(-0.3 + slope * latent + 0.2 * age + 0.1 * duration)`.
  Slopes are 0 (null), 0.25 (gradual), 0.90 (transition).
- A transition scenario **assumes** a steeper generating function. Recovery tests
  that assumption in synthetic data; it does not independently justify it for PD.

## What is implemented

1. Separate feature standardization and domain PCAs, outcome-independent signs.
2. Tight-anchor and relaxed-anchor Bayesian measurement models; no fall outcome
   enters estimation of the latent construct.
3. PC1, complexity-PC2, all-PC2 and >=80% variance retention strategies.
4. Fixed-slope probit, free-slope probit and parsimonious natural cubic spline on
   the same observations and age/duration covariates.
5. Modular integration over marginal latent-score posterior quantiles (31 nodes).
   Check quadrature resolution for a substantive application. Shared posterior
   dependencies and standardization-constant uncertainty are not propagated.
6. Primary/expanded logistic association models and component-retention
   associations, reporting ORs, equal-tail 95% credible intervals, and P(beta>0).
7. Pointwise conditional PSIS-LOO, paired delta-ELPD and SE; all Pareto warnings
   exported. **No exact-refit correction is implemented:** flagged comparisons
   are marked unreliable and require additional exact LOO or K-fold analysis.
8. Posterior predictive overall and score-quartile prevalence checks (internal
   checks, not cross-validated calibration).
9. Draw-wise derivative regions using 70%, 80% and 90% of each draw's positive
   maximum. Disjoint regions, absent positive slopes and boundary contact remain
   explicit. These are not tau credible intervals or treatment windows.
10. Illustrative fixed shifts, curve intervals and diagnostic/provenance exports.

## Important limits

The feature construction is synthetic and simplified; it is not an exact replay
of clinical preprocessing/missingness. The clinical notebooks remain unchanged in
their source cells. Additional PCs from the same domain are not independent
replicate indicators; the one-factor specification is only a sensitivity model.

Conditional LOO holds outcome-blind measurement/PCA estimates fixed. It is **not
nested validation of the full pipeline**. Added predictive value versus a single
PC, measurement invariance, calibration slope/intercept, bootstrap region
stability and treatment-response heterogeneity are not established by this code.
No assumed favorable values for these endpoints are exported as computed results.

All comparison results must be reported, including null, weak, nonconverged or
model-sensitive ones. There is no acceptance-probability calculation.

## Existing scripts

Keep `scripts/07_sensitivity_covariate_adjustment_bayesian.py`: it runs the
clinical complete-case covariate analysis, not this synthetic experiment.
Keep `scripts/08_pca_domain_axes_summary.py`: it summarizes clinical PCA loadings
and variance. It is not interchangeable with notebook 09's construct sensitivity.
Its positional matching to the processed axes should be verified against source
row ordering before any new clinical run; it is not imported by this simulation.
