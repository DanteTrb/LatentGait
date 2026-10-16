# LatentGait replication package

Computational companion and analytical pipeline for *Latent gait instability underlying retrospective fall occurrence in Parkinson's disease* and its final Supplementary Sections S1–S8. It supports the analyses reported in the manuscript. The anonymized analytical dataset is publicly available through Zenodo (DOI: [10.5281/zenodo.18095243](https://doi.org/10.5281/zenodo.18095243)). No participant identifiers, fitted cohort results, logs, posterior draws, or generated figures are distributed in this repository.

Version 0.4.1 implements the analytical stages reported in the manuscript, including the exploratory, hypothesis-generating latent biomechanical framework, functional-form sensitivity analyses, model-estimated hypothetical contrasts, and posterior predictive checks. PCA sign conventions and model specifications are documented in the public configuration and [protocol](docs/PROTOCOL.md).

## Data availability

The anonymized analytical dataset is publicly available on Zenodo: [10.5281/zenodo.18095243](https://doi.org/10.5281/zenodo.18095243).

## Code availability

The complete analytical pipeline supporting the analyses reported in the manuscript is available in this repository and archived on Zenodo: [10.5281/zenodo.21642236](https://doi.org/10.5281/zenodo.21642236).

## Install

Python 3.10 is the tested version. Use a fresh environment:

```sh
python3.10 -m venv /secure/location/latentgait-env
source /secure/location/latentgait-env/bin/activate
python -m pip install -c constraints-tested-py310.txt -e '.[dev]'
python -m pytest
python scripts/check_code_only.py
```

Alternatively, `conda env create -f environment.yml` installs the same package with its pinned core dependencies. `constraints-tested-py310.txt` also records the resolved transitive/test versions from a fresh Python 3.10 installation on macOS arm64. These are tested versions for this reconstruction, not a historical author-environment lock or a guarantee about platform/BLAS behavior. [Validation](docs/VALIDATION.md) describes the checks and their limits.

## Prepare the input

Use the anonymized Zenodo dataset to prepare a UTF-8 CSV with **269 unique participants**, **105 fallers and 164 non-fallers**, one valid retained trial per person, complete features and covariates. The cohort lock applies identically to PCA, measurement, association, functional-form sensitivity analyses, hypothetical contrasts, PPC, and gait-speed OLS. There is no production bypass of the counts, row dropping, imputation, averaging of repeated trials, ID relabeling, synthetic fallback, or positional table linkage.

The [input schema](docs/INPUT_SCHEMA.md) specifies exact columns, units, outcome coding, record-level QC attestations and linkage keys. The package accepts a final feature table, including validated BTS exports for pelvic/spatiotemporal measures. Optional raw acceleration extraction implements HR, iHR, RQA %DET and scale-4 MSE; [the signal protocol](docs/SIGNAL_PROCESSING.md) the signal protocol describes its inputs, required processing parameters, implementation assumptions, and vendor boundaries.

If clinical/BTS and custom features are separate, align them explicitly:

```sh
latentgait assemble --clinical /secure/location/clinical_bts.csv \
  --features /secure/location/custom_features.csv \
  --output /secure/location/participants.csv
```

Both tables need matching `participant_id` **and** `trial_id`. The join rejects duplicates, conflicting columns and unmatched keys. Input, protocol and output paths must be outside the checkout. Treat generated artifacts according to the applicable data-governance requirements.

## One production workflow

```sh
cp configs/analysis.example.yaml /secure/location/analysis.yaml
# Record the local QC protocol reference and acknowledge the measurement-prior source.
latentgait validate --config /secure/location/analysis.yaml \
  --input /secure/location/participants.csv
latentgait run --config /secure/location/analysis.yaml \
  --input /secure/location/participants.csv \
  --output /secure/location/new-analysis-run
```

Add `--figures` to generate PDF plots from the fitted distributions and probability draws (measurement distributions, primary association, functional-form sensitivity and hypothetical-contrast panels, cumulative contrast). No stored manuscript curve or fitted number is used for plotting.

The output directory must not already exist. The run records code revision, input/config hashes, software versions, protocol decisions and completion status. It exports aggregate analytical summaries; IDs, per-person scores, per-person contrasts and full posterior draws are not exported by the analysis runner. The optional extraction/assembly steps retain linkage keys in their local intermediate CSVs.

The production run performs:

1. Separate standardized PCA of the 9 rhythmicity/recurrence, 3 complexity and 3 pelvic indicators, fixed outcome-independent PC1 orientation, and 500 participant bootstrap resamples. Kinematics PC1 retains positive projection on pelvic excursions; rhythmicity/complexity PC1 have their largest absolute loading positive, with exact ties resolved by schema order. This is a sign convention, not a claim that larger domain scores mean worse function. Every resample refits standardization/PCA; participant draws are common across domains and bootstrap PC1 signs are aligned to the full-sample reference, not independently reoriented. 
2. The primary anchored latent measurement model and three component-retention alternatives: complexity PC2, PC2 in each domain, and >=80% within-domain variance. Scores are standardized consistently; their posterior SDs are divided by the same scale as their posterior means.
3. Bayesian logistic association adjusted for age and disease duration; separate additions of UPDRS-III, H&Y, LEDD and gait speed; expanded adjustment; and an incremental complexity-PC2 model. Coefficients, 95% HDIs, ORs and directional probabilities are computed from posterior draws. ORs are exported only for logistic coefficients; probit/spline parameters retain their probit scale.
4. The reference fixed-slope probit, positive free-slope probit and restricted cubic spline probit. All propagate the same normal latent-score uncertainty. The fixed model uses tau ~ Normal(0,1); the free model uses tau ~ Normal(0,1.5), log(slope) ~ Normal(0,0.5); the spline uses predictor-quantile knots, standardized basis terms and hierarchical nonlinear priors.
5. Functional-form sensitivity curves for the fixed-slope probit, free-slope probit, and restricted cubic spline probit over the observed range. The derivative-based high-sensitivity region is model-dependent and descriptive; it is not a biological or clinical threshold. PSIS-LOO model comparison and the incremental PC2 predictive comparison are also included. LOO integrates each held-out score's measurement distribution consistently across models; PCA/measurement are held fixed, so this is conditional outcome-model comparison, not end-to-end external validation.
6. Model-estimated hypothetical contrasts for reference-model shifts of 0.1/0.2/0.3/0.5 SD; pointwise intervals at grid-selected maxima; subject-specific posterior contrasts summarized only in aggregate; and an empirical weighted central-90% cumulative interval (5th–95th percentiles), with no KDE.
7. Measurement/outcome PPC, rank R-hat, bulk/tail ESS, divergences and BFMI; and OLS of log gait speed on mean-centered age/disease duration.

All production Bayesian fits use four NUTS chains, 3,000 warm-up + 3,000 retained draws per chain, with target acceptance .95. Sampling diagnostics are reported with the analysis outputs.

## Test the software without participant data

```sh
latentgait synthetic-check --output /external/new-synthetic-check --figures
```

This separate command generates 40 software records from scratch, runs every analysis stage with deliberately short chains, and marks the run as synthetic and unsuitable for inference. It uses 8 primary + 16 extended bootstrap resamples and 20 prior-curve draws; production uses 500 + 5,000 resamples and 500 prior-curve draws. It never runs automatically when an input file is absent. Exact LOO refits are separately unit tested; short synthetic execution skips expensive automatic refits and cannot mark comparisons reliable.

## Source traceability and interpretation

See [S1–S8 coverage](docs/METHODS_COVERAGE.md), [model decisions](docs/PROTOCOL.md), [software validation](docs/VALIDATION.md) and [migration history](docs/MIGRATION.md). The final DOCX filenames and SHA-256 hashes identify the source versions without redistributing the documents or extracting their cohort result tables into code. The old competing analysis entry points have been retired; their history remains recoverable in Git.

This exploratory, hypothesis-generating latent biomechanical framework estimates conditional associations, functional-form sensitivity, and model-estimated hypothetical fitted-probability contrasts. It establishes neither causal treatment effects, a biological or clinical threshold, a validated clinical cut-point, prospective prediction, nor external transportability. The high-sensitivity region is model-dependent and should not be interpreted as a biological or clinical transition point. Pelvic anchoring is an identifying assumption, and the fixed probit's steepness is imposed by its link and unit slope.

Please cite the manuscript and the repository reference in `CITATION.cff`. The MIT license applies to the source code only. Access to and reuse of the anonymized dataset are governed separately by the terms and licensing information provided with the Zenodo dataset.
