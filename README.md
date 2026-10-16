# LatentGait replication package

Computational companion to *Latent gait instability underlying retrospective fall occurrence in Parkinson's disease* and its final Supplementary Sections S1–S8. One canonical command runs the analyses on an authorised local participant table. No participant data, identifiers, real-data samples, fitted cohort results, logs, posterior draws or generated figures are distributed.

Version 0.4.0 implements all reported analysis stages under a fixed, explicitly adopted specification for bootstrap, spline, LOO and curve summaries. See [the final protocol and source audit](docs/PROTOCOL.md) for the distinction between documented study methods, recovered historical code and decisions adopted for this package. The local configuration still requires the data custodian's QC and PCA-orientation inputs. A successful synthetic check establishes software execution; agreement with the paper's cohort results requires a private authorised comparison.

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

## Prepare the private input

Supply a UTF-8 CSV with **269 unique participants**, **105 fallers and 164 non-fallers**, one valid retained trial per person, complete features and covariates. The cohort lock applies identically to PCA, measurement, association, functional forms, contrasts, PPC and gait-speed OLS. There is no production bypass of the counts, row dropping, imputation, averaging of repeated trials, ID relabeling, synthetic fallback, or positional table linkage.

The [input schema](docs/INPUT_SCHEMA.md) specifies exact columns, units, outcome coding, record-level QC attestations and linkage keys. The package accepts a final feature table, including validated BTS exports for pelvic/spatiotemporal measures. Optional raw acceleration extraction implements HR, iHR, RQA %DET and scale-4 MSE; [the signal protocol](docs/SIGNAL_PROCESSING.md) describes its inputs, mandatory unresolved parameters and vendor boundaries.

If clinical/BTS and custom features are separate, align them explicitly:

```sh
latentgait assemble --clinical /secure/location/clinical_bts.csv \
  --features /secure/location/custom_features.csv \
  --output /secure/location/participants.csv
```

Both tables need matching `participant_id` **and** `trial_id`. The join rejects duplicates, conflicting columns and unmatched keys. Input, protocol and output paths must be outside the checkout. Treat every generated artifact as private, including aggregate results.

## One production workflow

```sh
cp configs/analysis.example.yaml /secure/location/analysis.yaml
# Resolve every null field and acknowledge the documented source of measurement priors.
latentgait validate --config /secure/location/analysis.yaml \
  --input /secure/location/participants.csv
latentgait run --config /secure/location/analysis.yaml \
  --input /secure/location/participants.csv \
  --output /secure/location/new-analysis-run
```

Add `--figures` to generate private PDF plots from the actual fitted distributions and probability draws (measurement distributions, primary association, functional-form/contrast panels, cumulative contrast). No stored manuscript curve or fitted number is used for plotting.

The output directory must not already exist. The run records code revision, input/config hashes, software versions, protocol decisions and completion status. It exports aggregate analytical summaries to that private directory; IDs, per-person scores, per-person contrasts and full posterior draws are not exported by the analysis runner. The optional extraction/assembly steps necessarily retain private linkage keys in their local intermediate CSVs.

The production run performs:

1. Separate standardized PCA of the 9 rhythmicity/recurrence, 3 complexity and 3 pelvic indicators, declared outcome-independent PC1 orientation, and 500 participant bootstrap resamples. Every resample refits standardization/PCA; participant draws are common across domains and PC1 signs are aligned to the reference. A separate 5,000-resample check extends the primary sequence, with interval-endpoint changes and component-instability diagnostics reported explicitly.
2. The primary anchored latent measurement model and three component-retention alternatives: complexity PC2, PC2 in each domain, and >=80% within-domain variance. Scores are standardized consistently; their posterior SDs are divided by the same scale as their posterior means.
3. Bayesian logistic association adjusted for age and disease duration; separate additions of UPDRS-III, H&Y, LEDD and gait speed; expanded adjustment; and an incremental complexity-PC2 model. Coefficients, 95% HDIs, ORs and directional probabilities are computed from posterior draws. ORs are exported only for logistic coefficients; probit/spline parameters retain their probit scale.
4. The paper's **unadjusted** fixed-slope probit, positive free-slope probit and restricted cubic spline probit. All propagate the same normal latent-score uncertainty. The fixed model uses tau ~ Normal(0,1); the free model uses tau ~ Normal(0,1.5), log(slope) ~ Normal(0,0.5); the spline uses predictor-quantile knots, standardized basis terms and hierarchical nonlinear priors.
5. Curves over 1,000 observed-range grid points, maxima of the derivative of posterior mean probability and connected >=80% derivative regions; added 70%/90% and 2,000-point checks; prior-only probability-curve diagnostics; PSIS-LOO with flagged-case exact refits; and the incremental PC2 predictive comparison. LOO integrates each held-out score's measurement distribution consistently across models; PCA/measurement are held fixed, so this is conditional outcome-model comparison, not end-to-end external validation.
6. Reference-model hypothetical shifts of 0.1/0.2/0.3/0.5 SD; pointwise intervals at grid-selected maxima; subject-specific posterior contrasts summarized only in aggregate; and an empirical weighted central-90% cumulative interval (5th–95th percentiles), with no KDE. The lower-tail90 convention is a secondary sensitivity. The central empirical convention is adopted for this package; it is not asserted to reproduce historical Figure S1.
7. Measurement/outcome PPC, rank R-hat, bulk/tail ESS, divergences and BFMI; and OLS of log gait speed on mean-centered age/disease duration.

All production Bayesian fits use four NUTS chains, 3,000 warm-up + 3,000 retained draws per chain, target acceptance .95. The common settings for upstream measurement/alternative fits are an explicit computational harmonization; original archived measurement code used fewer iterations. Diagnostics can fail, and failure is retained in the report. Completion alone is not evidence of adequate inference.

## Test the software without participant data

```sh
latentgait synthetic-check --output /external/new-synthetic-check --figures
```

This separate command generates 40 software records from scratch, runs every analysis stage with deliberately short chains, and marks the run as synthetic and unsuitable for inference. It uses 8 primary + 16 extended bootstrap resamples and 20 prior-curve draws; production uses 500 + 5,000 resamples and 500 prior-curve draws. It never runs automatically when an input file is absent. Exact LOO refits are separately unit tested; short synthetic execution skips expensive automatic refits and cannot mark comparisons reliable.

## Source traceability and interpretation

See [S1–S8 coverage](docs/METHODS_COVERAGE.md), [model decisions](docs/PROTOCOL.md), [software validation](docs/VALIDATION.md) and [migration history](docs/MIGRATION.md). The final DOCX filenames and SHA-256 hashes identify the source versions without redistributing the documents or extracting their cohort result tables into code. The old competing analysis entry points have been retired; their history remains recoverable in Git.

This retrospective observational framework estimates conditional associations and hypothetical fitted-probability contrasts. It establishes neither causal treatment effects, a biological threshold, a validated clinical cut-point, prospective prediction nor external transportability. Pelvic anchoring is an identifying assumption, and small score uncertainty under a tight anchor can be imposed by that assumption. The fixed probit's steepness is imposed by its link and unit slope. A derivative region is distinct from a credible interval for tau.

Please cite the manuscript and the repository reference in `CITATION.cff`. MIT license applies to the code; it conveys no permission to access or redistribute participant data.
