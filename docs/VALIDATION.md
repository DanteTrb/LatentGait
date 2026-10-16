# Software verification and limits

The package is checked using values generated entirely from scratch. No real participant table, paper-derived coefficients, fitted result targets or synthetic records calibrated from the cohort are used in tests.

```sh
python -m pip install -c constraints-tested-py310.txt -e '.[dev]'
ruff check src tests
python -m pytest
python scripts/check_code_only.py
latentgait synthetic-check --output /external/new-software-check --figures
```

The tests cover strict cohort/QC/schema checks, duplicate/blank IDs after whitespace normalization, nonfinite values, the production cohort lock, one-to-one person+trial linkage despite shuffled input, fail-closed protocol fields, and external-output paths. They also check separate domain PCA and retention strategies, outcome-independent orientation, aligned bootstrap intervals, scale conversion, analytic HR/iHR parity/power values, entropy against independent brute-force counts, recurrence range, restricted-cubic linear tails/continuity, tensor/numerical basis agreement and finite gradients.

Bayesian graph tests compile finite initial log densities for all outcome forms, verify the free log-slope parameter and absence of unreported fixed-probit adjustment, check anchor residual SD through prior-predictive sampling, and compare analytic score-marginalized probit probabilities with independent numerical integration. The exact-LOO control-flow test forces one problematic case, verifies that its outcome is omitted from a real model graph, and checks the replacement predictive score; it does not claim a long-chain clinical refit was run.

The explicit synthetic command executes the four measurement/association strategies, five covariate sensitivities, incremental PC2 model, three functional forms, PPC, LOO calculations, derivative/contrast reporting and speed OLS. Its 20 warm-up/20 retained draws in two chains and eight bootstrap resamples intentionally exercise execution cheaply. Poor R-hat, low ESS, divergences and PSIS warnings in this short mode are expected and retained. The mode sets all comparisons unsuitable for inference and skips automatic exact LOO sampling refits; test the production refits under adequate sampling before interpreting their estimates.

All production Bayesian fits enforce 4 chains and 3,000/3,000 iterations at .95 target acceptance. Reported adequacy requires finite rank R-hat <=1.01, bulk and tail ESS >=400, and zero divergences. BFMI is reported separately. This is a declared software diagnostic gate, not a criterion stated by the manuscript or proof of model validity. A run with weak diagnostics is recorded as complete computationally but its inference remains flagged inadequate.

CI runs installation, static checks, unit/graph tests, a complete synthetic short execution and a post-run privacy scan. It uploads no outputs or logs as artifacts. Git hygiene checks reject tracked generated files and notebook execution results. These checks establish software behavior; exact cohort-number agreement, clinical validation and external transportability remain unverified.
