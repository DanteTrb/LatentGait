# Changes

## 0.4.0

Final adopted analysis specification (protocol version 2): refitted standardization and common participant draws in the 500-resample PCA bootstrap; separate 5,000-resample endpoint/component-instability check; fixed Harrell spline basis and documented priors with prior-curve diagnostics; existing marginalized conditional-outcome PSIS-LOO with exact flagged refits retained; primary derivative of posterior mean probability with added 70%/90% and 2,000-grid checks; empirical weighted central-90% cumulative interval without KDE, plus lower-tail90 sensitivity. New checks and the cumulative convention are identified as adopted reconstruction choices, not asserted as historical study procedures. Odds-ratio reporting is restricted to logistic coefficients; probit/spline coefficients are labelled on their native scale. No cohort data/results are included.

## 0.3.0

Single canonical S1–S8 workflow, strict subject/trial input contract, signal feature functions, source audit, synthetic software tests and code-only CI. Competing legacy entry points retired with history preserved in Git.
