# Final adopted specification and source audit

The target is the final manuscript and final supplement supplied for this update:

| Source | SHA-256 |
| --- | --- |
| Latent_gait_instability_FINAL_SUBMISSION_55_REFERENCES_CITATIONS_Woltemade.docx | d6458303f283233d8ef0552d911e279ac81b8dd5904a6574b9e7b4a794fde5b2 |
| SupplementaryMaterial_FINAL_SUBMISSION.docx | 503f8490290dfc78b61c34b716ed6ed29604df76ce78a88628113d4158c8e6eb |

These documents are the scientific specification. Earlier notebooks are evidence about historical code, not authority to override the final text. No published coefficient, loading, interval, curve or diagnostic result is embedded as a target or fallback. The 269/105/164 numbers are design/cohort checks, not fitted results.

Version 0.4.0 / protocol version 2 fixes the bootstrap, spline, LOO and curve-summary choices agreed for the final public package. These decisions are complete computational specifications. They are not a claim that every choice was documented or used identically in the historical analysis. Additional numerical/descriptive checks below are identified as additions to this package, not retrospectively described as prespecified study analyses.

## Reported and directly implemented

The final manuscript specifies one valid trial per participant, a common complete-case cohort, separate 9/3/3 standardized domain PCAs, no rotation, primary PC1, three component alternatives and incremental complexity PC2. It specifies normal latent posterior error propagation in downstream Bayesian regressions. Logistic priors are intercept Normal(0,1.5), coefficients Normal(0,1). Primary age/duration adjustment and the listed sensitivity covariates are implemented on the same rows.

The fixed-slope probit is P(Y=1)=Phi(Z-tau), tau ~ Normal(0,1). It has no age/duration terms in its stated equation. S5's free-slope model is Phi(beta*(Z-tau)), tau ~ Normal(0,1.5), log(beta) ~ Normal(0,0.5). It is not an unconstrained Normal slope. The spline probit uses four observed-predictor quantile knots (.05,.35,.65,.95), standardized basis terms, a Normal(0,1) linear coefficient and nonlinear Normal(0,sigma_nl) coefficients with sigma_nl ~ HalfNormal(0.5). All forms use the same participants and score-error distributions.

The production settings are 4 chains x 3,000 warm-up + 3,000 retained draws, target acceptance .95. Applying these settings to upstream/alternative measurement fits is a computational harmonization: S3a explicitly prescribes downstream settings, while the archived original measurement model used 2,000/2,000. More iterations do not prove convergence.

## Missing details that cannot be inferred from result tables

| Choice | Evidence and handling |
| --- | --- |
| Exact QC exclusion and trial-selection decisions | Not recoverable from the text; custodian supplies an approved protocol and verified final table. No automatic selection or imputation. |
| Exact feature-level PC1 signs | Text describes biomechanical direction but does not uniquely specify all signed vectors. Rhythmicity/complexity vectors and rationale are mandatory in the local protocol. Kinematics uses positive projection on excursions. Fall status is excluded from sign selection. |
| Measurement anchor residual and priors | S3 fixes loading to 1 and calls residual variance small, but omits its value and full priors. Archived `b3ecf65`, notebook 05 cell 5 specifies latent Normal(0,1), free loadings Normal(0,1), free residual SD HalfNormal(0.5), anchor residual **SD 0.1** (variance 0.01). Config exposes these with a required acknowledgement. They are not presented as uniquely specified by the DOCX. |
| Standardization SD convention | Population SD (ddof=0) follows archived StandardScaler usage. The same score scale divides posterior SD; leaving SD unscaled would be inconsistent. |
| H&Y sensitivity encoding | H&Y enters as a standardized numeric stage, following archived covariate-sensitivity code. The final text does not uniquely specify ordinal contrasts; this encoding is disclosed rather than inferred from outcome fit. |
| Bootstrap preprocessing | `refit` is now fixed. Archived `b3ecf65`, notebook 02 cell 11 refits StandardScaler and PCA in every resample and reports percentile 95% intervals. The final package uses 500 primary subject resamples, common participant draws across domains, PC1 sign alignment by positive dot product, and separate 5,000-resample endpoint stability checks. Shared draws and explicit sign alignment are adopted implementation refinements. |
| Higher-PC signs | Largest absolute loading positive, an outcome-independent software convention. PC1 follows the declared biomechanical direction. |
| Spline parameterization/intercept | Fixed Harrell-type linear-plus-two-nonlinear truncated-power basis, population-SD column scaling, intercept Normal(0,1.5). The knots and spline priors follow S5; this exact basis/intercept choice is adopted explicitly. Basis changes alter regularizing priors and can alter results. Prior-only probability curves are reported as a diagnostic; priors are not tuned using the observed fall results. |
| Derivative posterior summary | Primary: derivative of posterior mean probability on 1,000 full-observed-range points, maximum positive derivative, connected >=80% sets and boundary flags. Added checks use 70%/90% and 2,000 points; they do not replace the primary definition according to their results. |
| 90% cumulative interval endpoints | Fixed `central_90`: empirical weighted 5th/95th percentiles of baseline scores, weights = absolute individual posterior-mean contrasts, no KDE, tied scores combined, endpoint weights included. Actual included mass is reported (at least 90% for nonzero weights). The minimum-to-90% convention is secondary only. Archived notebook 06 contains both a lower-tail procedure and a central KDE procedure; the new empirical central rule is an adopted reconstruction choice and is not certified as the rule that produced historical Figure S1. |
| Signal numerical choices | See SIGNAL_PROCESSING.md; they remain mandatory parameters, not undocumented defaults. |

## Uncertainty and LOO prediction target

The measurement model is fitted without falls. Standardized posterior mean/SD define Z_i ~ Normal(mu_i,sd_i) in the outcome stage. The regression likelihood can update its local Z_i; this implements the reported normal-error hierarchy rather than a plug-in predictor. It is an approximation: cross-person measurement-posterior dependence, PCA loading uncertainty and estimated scaling uncertainty are not propagated jointly. Bootstrap loadings are a stability analysis, not a posterior error model for PCA.

Outcome LOO holds full-cohort PCA, measurement summaries, knots and scaling fixed, and leaves out the fall outcome. For all models, its pointwise likelihood integrates over the upstream normal score distribution, not an outcome-updated individual score. Fixed/free probit integration is analytic; logistic/spline use Gauss-Hermite quadrature checked at 31 vs 63 nodes. If the check fails, inference stops. PSIS points above min(0.7, the sample-size-dependent good-k threshold) receive exact outcome refits using the same frozen representation. Held-out prediction uses the same marginalized target. Refitting does not change the upstream representation; this is not nested or external validation.

The source reports PSIS-LOO but does not fully define local-latent marginalization. This consistent implementation is identified explicitly and is not claimed to recover an undocumented original conditional-likelihood calculation. Paired ELPD standard errors use sqrt(n * sample variance(pointwise difference)). Results stay flagged unreliable when sampling/refit diagnostics fail.

## Fixed reporting and additional checks

PCA loading coefficients are unit-norm eigenvectors, not correlations between features and scores. The first 500 shared resamples define the primary bootstrap summary; the 5,000-resample check extends the same sequence and reports endpoint differences without selecting whichever interval looks more favorable. Eigenvalue gaps/ratios, PC1 alignment quantiles, and the fraction closer to reference PC2 are reported to reveal genuine component instability; no Procrustes rotation or component substitution is used to manufacture narrow PC1 intervals.

The primary derivative maximum belongs to the posterior mean curve. It is not an average of per-draw maximum locations. A 95% interval for tau does not define the derivative-region width. Probability/contrast intervals are pointwise equal-tailed 95% intervals, including those at the location selected from the posterior-mean contrast curve. They are not uncertainty intervals for the maximizing location.

Odds ratios and their HDIs are reported only for logistic regression coefficients. Exponentiated probit or spline coefficients are not odds ratios and are not exported as such. Parameter scale is labelled explicitly. The public package contains no fitted paper results.

## Consequence for replication claims

The repository supplies an executable, fixed analysis specification and documented signal equations, with tests and a single production path. The remaining local inputs are the authorised data, verified QC/trial-selection reference, PCA orientation rationale/vectors, archived measurement-prior acknowledgement, and signal-processing settings if extraction from raw signals is needed. These are not inferred from outcomes. Exact published-number agreement has not been tested on participant data. In particular, the adopted central-90% cumulative rule and corrected LOO/reporting conventions can change summaries relative to earlier code. An authorised analyst must align any corresponding manuscript numbers/captions before claiming numerical reproduction or submitting changed analytical results. Synthetic checks cannot establish this alignment.
