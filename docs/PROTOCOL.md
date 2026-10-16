# Source audit and explicit model decisions

The target is the final manuscript and final supplement supplied for this update:

| Source | SHA-256 |
| --- | --- |
| Latent_gait_instability_FINAL_SUBMISSION_55_REFERENCES_CITATIONS_Woltemade.docx | d6458303f283233d8ef0552d911e279ac81b8dd5904a6574b9e7b4a794fde5b2 |
| SupplementaryMaterial_FINAL_SUBMISSION.docx | 503f8490290dfc78b61c34b716ed6ed29604df76ce78a88628113d4158c8e6eb |

These documents are the scientific specification. Earlier notebooks are evidence about historical code, not authority to override the final text. No published coefficient, loading, interval, curve or diagnostic result is embedded as a target or fallback. The 269/105/164 numbers are design/cohort checks, not fitted results.

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
| Bootstrap preprocessing | S2 specifies 500 resamples but not whether the scaler is refitted. Select `refit` or `fixed` explicitly. Each resample aligns sign to full-sample PC1 by dot product. Percentile 95% intervals are a declared implementation convention. |
| Higher-PC signs | Largest absolute loading positive, an outcome-independent software convention. PC1 follows the declared biomechanical direction. |
| Spline parameterization/intercept | The source does not identify a unique restricted-cubic basis or intercept prior. The package exposes a documented Harrell-type linear-plus-two-nonlinear truncated-power basis, population-SD column scaling, intercept Normal(0,1.5). These are explicit implementation choices; basis changes alter regularizing priors and can alter results. |
| Derivative posterior summary | Differentiate posterior probability draws on 1,000 full-observed-range points, then summarize derivative of posterior mean. Retain all connected >=80% sets and boundary contact. This summary convention is explicit because the text does not uniquely define posterior aggregation. |
| 90% cumulative interval endpoints | S6 gives cumulative absolute contrast but not a unique endpoint rule. Choose minimum-to-90% or central-90% explicitly. Neither is called a treatment window or tau interval. |
| Signal numerical choices | See SIGNAL_PROCESSING.md; they remain mandatory parameters, not undocumented defaults. |

## Uncertainty and LOO prediction target

The measurement model is fitted without falls. Standardized posterior mean/SD define Z_i ~ Normal(mu_i,sd_i) in the outcome stage. The regression likelihood can update its local Z_i; this implements the reported normal-error hierarchy rather than a plug-in predictor. It is an approximation: cross-person measurement-posterior dependence, PCA loading uncertainty and estimated scaling uncertainty are not propagated jointly. Bootstrap loadings are a stability analysis, not a posterior error model for PCA.

Outcome LOO holds full-cohort PCA, measurement summaries, knots and scaling fixed, and leaves out the fall outcome. For all models, its pointwise likelihood integrates over the upstream normal score distribution, not an outcome-updated individual score. Fixed/free probit integration is analytic; logistic/spline use Gauss-Hermite quadrature checked at 31 vs 63 nodes. If the check fails, inference stops. PSIS points above min(0.7, the sample-size-dependent good-k threshold) receive exact outcome refits using the same frozen representation. Held-out prediction uses the same marginalized target. Refitting does not change the upstream representation; this is not nested or external validation.

The source reports PSIS-LOO but does not fully define local-latent marginalization. This consistent implementation is identified explicitly and is not claimed to recover an undocumented original conditional-likelihood calculation. Paired ELPD standard errors use sqrt(n * sample variance(pointwise difference)). Results stay flagged unreliable when sampling/refit diagnostics fail.

## Consequence for replication claims

The repository supplies executable analysis stages and documented signal equations, with tests and a single production path. A dataset owner can run the package after resolving the local protocol. Exact reproduction of published numbers additionally requires the original authorised cohort and confirmation of the under-specified choices above. No software change can make those unknown choices identifiable from the paper. The package must not be described as having numerically reproduced the study until that comparison is actually performed by an authorised analyst.
