# Alignment of the final software specification with the manuscript

The package is ready to execute the agreed protocol. No participant analysis was run for the software update, so the update does not certify agreement of manuscript numbers with this specification. An authorised analyst can keep all inputs, generated summaries and the comparison private.

| Item | Status and required manuscript check |
| --- | --- |
| Cohort and primary model family | 269 unique participants, 105/164; one trial each; reported primary adjustment/prior structure retained. Confirm the custodian's QC inputs before execution. |
| PCA orientation | Protocol 3 fixes positive-excursion kinematics PC1 and largest-absolute-loading-positive rhythmicity/complexity PC1, with schema-order ties. The latter rule was available in historical notebook 09; its historical execution is not certified. Non-anchor PC/loading signs may reverse in displayed tables without changing the current symmetric-prior latent model. Keep labels/loadings consistent; do not reinterpret the sign convention as a clinical impairment direction. |
| Bootstrap | Historical code confirms refitted scaling and percentile intervals. Common resamples and explicit sign alignment are adopted refinements. Verify Table S2 after any rerun; identify the 5,000-resample check as added software/numerical sensitivity, if reported. |
| Spline | Fixed basis/scaling/intercept and S5 prior hierarchy are documented. Verify any affected curve, derivative-region and LOO summaries rather than asserting numerical identity to an unspecified historical implementation. |
| LOO | Marginalized conditional-outcome prediction target and flagged refits are explicit. Ensure ELPD descriptions match that target; no whole-pipeline or external validation claim. |
| Derivative region | Primary 1,000-point/80% posterior-mean-curve definition retained. The 70/90% and 2,000-point checks are additions, not retrospectively prespecified study analyses. |
| Cumulative 90% contrast | Final package uses empirical weighted 5th–95th percentiles without KDE. Historical notebook has different candidate procedures. If adopting this summary in Figure S1 or text, recalculate the corresponding endpoints and revise the method/caption together; do not reuse an old number under the new definition. |
| Odds ratios | Logistic coefficients only. Probit/spline coefficient exponentials are not ORs and are excluded from new reports. |

This is a targeted consistency check, not a request to add further exploratory analyses. Do not change specification to recover a preferred published number. No journal submission or manuscript-result revision is performed by the software workflow.
