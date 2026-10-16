# Canonical workflow and preserved history

Version 0.4.1 uses protocol version 3. Copy the current analysis example to the private location and transfer the approved local QC reference and measurement-prior acknowledgement. Keep the supplied public orientation rules/rationale: kinematics `[1, 1, 1]`, rhythmicity/complexity `largest_absolute_loading_positive`. Older version-2 direction-vector configurations are rejected with a migration message, not silently reinterpreted. No archived participant score file is needed, imported or published. Input tables and other model settings are unchanged.

Version 0.3 retires competing notebook and `review_analysis` entry points. Their source remains recoverable from Git history; no force-push or history rewriting is part of this change. The old notebooks used incompatible cohort reconstruction, median imputation/assessment aggregation, covariate adjustment, spline priors/knots, anchoring and sampling settings. They are not alternative commands for the final-paper package.

Use only `latentgait validate`, `latentgait run` and the optional preparation commands documented in README. `synthetic-check` is a separately identified software test; it cannot substitute for missing participant input. This migration also corrects earlier claims that every notebook had already been removed: notebooks 08–10 had remained, and are now retired with the other conflicting workflows.

The previous `src/latentgait` scaffold did not implement the full workflow. Its CLI ended at PCA; its latent anchor SD was 0.001, its free slope was unconstrained, it adjusted probits not stated as adjusted in the final equations, and it used fixed spline coefficient priors rather than S5's hierarchy. The implementation now follows the final sources and documents omitted details instead of claiming those earlier choices were faithful replications.

Removed files can be inspected with `git show c592fd9:<path>`. Restoring them does not make their methods consistent with the final protocol. Existing local user data and outputs are not migrated, modified or deleted by the package.
