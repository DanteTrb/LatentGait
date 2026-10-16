# Authorised local input contract

All analysis inputs are one common complete-case participant table. Recruitment and exclusions are resolved upstream. The analysis does not attempt to reconstruct excluded individuals, infer identity from row position, trim an ID to its first token, manufacture IDs, average repeated trials or choose an outcome from conflicting records.

| Field | Contract |
| --- | --- |
| participant_id | Nonblank string identifying a unique verified person; leading zeroes preserved; whitespace stripped before duplicate checks |
| trial_id | Nonblank string identifying the selected valid trial; also used in explicit table linkage |
| fall_last_year_binary | 1 = at least one self-reported fall in preceding year; 0 = none; both classes required |
| qc_pass | 1 confirms acquisition inspection and eligibility/QC approval by data custodian |
| steady_state_confirmed | 1 confirms first/last two strides excluded and steady-state data verified |
| retained_strides | Integer >=20 AFTER trimming; requires adequate acquisition length |
| age_years | Age in years, positive |
| disease_duration_years | Duration in years, nonnegative |
| updrs_iii | UPDRS-III score, nonnegative |
| hoehn_yahr | H&Y stage, 1–4 eligibility range |
| ledd_mg_day | LEDD in mg/day, nonnegative |
| gait_speed_m_s | Positive gait speed in m/s from validated BTS export |
| sex_male | 1 = male, 0 = female, explicitly mapped from source coding |
| height_cm, weight_kg | Positive values for descriptive S1 summaries |
| onset_age_years | Nonnegative age at disease onset, no greater than age |
| cadence_steps_min, stride_length_m | Positive BTS cadence/stride length for Table 1 and S1 |
| stance_pct, swing_pct, double_support_pct, single_support_pct | BTS phase percentages 0–100 for S1 |
| pelvis_tilt_range, pelvis_obliquity_range, pelvis_rotation_range | Nonnegative excursion ranges in degrees (S1), with angle/range definition confirmed in local vendor dictionary |
| hr_v, hr_ml, hr_ap | Harmonic ratios, dimensionless |
| ihr_v, ihr_ml, ihr_ap | Improved harmonic ratios, percentages 0–100 |
| det_v, det_ml, det_ap | RQA determinism, percentages 0–100 |
| mse_v, mse_ml, mse_ap | Scale-4 sample entropy, natural logarithm units |

Every numeric value must be finite. Constant columns and rank-deficient domains fail before analysis. IDs are retained only in memory by the analysis runner. Error messages do not echo record IDs or measurements. QC flags are attestations, not a substitute for source verification: unique strings cannot prove that participants are distinct, that ON-medication requirements were met, or that raw signal quality is acceptable.

PCA feature order is fixed: kinematics uses tilt, obliquity, rotation; rhythmicity uses HR V/ML/AP, iHR V/ML/AP, DET V/ML/AP; complexity uses MSE V/ML/AP, as listed above. For rhythmicity/complexity PC1 and every higher PC, the largest absolute loading is positive; exact ties select the first feature in that order. Kinematics PC1 retains positive projection on `[1, 1, 1]`. The convention does not change the input measurements or assert that higher domain scores are worse. See PROTOCOL.md for historical provenance and bootstrap alignment.

Before running, record a local protocol reference covering recruitment eligibility, ON medication, sensor placement at L5, 100-Hz calibration and axes, visual acquisition inspection, full-stride event verification, trial selection, trimming and complete-case exclusions. Six exclusions are described in the manuscript, but their record-level reasons cannot be reconstructed from the text. No reason is fabricated by this package.

For `assemble`, use the same two linkage keys in both tables. Store custom features in the feature table, and clinical/BTS variables plus QC flags in the clinical table. If raw extraction provides `retained_strides`, it must appear in only that table: duplicate non-key columns are rejected rather than reconciled silently. A shuffled feature table is safe because linkage uses explicit keys; unmatched person/trial sets fail.

File paths are supplied explicitly on the command line and must be outside the repository. No data are downloaded. The analysis runner's generated summaries and file hashes remain private; do not upload them to Git, CI, issues or a PR. The publication data-access statement does not grant this software permission to obtain or redistribute data.
