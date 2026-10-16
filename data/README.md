# Data are not stored in this repository

Do not place participant data, identifiers, derivatives, QC logs, exported BTS files,
model output, or any cohort-derived artifact in this directory or in Git. The anonymized
analytical dataset is publicly available on Zenodo at
[10.5281/zenodo.18095243](https://doi.org/10.5281/zenodo.18095243); download it and
store it outside the checkout before running the pipeline.

Use `docs/INPUT_SCHEMA.md` and `configs/analysis.example.yaml` as the schema contract. Before running, create one
QC-passed row per subject and document the deterministic trial-selection rule, feature
units, missing-data handling, BTS G-Walk version/export mapping, and unresolved signal
processing choices from Supplementary S1. The final lock validates 269 unique subjects,
105 fallers and 164 non-fallers. A different cohort requires an explicitly separate
analysis project; the canonical production command does not bypass this lock.
