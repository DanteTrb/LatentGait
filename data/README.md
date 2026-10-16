# Data are intentionally absent

Do not place participant data, identifiers, derivatives, QC logs, exported BTS files,
model output, or any cohort-derived artifact in this directory or in Git. Store and run
authorised material in an approved secure location outside the checkout.

Use `configs/analysis.example.yaml` as the schema contract. Before running, create one
QC-passed row per subject and document the deterministic trial-selection rule, feature
units, missing-data handling, BTS G-Walk version/export mapping, and unresolved signal
processing choices from Supplementary S1. The final lock validates 269 unique subjects,
105 fallers and 164 non-fallers unless a separate approved protocol says otherwise.
