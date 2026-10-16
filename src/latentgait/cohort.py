"""One common complete participant table; no imputation, aggregation or row linkage."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = {
    "kinematics": ["pelvis_tilt_range", "pelvis_obliquity_range", "pelvis_rotation_range"],
    "rhythmicity": [f"{family}_{axis}" for family in ("hr", "ihr", "det") for axis in ("v", "ml", "ap")],
    "complexity": ["mse_v", "mse_ml", "mse_ap"],
}
COVARIATES = ["age_years", "disease_duration_years", "updrs_iii", "hoehn_yahr", "ledd_mg_day", "gait_speed_m_s"]
DESCRIPTIVES = [
    "sex_male",
    "height_cm",
    "weight_kg",
    "onset_age_years",
    "cadence_steps_min",
    "stride_length_m",
    "stance_pct",
    "swing_pct",
    "double_support_pct",
    "single_support_pct",
]
FEATURES = [c for columns in DOMAINS.values() for c in columns]
QC = ["qc_pass", "steady_state_confirmed", "retained_strides"]
REQUIRED = ["participant_id", "trial_id", "fall_last_year_binary", *QC, *COVARIATES, *FEATURES, *DESCRIPTIVES]


@dataclass
class Cohort:
    frame: pd.DataFrame
    digest: str

    @property
    def y(self):
        return self.frame.fall_last_year_binary.to_numpy(dtype=int)


def zscore(x):
    """Population-SD scaling, including correct conversion of posterior SDs later."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Missing/nonfinite values cannot be standardized")
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
    if np.any(sd <= 1e-12):
        raise ValueError("Constant/near-constant variable cannot be standardized")
    return (x - mean) / sd, mean, sd


def validate_frame(frame, expected=(269, 105, 164)):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError("Missing schema columns: " + ", ".join(missing))
    d = frame[REQUIRED].copy()
    for c in ("participant_id", "trial_id"):
        ids = d[c].astype("string").str.strip()
        if ids.isna().any() or ids.eq("").any():
            raise ValueError(f"Missing/blank {c}; identifiers are never generated")
        d[c] = ids
    if d.participant_id.duplicated().any():
        raise ValueError("Repeated participant ID; resolve one valid trial per person externally")
    numeric = ["fall_last_year_binary", *QC, *COVARIATES, *FEATURES, *DESCRIPTIVES]
    d[numeric] = d[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[numeric].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite record; final cohort must be complete across ALL analyses")
    if set(d.fall_last_year_binary.unique()) != {0, 1}:
        raise ValueError("Outcome must be binary 0/1 with both classes present")
    if not d.qc_pass.eq(1).all() or not d.steady_state_confirmed.eq(1).all():
        raise ValueError("Record-level QC/steady-state confirmation failed")
    strides = d.retained_strides.to_numpy()
    if np.any(strides < 20) or np.any(strides != np.floor(strides)):
        raise ValueError("At least 20 complete retained strides are required AFTER boundary trimming")
    n, f = len(d), int(d.fall_last_year_binary.sum())
    if expected is not None and (n, f, n - f) != tuple(expected):
        raise ValueError("Cohort counts differ from the prespecified (unique subjects, fallers, non-fallers) lock")
    if len(d) <= max(map(len, DOMAINS.values())):
        raise ValueError("Insufficient rows for domain PCA")
    if (d.gait_speed_m_s <= 0).any() or (d.age_years <= 0).any():
        raise ValueError("Gait speed and age must be positive")
    if (d[["disease_duration_years", "updrs_iii", "ledd_mg_day"]] < 0).any().any():
        raise ValueError("Negative clinical value")
    if not d.sex_male.isin([0, 1]).all():
        raise ValueError("sex_male must be 1 male, 0 female; no source recoding is inferred")
    if (d[["height_cm", "weight_kg", "cadence_steps_min", "stride_length_m"]] <= 0).any().any():
        raise ValueError("Invalid anthropometric or spatiotemporal value")
    if (d.onset_age_years < 0).any() or (d.onset_age_years > d.age_years).any():
        raise ValueError("Invalid disease onset age")
    if (
        not d[["stance_pct", "swing_pct", "double_support_pct", "single_support_pct"]]
        .apply(lambda x: x.between(0, 100))
        .all()
        .all()
    ):
        raise ValueError("Spatiotemporal phase values must be percentages 0–100")
    if not d.hoehn_yahr.between(1, 4).all():
        raise ValueError("H&Y must be within the manuscript's 1–4 eligibility range")
    for family in ("ihr", "det"):
        if not d[[f"{family}_{a}" for a in ("v", "ml", "ap")]].apply(lambda x: x.between(0, 100)).all().all():
            raise ValueError("iHR and %DET must use the 0–100 percentage scale")
    if (d[[f"hr_{a}" for a in ("v", "ml", "ap")] + DOMAINS["kinematics"]] < 0).any().any():
        raise ValueError("Negative harmonic ratio or pelvic range")
    zscore(d[[*COVARIATES, *FEATURES]])
    for columns in DOMAINS.values():
        x = d[columns].to_numpy()
        if np.linalg.matrix_rank(x - x.mean(0)) < len(columns):
            raise ValueError("Rank-deficient domain; higher-component sensitivity is undefined")
    # Preserve row order within the single locked table. All downstream arrays use this order.
    return d.reset_index(drop=True)


def load_cohort(path, expected=(269, 105, 164)):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".csv":
        raise ValueError("Supply an explicit UTF-8 participant-level CSV")
    payload = path.read_bytes()
    frame = pd.read_csv(BytesIO(payload), dtype={"participant_id": "string", "trial_id": "string"})
    return Cohort(validate_frame(frame, expected), sha256(payload).hexdigest())


def link_tables(clinical, features):
    """Explicit subject+trial keys; forbid positional linkage or partial inner joins."""
    keys = ["participant_id", "trial_id"]
    frames = [clinical.copy(), features.copy()]
    for frame in frames:
        for key in keys:
            if key not in frame:
                raise ValueError("Linkage requires participant_id AND trial_id in both tables")
            frame[key] = frame[key].astype("string").str.strip()
            if frame[key].isna().any() or frame[key].eq("").any():
                raise ValueError("Missing linkage key")
        if frame.participant_id.duplicated().any():
            raise ValueError("Duplicate participant before one-to-one linkage")
    overlap = set(frames[0].columns) & set(frames[1].columns) - set(keys)
    if overlap:
        raise ValueError("Overlapping non-key columns require explicit reconciliation")
    merged = frames[0].merge(frames[1], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Participant/trial key sets differ; no row may be silently discarded")
    return merged.drop(columns="_merge")
