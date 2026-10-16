from __future__ import annotations

import pandas as pd


class CohortError(ValueError):
    """Raised when data cannot form the prespecified one-row-per-subject cohort."""


def lock_subject_cohort(df: pd.DataFrame, *, id_col: str, fall_col: str, expected_n: int | None = 269) -> pd.DataFrame:
    """Validate the final analysis table; never aggregate repeated trials silently.

    The publication analysis has one eligible, QC-passed trial per unique participant.
    Selecting a trial (if needed) is a documented upstream decision, not an averaging step.
    """
    required = [id_col, fall_col]
    absent = [c for c in required if c not in df]
    if absent:
        raise CohortError(f"Required columns absent: {absent}")
    out = df.copy()
    if out[id_col].isna().any() or out[id_col].astype(str).str.strip().eq("").any():
        raise CohortError("Participant ID is missing or blank; do not manufacture an ID.")
    if out[id_col].duplicated().any():
        raise CohortError("More than one row per participant. Apply documented record-level QC before this step.")
    y = pd.to_numeric(out[fall_col], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise CohortError("Fall outcome must be complete and coded 0/1.")
    if expected_n is not None and len(out) != expected_n:
        raise CohortError(f"Expected {expected_n} unique participants, found {len(out)}.")
    return out.reset_index(drop=True)
