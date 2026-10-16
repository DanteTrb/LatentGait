"""Pure software test generator. No fitted cohort parameters or real records are used."""

import numpy as np
import pandas as pd

from .cohort import COVARIATES, DOMAINS


def synthetic_frame(n=40, seed=173):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        dict(
            participant_id=[f"SOFTWARE_{j:04d}" for j in range(n)],
            trial_id=[f"GENERATED_{j:04d}" for j in range(n)],
            fall_last_year_binary=np.arange(n) % 2,
            qc_pass=1,
            steady_state_confirmed=1,
            retained_strides=24,
        )
    )
    clinical = {
        "age_years": (55, 5),
        "disease_duration_years": (4, 0.5),
        "updrs_iii": (20, 2),
        "hoehn_yahr": (2, 0.1),
        "ledd_mg_day": (300, 20),
        "gait_speed_m_s": (1.2, 0.1),
    }
    for name in COVARIATES:
        loc, scale = clinical[name]
        frame[name] = loc + scale * rng.normal(size=n)
    frame["sex_male"] = rng.integers(0, 2, n)
    for name, loc, scale in [
        ("height_cm", 170, 5),
        ("weight_kg", 70, 5),
        ("onset_age_years", 40, 2),
        ("cadence_steps_min", 110, 3),
        ("stride_length_m", 1.1, 0.1),
        ("stance_pct", 60, 1),
        ("swing_pct", 40, 1),
        ("double_support_pct", 20, 1),
        ("single_support_pct", 40, 1),
    ]:
        frame[name] = loc + scale * rng.normal(size=n)
    common = rng.normal(size=n)
    for name, columns in DOMAINS.items():
        for c in columns:
            frame[c] = 5 + 0.3 * common + 0.4 * rng.normal(size=n)
    return frame


def synthetic_protocol():
    return dict(
        protocol_version=2,
        seed=173,
        orientation={name: [1] * len(cols) for name, cols in DOMAINS.items()},
        orientation_rationale="Generated test convention only; no clinical interpretation",
        qc_protocol_reference="Generated test QC assertions only",
        measurement=dict(anchor_sd=0.1, loading_prior_sd=1.0, residual_prior_sd=0.5, acknowledge_archived_priors=True),
        bootstrap_scaling="refit",
        cumulative_rule="central_90",
        sampling=dict(chains=4, warmup=3000, draws=3000, target_accept=0.95),
    )
