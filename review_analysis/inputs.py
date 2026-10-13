"""Strict input contract. No generated fallback, ID relabeling or silent filtering."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DOMAINS = {
    'kinematics': ['Tilt', 'Obliquity', 'Rotation (range)'],
    'rhythmicity': ['HR V', 'HR ML', 'HR AP', 'iHR V', 'iHR ML', 'iHR AP',
                   '%det V', '%det ML', '%det AP'],
    'complexity': ['MSE V', 'MSE ML', 'MSE AP'],
}
COVARIATES = ['Age', 'Duration_years', 'UPDRS_III', 'LEDD', 'Gait_Speed']
ROOT = Path(__file__).resolve().parents[1]


def load_participant_table(path):
    """Read a user-supplied CSV representing an already resolved analytic cohort.

    Units/eligibility and identity resolution must be verified by the data owner.
    Unique strings alone cannot establish that rows represent distinct people.
    """
    path = Path(path).expanduser().resolve(strict=True)
    if ROOT in path.parents or path == ROOT:
        raise ValueError('Input must be outside the repository.')
    if path.suffix.lower() != '.csv':
        raise ValueError('Provide a participant-level CSV, not a workbook or model output.')
    payload = path.read_bytes()
    frame = pd.read_csv(BytesIO(payload), dtype={'ID': 'string'})
    features = [c for cols in DOMAINS.values() for c in cols]
    required = ['ID', 'target_bin', *COVARIATES, *features]
    missing = sorted(set(required)-set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: '+', '.join(missing))
    ids = frame.ID.str.strip()
    if ids.isna().any() or ids.eq('').any():
        raise ValueError('Missing/empty participant IDs. No IDs will be generated.')
    if ids.duplicated().any():
        raise ValueError('Repeated participant IDs: resolve repeated assessments externally; no relabeling is permitted.')
    if len(frame) < 30:
        raise ValueError('At least 30 complete participants are required for this workflow.')
    numeric = frame[['target_bin', *COVARIATES, *features]].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Missing/nonfinite measurements: supply a documented common analytic cohort; no rows are silently dropped.')
    if set(numeric.target_bin.unique()) != {0, 1}:
        raise ValueError('target_bin must be binary 0/1 and include both outcome classes.')
    if (numeric[[*COVARIATES, *features]].std(ddof=0) <= 1e-12).any():
        raise ValueError('Constant feature/covariate: standardization would be undefined.')
    for domain, columns in DOMAINS.items():
        x = numeric[columns].to_numpy(dtype=float)
        if np.linalg.matrix_rank(x-x.mean(axis=0)) < len(columns):
            raise ValueError(f'Rank-deficient {domain} features: resolve redundant indicators before analysis.')
    return dict(ids=ids.to_numpy(),y=numeric.target_bin.to_numpy(dtype=int),
        covariates=StandardScaler().fit_transform(numeric[COVARIATES]),
        features={domain:numeric[columns].to_numpy(dtype=float) for domain,columns in DOMAINS.items()},
        input_sha256=sha256(payload).hexdigest())
