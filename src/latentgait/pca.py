from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DomainPCA:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    variance: pd.DataFrame
    bootstrap: pd.DataFrame


def _orient(component: np.ndarray, score: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Directions are protocol-declared impairment directions (+1/-1), never outcome-derived.
    if float(np.dot(component, directions)) < 0:
        return -component, -score
    return component, score


def fit_domain_pcas(df: pd.DataFrame, domains: dict[str, dict], *, n_bootstrap: int = 500, seed: int = 42) -> DomainPCA:
    """Fit separate standardized PCAs and bootstrap oriented PC1 loadings.

    `domains[name]` needs `features` and `directions` (one +/-1 per feature).  The
    directions must be fixed before outcome modelling; this function never inspects falls.
    """
    rng = np.random.default_rng(seed)
    score_cols, loading_rows, variance_rows, boot_rows = {}, [], [], []
    for name, spec in domains.items():
        features = list(spec["features"])
        directions = np.asarray(spec["directions"], dtype=float)
        if len(features) != len(directions) or not np.isin(directions, [-1, 1]).all():
            raise ValueError(f"{name}: directions must be +/-1 and match features")
        X = df[features].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise ValueError(f"{name}: missing feature values; resolve prespecified missingness before PCA")
        Z = StandardScaler().fit_transform(X)
        pca = PCA(random_state=seed).fit(Z)
        loading, score = _orient(pca.components_[0], pca.transform(Z)[:, 0], directions)
        score_cols[f"pc1_{name}"] = score
        variance_rows.append({"domain": name, "component": 1, "explained_variance_ratio": pca.explained_variance_ratio_[0]})
        for feature, value in zip(features, loading):
            loading_rows.append({"domain": name, "component": 1, "feature": feature, "loading": value})
        for b in range(n_bootstrap):
            idx = rng.integers(0, len(Z), len(Z))
            c = PCA(random_state=seed).fit(Z[idx]).components_[0]
            c, _ = _orient(c, np.zeros(len(idx)), directions)
            for feature, value in zip(features, c):
                boot_rows.append({"domain": name, "bootstrap": b, "feature": feature, "loading": value})
    boot = pd.DataFrame(boot_rows)
    summary = boot.groupby(["domain", "feature"], as_index=False)["loading"].agg(
        bootstrap_mean="mean", ci_low=lambda x: x.quantile(.025), ci_high=lambda x: x.quantile(.975)
    )
    return DomainPCA(pd.DataFrame(score_cols), pd.DataFrame(loading_rows), pd.DataFrame(variance_rows), summary)
