"""Fall-blind, separate 3/9/3 PCAs and aligned participant-bootstrap intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .cohort import DOMAINS, zscore


@dataclass
class DomainPCA:
    scores: dict
    loadings: pd.DataFrame
    variance: pd.DataFrame
    bootstrap: pd.DataFrame
    scaling: pd.DataFrame


def fit_domain_pcas(frame, directions, *, n_bootstrap=500, seed=42, bootstrap_scaling="refit"):
    if bootstrap_scaling not in ("refit", "fixed") or n_bootstrap < 2:
        raise ValueError("Select refit/fixed bootstrap scaling and at least two resamples")
    rng = np.random.default_rng(seed)
    scores, loading_rows, variance_rows, boot_rows, scaling = {}, [], [], [], []
    for name, columns in DOMAINS.items():
        direction = np.asarray(directions[name], float)
        if direction.shape != (len(columns),) or not np.isin(direction, [-1, 1]).all():
            raise ValueError(f"A declared +/-1 direction for every {name} feature is required")
        raw = frame[columns].to_numpy(float)
        z, center, scale = zscore(raw)
        pca = PCA(svd_solver="full").fit(z)
        components = pca.components_.copy()
        # A nonzero projection on the declared biomechanical direction identifies PC1 sign.
        projection = components[0] @ direction
        if abs(projection) < 1e-8:
            raise ValueError(f"Ambiguous PC1 orientation in {name}; review the prespecified convention")
        components[0] *= np.sign(projection)
        for j in range(1, len(columns)):
            components[j] *= np.sign(components[j, np.argmax(abs(components[j]))])
        scores[name] = z @ components.T
        for c, m, s in zip(columns, center, scale):
            scaling.append(dict(feature=c, center=m, sd=s))
        for j, row in enumerate(components):
            variance_rows.append(dict(domain=name, component=j + 1, variance_ratio=pca.explained_variance_ratio_[j]))
            for c, value in zip(columns, row):
                loading_rows.append(dict(domain=name, component=j + 1, feature=c, loading=value))
        samples = []
        for _ in range(n_bootstrap):
            ix = rng.integers(0, len(z), size=len(z))
            zb = zscore(raw[ix])[0] if bootstrap_scaling == "refit" else z[ix]
            component = PCA(svd_solver="full").fit(zb).components_[0]
            # Align to full-sample loading, not an outcome or an unstable feature-wise sign.
            dot = component @ components[0]
            if abs(dot) < 1e-10:
                raise ValueError("Bootstrap PC1 cannot be sign-aligned to the reference")
            samples.append(component * np.sign(dot))
        samples = np.asarray(samples)
        for j, c in enumerate(columns):
            lo, hi = np.quantile(samples[:, j], [0.025, 0.975])
            boot_rows.append(dict(domain=name, feature=c, mean=samples[:, j].mean(), lower=lo, upper=hi))
    return DomainPCA(
        scores, pd.DataFrame(loading_rows), pd.DataFrame(variance_rows), pd.DataFrame(boot_rows), pd.DataFrame(scaling)
    )


def indicator_matrix(pca, strategy="pc1"):
    if strategy not in ("pc1", "complexity_pc2", "all_pc2", "variance80"):
        raise ValueError("Unknown component-retention strategy")
    values, labels = [], []
    for name, x in pca.scores.items():
        k = 1
        if strategy == "complexity_pc2" and name == "complexity":
            k = 2
        elif strategy == "all_pc2":
            k = 2
        elif strategy == "variance80":
            ratio = pca.variance.loc[pca.variance.domain.eq(name), "variance_ratio"].to_numpy()
            k = min(x.shape[1], np.searchsorted(np.cumsum(ratio), 0.8) + 1)
        values.append(x[:, :k])
        labels.extend(f"{name}_PC{j + 1}" for j in range(k))
    matrix = zscore(np.column_stack(values))[0]
    if labels[0] != "kinematics_PC1":
        raise ValueError("Kinematics PC1 must be the fixed-loading anchor")
    return matrix, labels
