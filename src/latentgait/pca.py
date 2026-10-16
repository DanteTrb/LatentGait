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
    bootstrap_stability: pd.DataFrame
    component_diagnostics: pd.DataFrame


def fit_domain_pcas(frame, directions, *, n_bootstrap=500, seed=42, bootstrap_scaling="refit", n_bootstrap_check=None):
    if bootstrap_scaling not in ("refit", "fixed") or n_bootstrap < 2:
        raise ValueError("Select refit/fixed bootstrap scaling and at least two resamples")
    rng = np.random.default_rng(seed)
    if n_bootstrap_check is not None and n_bootstrap_check < n_bootstrap:
        raise ValueError("The bootstrap stability check must include at least the primary resamples")
    total = max(n_bootstrap, n_bootstrap_check or n_bootstrap)
    # One set of participant draws shared by all domains; no outcome stratification.
    indices = rng.integers(0, len(frame), size=(total, len(frame)))
    scores, loading_rows, variance_rows, boot_rows, scaling = {}, [], [], [], []
    checks, diagnostics = [], []
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
        samples, alignments, dominance, closer_pc2 = [], [], [], []
        for ix in indices:
            zb = zscore(raw[ix])[0] if bootstrap_scaling == "refit" else z[ix]
            fitted = PCA(svd_solver="full").fit(zb)
            component = fitted.components_[0]
            # Align to full-sample loading, not an outcome or an unstable feature-wise sign.
            dot = component @ components[0]
            if abs(dot) < 1e-10:
                raise ValueError("Bootstrap PC1 cannot be sign-aligned to the reference")
            samples.append(component * np.sign(dot))
            alignments.append(abs(dot))
            dominance.append(fitted.explained_variance_[0] / fitted.explained_variance_[1])
            closer_pc2.append(abs(component @ components[1]) > abs(dot))
        samples = np.asarray(samples)
        for j, c in enumerate(columns):
            primary = samples[:n_bootstrap, j]
            lo, hi = np.quantile(primary, [0.025, 0.975])
            boot_rows.append(
                dict(domain=name, feature=c, n_resamples=n_bootstrap, mean=primary.mean(), lower=lo, upper=hi)
            )
            check_lo, check_hi = np.quantile(samples[:, j], [0.025, 0.975])
            checks.append(
                dict(
                    domain=name,
                    feature=c,
                    n_resamples=total,
                    mean=samples[:, j].mean(),
                    lower=check_lo,
                    upper=check_hi,
                    lower_change_vs_primary=check_lo - lo,
                    upper_change_vs_primary=check_hi - hi,
                )
            )
        diagnostics.append(
            dict(
                domain=name,
                eigenvalue_ratio_pc1_pc2=pca.explained_variance_[0] / pca.explained_variance_[1],
                eigenvalue_gap_pc1_pc2=pca.explained_variance_[0] - pca.explained_variance_[1],
                alignment_min=min(alignments),
                alignment_q025=np.quantile(alignments, 0.025),
                alignment_median=np.median(alignments),
                bootstrap_eigenvalue_ratio_q025=np.quantile(dominance, 0.025),
                bootstrap_eigenvalue_ratio_q975=np.quantile(dominance, 0.975),
                closer_to_reference_pc2_fraction=np.mean(closer_pc2),
                n_resamples=total,
            )
        )
    return DomainPCA(
        scores,
        pd.DataFrame(loading_rows),
        pd.DataFrame(variance_rows),
        pd.DataFrame(boot_rows),
        pd.DataFrame(scaling),
        pd.DataFrame(checks),
        pd.DataFrame(diagnostics),
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
