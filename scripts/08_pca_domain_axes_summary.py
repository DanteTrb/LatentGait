from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RAW_PATH = Path("data/raw/DatasetIcotMond.xlsx")
AXES_PATH = Path("data/processed/latent_trunk_axes.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

raw = pd.read_excel(RAW_PATH)
existing_axes = pd.read_csv(AXES_PATH)

domains = {
    "Rhythmicity-recurrence": {
        "features": [
            "HR V", "HR ML", "HR AP",
            "iHR V", "iHR ML", "iHR AP",
            "%det V", "%det ML", "%det AP",
        ],
        "existing_axis": "PC1_Rhythmic",
    },
    "Neuromotor complexity": {
        "features": [
            "MSE V", "MSE ML", "MSE AP",
        ],
        "existing_axis": "PC1_Complexity",
    },
    "Lower trunk kinematics": {
        "features": [
            "Tilt", "Obliquity", "Rotation (range)",
        ],
        "existing_axis": "PC1_Postural",
    },
}

summary_rows = []
loading_rows = []

for domain_name, info in domains.items():
    features = info["features"]
    axis_col = info["existing_axis"]

    d = raw[features].copy()
    for col in features:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    valid = d.dropna().index
    X = d.loc[valid, features]

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(Xz)[:, 0]
    loadings = pca.components_[0].copy()

    # Match the orientation used in the existing processed PC1 axes
    existing = existing_axes.loc[valid, axis_col].to_numpy()
    corr = np.corrcoef(scores, existing)[0, 1]

    if corr < 0:
        scores = -scores
        loadings = -loadings
        corr = -corr

    summary_rows.append(
        {
            "Domain": domain_name,
            "N_complete": len(valid),
            "N_features": len(features),
            "PC1_explained_variance_percent": pca.explained_variance_ratio_[0] * 100,
            "Correlation_with_existing_axis_after_orientation": corr,
        }
    )

    for feature, loading in zip(features, loadings):
        loading_rows.append(
            {
                "Domain": domain_name,
                "Feature": feature,
                "PC1_loading_oriented": loading,
            }
        )

summary = pd.DataFrame(summary_rows)
loadings_df = pd.DataFrame(loading_rows)

summary_path = OUT_DIR / "pca_domain_pc1_summary.csv"
loadings_path = OUT_DIR / "pca_domain_pc1_loadings.csv"

summary.to_csv(summary_path, index=False)
loadings_df.to_csv(loadings_path, index=False)

print("\nPCA PC1 summary:")
print(summary.to_string(index=False))

print("\nPCA PC1 loadings:")
print(loadings_df.to_string(index=False))

print("\nSaved:")
print(summary_path)
print(loadings_path)