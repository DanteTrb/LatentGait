from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = Path("data/processed/assessment_level_sensitivity_dataset.csv")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

for col in [
    "target_bin",
    "latent_severity",
    "Age",
    "Duration_years",
    "UPDRS_III",
    "HY",
    "LEDD",
    "Gait_Speed",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


models = {
    "Primary: latent + age + disease duration": [
        "latent_severity",
        "Age",
        "Duration_years",
    ],
    "Sensitivity + UPDRS-III": [
        "latent_severity",
        "Age",
        "Duration_years",
        "UPDRS_III",
    ],
    "Sensitivity + H&Y": [
        "latent_severity",
        "Age",
        "Duration_years",
        "HY",
    ],
    "Sensitivity + LEDD": [
        "latent_severity",
        "Age",
        "Duration_years",
        "LEDD",
    ],
    "Sensitivity + gait speed": [
        "latent_severity",
        "Age",
        "Duration_years",
        "Gait_Speed",
    ],
    "Expanded: + UPDRS-III + LEDD + gait speed": [
        "latent_severity",
        "Age",
        "Duration_years",
        "UPDRS_III",
        "LEDD",
        "Gait_Speed",
    ],
}


rows = []

for model_name, predictors in models.items():
    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    cols = ["target_bin"] + predictors
    d = df[cols].dropna().copy()

    y = d["target_bin"].astype(int).to_numpy()
    X_df = d[predictors].copy()

    for p in predictors:
        X_df[p] = zscore(X_df[p])

    X = X_df.to_numpy()
    latent_idx = predictors.index("latent_severity")

    coords = {"predictor": predictors}

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("observation", "predictor"))
        y_data = pm.Data("y", y, dims="observation")

        intercept = pm.Normal("intercept", mu=0, sigma=1.5)
        beta = pm.Normal("beta", mu=0, sigma=1, dims="predictor")

        logit_p = intercept + pm.math.dot(X_data, beta)
        pm.Bernoulli("obs", logit_p=logit_p, observed=y_data, dims="observation")

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
            progressbar=True,
        )

    beta_latent = idata.posterior["beta"].sel(predictor="latent_severity").values.ravel()
    or_latent = np.exp(beta_latent)

    beta_hdi = az.hdi(beta_latent, hdi_prob=0.95)
    or_hdi = az.hdi(or_latent, hdi_prob=0.95)

    prob_gt_0 = float((beta_latent > 0).mean())

    # convergence diagnostics
    summary = az.summary(idata, var_names=["beta"], hdi_prob=0.95)
    latent_row_name = "beta[latent_severity]"

    if latent_row_name in summary.index:
        rhat = float(summary.loc[latent_row_name, "r_hat"])
        ess_bulk = float(summary.loc[latent_row_name, "ess_bulk"])
    else:
        rhat = np.nan
        ess_bulk = np.nan

    rows.append(
        {
            "Model": model_name,
            "N": len(d),
            "Events": int(y.sum()),
            "Beta_latent_mean": float(beta_latent.mean()),
            "Beta_latent_HDI_2.5": float(beta_hdi[0]),
            "Beta_latent_HDI_97.5": float(beta_hdi[1]),
            "OR_latent_mean": float(or_latent.mean()),
            "OR_latent_HDI_2.5": float(or_hdi[0]),
            "OR_latent_HDI_97.5": float(or_hdi[1]),
            "P_beta_latent_gt_0": prob_gt_0,
            "Rhat_beta_latent": rhat,
            "ESS_bulk_beta_latent": ess_bulk,
        }
    )

    # Save model-specific inference object
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace(":", "")
        .replace("&", "and")
    )
    idata.to_netcdf(OUT_DIR / f"{safe_name}_idata.nc")


res = pd.DataFrame(rows)
out_path = OUT_DIR / "sensitivity_logistic_adjustment_bayesian.csv"
res.to_csv(out_path, index=False)

print("\nFinal Bayesian sensitivity results:")
print(res.to_string(index=False))
print("\nSaved:", out_path)