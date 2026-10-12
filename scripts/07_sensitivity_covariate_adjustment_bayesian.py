#!/usr/bin/env python3
"""Bayesian covariate-adjustment sensitivity analysis.

This script reconstructs one row per ID using the aggregation rules of the
original pipeline, locks one common complete-case cohort across every model,
and propagates participant-level posterior uncertainty in latent gait
instability. A fixed-score analysis on the same cohort is retained only as a
benchmark for quantifying the impact of uncertainty propagation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


RANDOM_SEED = 42
NUMERIC_COLUMNS = [
    "target_bin",
    "latent_severity",
    "latent_severity_sd",
    "Age",
    "Duration_years",
    "UPDRS_III",
    "HY",
    "LEDD",
    "Gait_Speed",
]
LOCK_COLUMNS = NUMERIC_COLUMNS.copy()

MODEL_SPECS = {
    "Primary: latent + age + disease duration": ["Age", "Duration_years"],
    "Sensitivity + UPDRS-III": ["Age", "Duration_years", "UPDRS_III"],
    "Sensitivity + H&Y": ["Age", "Duration_years", "HY"],
    "Sensitivity + LEDD": ["Age", "Duration_years", "LEDD"],
    "Sensitivity + gait speed": ["Age", "Duration_years", "Gait_Speed"],
    "Expanded: + UPDRS-III + LEDD + gait speed": [
        "Age",
        "Duration_years",
        "UPDRS_III",
        "LEDD",
        "Gait_Speed",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="LatentGait repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT/results/07_covariate_sensitivity).",
    )
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--measurement-error-only",
        action="store_true",
        help="Skip the same-cohort fixed-score benchmark.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use short sampling settings to validate end-to-end execution.",
    )
    return parser.parse_args()


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def first_nonmissing(series: pd.Series) -> float:
    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[0]


def reconstruct_subject_table(assessment: pd.DataFrame) -> pd.DataFrame:
    """Apply the original pipeline's participant-level aggregation rules.

    Age and disease duration use the first nonmissing assessment, fall status
    uses the OR/max rule, and latent posterior summaries use the median. The
    same first-nonmissing rule is used for clinical severity and medication;
    gait speed uses the mean, matching notebook 04's subject-level contrast.
    """
    df = assessment.copy()
    df["ID"] = clean_id(df["ID"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    subject = (
        df.groupby("ID", sort=False, as_index=False)
        .agg(
            Age=("Age", first_nonmissing),
            Duration_years=("Duration_years", first_nonmissing),
            target_bin=("target_bin", "max"),
            latent_severity=("latent_severity", "median"),
            latent_severity_sd=("latent_severity_sd", "median"),
            n_assessments=("ID", "size"),
            UPDRS_III=("UPDRS_III", first_nonmissing),
            HY=("HY", first_nonmissing),
            LEDD=("LEDD", first_nonmissing),
            Gait_Speed=("Gait_Speed", "mean"),
        )
    )
    if subject["ID"].duplicated().any():
        raise ValueError("Subject reconstruction did not produce unique IDs.")
    if not subject["target_bin"].dropna().isin([0, 1]).all():
        raise ValueError("target_bin must be binary after reconstruction.")
    return subject


def standardize(series: pd.Series) -> tuple[np.ndarray, float, float]:
    values = series.to_numpy(dtype=float)
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Cannot standardize {series.name}: non-positive scale.")
    return (values - mean) / scale, mean, scale


def safe_slug(value: str) -> str:
    value = value.lower().replace("&", "and").replace("+", "plus")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def posterior_hdi(values: np.ndarray) -> tuple[float, float]:
    interval = np.asarray(az.hdi(values, hdi_prob=0.95))
    return float(interval[0]), float(interval[1])


def fit_model(
    *,
    name: str,
    covariates: list[str],
    cohort: pd.DataFrame,
    standardized: dict[str, np.ndarray],
    severity_sd_z: np.ndarray,
    approach: str,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
) -> tuple[pm.Model, az.InferenceData]:
    y = cohort["target_bin"].astype(int).to_numpy()
    x_cov = np.column_stack([standardized[c] for c in covariates])
    n = len(cohort)
    coords = {
        "observation": np.arange(n),
        "covariate": covariates,
    }

    with pm.Model(coords=coords) as model:
        y_data = pm.Data("y", y, dims="observation")
        x_data = pm.Data("X_covariate", x_cov, dims=("observation", "covariate"))
        severity_mean_data = pm.Data(
            "latent_severity_mean_z", standardized["latent_severity"], dims="observation"
        )

        intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)
        beta_latent = pm.Normal("beta_latent", mu=0.0, sigma=1.0)
        beta_covariate = pm.Normal("beta_covariate", mu=0.0, sigma=1.0, dims="covariate")

        if approach == "measurement_error":
            severity_sd_data = pm.Data(
                "latent_severity_sd_z", severity_sd_z, dims="observation"
            )
            latent_term = pm.Normal(
                "latent_severity_true_z",
                mu=severity_mean_data,
                sigma=severity_sd_data,
                dims="observation",
            )
        elif approach == "point_estimate":
            latent_term = severity_mean_data
        else:
            raise ValueError(f"Unknown approach: {approach}")

        eta = intercept + beta_latent * latent_term + pm.math.dot(x_data, beta_covariate)
        p_fall = pm.Deterministic("p_fall", pm.math.sigmoid(eta), dims="observation")
        pm.Bernoulli("fall_obs", p=p_fall, observed=y_data, dims="observation")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(cores, chains),
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=True,
        )
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["fall_obs"],
            random_seed=seed + 10_000,
            progressbar=False,
        )
        idata.extend(ppc)

    return model, idata


def summarize_fit(
    *,
    model_name: str,
    approach: str,
    covariates: list[str],
    cohort: pd.DataFrame,
    idata: az.InferenceData,
) -> tuple[dict, list[dict]]:
    beta = np.asarray(idata.posterior["beta_latent"]).ravel()
    odds_ratio = np.exp(beta)
    beta_low, beta_high = posterior_hdi(beta)
    or_low, or_high = posterior_hdi(odds_ratio)
    diag = az.summary(idata, var_names=["intercept", "beta_latent", "beta_covariate"])
    latent_diag = diag.loc["beta_latent"]
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

    y = cohort["target_bin"].astype(int).to_numpy()
    replicated = np.asarray(idata.posterior_predictive["fall_obs"])
    replicated_prevalence = replicated.mean(axis=-1).ravel()
    ppc_low, ppc_high = posterior_hdi(replicated_prevalence)
    p_draws = np.asarray(idata.posterior["p_fall"])
    brier_draws = ((p_draws - y[None, None, :]) ** 2).mean(axis=-1).ravel()
    brier_low, brier_high = posterior_hdi(brier_draws)

    summary_row = {
        "Model": model_name,
        "Approach": approach,
        "Covariates": "; ".join(covariates),
        "N": int(len(cohort)),
        "Events": int(y.sum()),
        "Observed_prevalence": float(y.mean()),
        "Beta_latent_mean": float(beta.mean()),
        "Beta_latent_HDI_2.5": beta_low,
        "Beta_latent_HDI_97.5": beta_high,
        "OR_latent_mean": float(odds_ratio.mean()),
        "OR_latent_HDI_2.5": or_low,
        "OR_latent_HDI_97.5": or_high,
        "P_beta_latent_gt_0": float((beta > 0).mean()),
        "Rhat_beta_latent": float(latent_diag["r_hat"]),
        "ESS_bulk_beta_latent": float(latent_diag["ess_bulk"]),
        "ESS_tail_beta_latent": float(latent_diag["ess_tail"]),
        "Max_Rhat_regression_parameters": float(diag["r_hat"].max()),
        "Min_ESS_bulk_regression_parameters": float(diag["ess_bulk"].min()),
        "Divergences": divergences,
        "PPC_prevalence_mean": float(replicated_prevalence.mean()),
        "PPC_prevalence_HDI_2.5": ppc_low,
        "PPC_prevalence_HDI_97.5": ppc_high,
        "Brier_score_mean": float(brier_draws.mean()),
        "Brier_score_HDI_2.5": brier_low,
        "Brier_score_HDI_97.5": brier_high,
    }

    severity = cohort["latent_severity"].to_numpy()
    quartile = pd.qcut(severity, q=4, labels=False, duplicates="drop")
    calibration_rows = []
    for group in sorted(pd.unique(quartile)):
        idx = np.flatnonzero(np.asarray(quartile) == group)
        rep_group = replicated[..., idx].mean(axis=-1).ravel()
        pred_group = p_draws[..., idx].mean(axis=-1).ravel()
        rep_low, rep_high = posterior_hdi(rep_group)
        pred_low, pred_high = posterior_hdi(pred_group)
        calibration_rows.append(
            {
                "Model": model_name,
                "Approach": approach,
                "Latent_quartile": int(group) + 1,
                "N": int(len(idx)),
                "Observed_prevalence": float(y[idx].mean()),
                "Predicted_probability_mean": float(pred_group.mean()),
                "Predicted_probability_HDI_2.5": pred_low,
                "Predicted_probability_HDI_97.5": pred_high,
                "Replicated_prevalence_mean": float(rep_group.mean()),
                "Replicated_prevalence_HDI_2.5": rep_low,
                "Replicated_prevalence_HDI_97.5": rep_high,
            }
        )
    return summary_row, calibration_rows


def write_rebuttal_summary(summary: pd.DataFrame, audit: dict, output: Path) -> None:
    reported = summary[summary["Approach"] == "measurement_error"].copy()
    lines = [
        "### Covariate-adjustment sensitivity with latent-score uncertainty",
        "",
        (
            f"All covariate-adjustment models were refitted in one locked complete-case "
            f"cohort of {audit['locked_n']} unique IDs ({audit['locked_events']} with "
            "retrospective fall occurrence). The participant-specific posterior standard "
            "deviation of latent gait instability was retained through a normal "
            "measurement-error layer in every model."
        ),
        "",
    ]
    for _, row in reported.iterrows():
        lines.append(
            f"- {row['Model']}: OR {row['OR_latent_mean']:.2f} "
            f"(95% HDI {row['OR_latent_HDI_2.5']:.2f}–{row['OR_latent_HDI_97.5']:.2f}); "
            f"P(β > 0)={row['P_beta_latent_gt_0']:.3f}."
        )
    lines.extend(
        [
            "",
            (
                "These analyses assess robustness of the latent-instability association to "
                "additional adjustment; they do not identify causal effects and should not "
                "be interpreted as prospective fall prediction."
            ),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_path = project_root / "data/processed/assessment_level_sensitivity_dataset.csv"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "results/07_covariate_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    draws, tune, chains = args.draws, args.tune, args.chains
    if args.smoke_test:
        draws, tune, chains = 100, 100, 2

    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed input: {data_path}")
    assessment = pd.read_csv(data_path)
    missing_columns = sorted(set(["ID", *NUMERIC_COLUMNS]) - set(assessment.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    subject = reconstruct_subject_table(assessment)
    missing_by_column = subject[LOCK_COLUMNS].isna().sum().astype(int).to_dict()
    exclusion_masks = {c: subject[c].isna() for c in LOCK_COLUMNS}
    complete_mask = subject[LOCK_COLUMNS].notna().all(axis=1)
    cohort = subject.loc[complete_mask].copy().reset_index(drop=True)
    if len(cohort) == 0:
        raise ValueError("The locked complete-case cohort is empty.")
    if (cohort["latent_severity_sd"] <= 0).any():
        raise ValueError("All latent_severity_sd values must be positive.")

    audit = {
        "input_file": str(data_path),
        "input_rows": int(len(assessment)),
        "input_unique_ids": int(clean_id(assessment["ID"]).nunique()),
        "subject_rows_after_aggregation": int(len(subject)),
        "subject_unique_ids_after_aggregation": int(subject["ID"].nunique()),
        "locked_n": int(len(cohort)),
        "locked_events": int(cohort["target_bin"].sum()),
        "locked_nonevents": int(len(cohort) - cohort["target_bin"].sum()),
        "locked_prevalence": float(cohort["target_bin"].mean()),
        "excluded_n": int((~complete_mask).sum()),
        "missing_by_column_before_lock": missing_by_column,
        "ids_missing_each_column": {
            c: int(mask.sum()) for c, mask in exclusion_masks.items() if mask.any()
        },
        "aggregation_rules": {
            "Age": "first nonmissing assessment",
            "Duration_years": "first nonmissing assessment",
            "target_bin": "maximum (OR rule across assessments)",
            "latent_severity": "median across assessments",
            "latent_severity_sd": "median across assessments",
            "UPDRS_III": "first nonmissing assessment",
            "HY": "first nonmissing assessment",
            "LEDD": "first nonmissing assessment",
            "Gait_Speed": "mean across assessments",
        },
        "sampling": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": args.target_accept,
            "random_seed": RANDOM_SEED,
            "smoke_test": bool(args.smoke_test),
        },
    }
    (output_dir / "cohort_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    cohort.to_csv(output_dir / "locked_subject_cohort.csv", index=False)

    standardized: dict[str, np.ndarray] = {}
    scaling_rows = []
    for column in ["latent_severity", "Age", "Duration_years", "UPDRS_III", "HY", "LEDD", "Gait_Speed"]:
        standardized[column], mean, scale = standardize(cohort[column])
        scaling_rows.append({"Variable": column, "Mean": mean, "SD_ddof0": scale})
    severity_scale = next(r["SD_ddof0"] for r in scaling_rows if r["Variable"] == "latent_severity")
    severity_sd_z = cohort["latent_severity_sd"].to_numpy(dtype=float) / severity_scale
    pd.DataFrame(scaling_rows).to_csv(output_dir / "standardization_parameters.csv", index=False)

    approaches = ["measurement_error"]
    if not args.measurement_error_only:
        approaches.append("point_estimate")

    summary_rows: list[dict] = []
    calibration_rows: list[dict] = []
    fit_counter = 0
    for approach in approaches:
        for model_name, covariates in MODEL_SPECS.items():
            fit_counter += 1
            print("\n" + "=" * 88)
            print(f"{model_name} | {approach} | N={len(cohort)} | events={int(cohort.target_bin.sum())}")
            print("=" * 88, flush=True)
            _, idata = fit_model(
                name=model_name,
                covariates=covariates,
                cohort=cohort,
                standardized=standardized,
                severity_sd_z=severity_sd_z,
                approach=approach,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=args.cores,
                target_accept=args.target_accept,
                seed=RANDOM_SEED + fit_counter * 101,
            )
            summary_row, model_calibration = summarize_fit(
                model_name=model_name,
                approach=approach,
                covariates=covariates,
                cohort=cohort,
                idata=idata,
            )
            summary_rows.append(summary_row)
            calibration_rows.extend(model_calibration)
            filename = f"{approach}__{safe_slug(model_name)}.nc"
            idata.to_netcdf(output_dir / filename)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "covariate_sensitivity_bayesian.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output_dir / "posterior_predictive_calibration_by_latent_quartile.csv", index=False
    )

    if set(summary["Approach"]) == {"measurement_error", "point_estimate"}:
        me = summary[summary["Approach"] == "measurement_error"].set_index("Model")
        pe = summary[summary["Approach"] == "point_estimate"].set_index("Model")
        comparison = pd.DataFrame(
            {
                "Model": me.index,
                "OR_measurement_error": me["OR_latent_mean"],
                "OR_point_estimate": pe["OR_latent_mean"],
                "OR_difference_ME_minus_PE": me["OR_latent_mean"] - pe["OR_latent_mean"],
                "P_positive_measurement_error": me["P_beta_latent_gt_0"],
                "P_positive_point_estimate": pe["P_beta_latent_gt_0"],
                "P_positive_difference_ME_minus_PE": (
                    me["P_beta_latent_gt_0"] - pe["P_beta_latent_gt_0"]
                ),
            }
        ).reset_index(drop=True)
        comparison.to_csv(output_dir / "measurement_error_vs_point_estimate.csv", index=False)

    write_rebuttal_summary(
        summary, audit, output_dir / "rebuttal_ready_covariate_sensitivity.md"
    )
    print("\nFinal results:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()
