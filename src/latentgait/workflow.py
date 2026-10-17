"""Canonical entry point: all final-manuscript analyses from ONE local locked table."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .cohort import COVARIATES, link_tables, load_cohort, validate_frame, zscore
from .diagnostics import outcome_loo, paired_elpd, ppc_prevalence, sampling_diagnostics
from .functional_form import common_region
from .models import Sampling, build_measurement, build_outcome, sample, score_summary, spline_spec
from .models import prior_probability_curves
from .pca import fit_domain_pcas, indicator_matrix, validate_orientation
from .reporting import (
    coefficient_summary,
    contrasts,
    curve_region_checks,
    descriptive_tables,
    gait_speed_model,
    measurement_ppc,
    probability_table,
)

ROOT = Path(__file__).resolve().parents[2]


def external_path(path):
    p = Path(path).expanduser().resolve()
    if p == ROOT or ROOT in p.parents:
        raise ValueError("Inputs/configs/generated outputs must be outside the repository")
    return p


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def validate_protocol(cfg):
    if cfg.get("protocol_version") != 3:
        raise ValueError("Unsupported protocol version; use the version-3 example with fixed PCA orientation")
    validate_orientation(cfg.get("orientation"))
    if not cfg.get("orientation_rationale") or not cfg.get("qc_protocol_reference"):
        raise ValueError("Record orientation rationale and local QC protocol reference")
    measurement = cfg["measurement"]
    if not measurement.get("acknowledge_archived_priors"):
        raise ValueError(
            "Measurement priors/anchor constant are omitted in the paper; acknowledge their archived-code source"
        )
    if any(measurement.get(k, 0) <= 0 for k in ("anchor_sd", "loading_prior_sd", "residual_prior_sd")):
        raise ValueError("Declare positive measurement scales")
    if cfg.get("bootstrap_scaling") != "refit":
        raise ValueError("Final protocol requires refitted bootstrap standardization")
    if cfg.get("cumulative_rule") != "lower_tail_90":
        raise ValueError("Final protocol requires the inclusive lower-tail-90% cumulative interval")
    s = cfg.get("sampling", {})
    if s != {"chains": 4, "warmup": 3000, "draws": 3000, "target_accept": 0.95}:
        raise ValueError("Production sampling must be 4 x (3000 warmup + 3000 draws), target_accept .95")


def run(frame, cfg, output, *, digest, smoke=False, figures=False):
    """Internal runner. `smoke` can ONLY be selected by the synthetic-check subcommand."""
    validate_protocol(cfg)
    out = external_path(output)
    if out.exists():
        raise ValueError("Output directory must be new; existing private results are never overwritten")
    settings = (
        Sampling(draws=20, tune=20, chains=2, seed=cfg.get("seed", 42)) if smoke else Sampling(seed=cfg.get("seed", 42))
    )
    out.mkdir(parents=True)
    provenance = dict(
        status="running",
        mode="synthetic_execution_check" if smoke else "authorised_local_analysis",
        input_sha256=digest,
        config_sha256=sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        sampling=asdict(settings),
        bootstrap_resamples=8 if smoke else 500,
        bootstrap_stability_resamples=16 if smoke else 5000,
        prior_curve_draws=20 if smoke else 500,
        additional_checks="5000 bootstrap; 70/80/90% regions; 1000/2000 grids; lower-tail90 cumulative sensitivity",
        figures=figures,
        cohort_n=len(frame),
        fallers=int(frame.fall_last_year_binary.sum()),
        protocol=cfg,
        validation_target="outcome-LOO conditional on fixed full-cohort PCA and measurement posterior",
        sources={
            "manuscript_sha256": "d6458303f283233d8ef0552d911e279ac81b8dd5904a6574b9e7b4a794fde5b2",
            "supplement_sha256": "503f8490290dfc78b61c34b716ed6ed29604df76ce78a88628113d4158c8e6eb",
        },
        software={
            p: version(p)
            for p in (
                "latentgait",
                "numpy",
                "pandas",
                "scipy",
                "pymc",
                "arviz",
                "pytensor",
                "scikit-learn",
                "statsmodels",
            )
        },
    )
    # Resolve the revision only for provenance; failures do not mutate the checkout.
    try:
        provenance["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        provenance["code_working_tree_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except (OSError, subprocess.SubprocessError):
        provenance["code_revision"] = "source_archive"
    write_json(out / "provenance.json", provenance)
    try:
        report = execute(frame, cfg, out, settings, smoke, figures=figures)
    except Exception:
        provenance["status"] = "failed_incomplete"
        write_json(out / "provenance.json", provenance)
        raise
    provenance.update(
        status="synthetic_check_complete_no_inference" if smoke else "complete",
        all_sampling_adequate=all(d["adequate"] for d in report["sampling"].values()),
    )
    write_json(out / "provenance.json", provenance)
    return out


def execute(frame, cfg, out, settings, smoke, *, figures=False):
    import logging

    log = logging.getLogger("latentgait")
    diagnostics, ppc, scales, associations, stability = {}, {}, {}, [], []
    descriptors, smd = descriptive_tables(frame)
    descriptors.to_csv(out / "descriptive_table1_s1.csv", index=False)
    smd.to_csv(out / "descriptive_smd.csv", index=False)
    latent_summaries = {}
    pca = fit_domain_pcas(
        frame,
        cfg["orientation"],
        n_bootstrap=8 if smoke else 500,
        seed=settings.seed,
        bootstrap_scaling=cfg["bootstrap_scaling"],
        n_bootstrap_check=16 if smoke else 5000,
    )
    for name in ("loadings", "variance", "bootstrap", "scaling", "bootstrap_stability", "component_diagnostics"):
        getattr(pca, name).to_csv(out / f"pca_{name}.csv", index=False)
    cov, cov_center, cov_sd = zscore(frame[COVARIATES])
    pd.DataFrame(dict(variable=COVARIATES, center=cov_center, sd=cov_sd)).to_csv(
        out / "covariate_scaling.csv", index=False
    )
    y = frame.fall_last_year_binary.to_numpy(int)
    fits, means, uncertainties = {}, {}, {}
    measurement = {k: cfg["measurement"][k] for k in ("anchor_sd", "loading_prior_sd", "residual_prior_sd")}
    for strategy in ("pc1", "complexity_pc2", "all_pc2", "variance80"):
        log.warning("Running measurement/component strategy: %s", strategy)
        x, labels = indicator_matrix(pca, strategy)
        latent = sample(build_measurement(x, labels, **measurement), settings, ppc=True)
        name = "measurement_" + strategy
        diagnostics[name] = sampling_diagnostics(latent)
        ppc[name] = measurement_ppc(latent, x)
        coefficient_summary(latent, ["loading", "residual", "residual_variance"]).assign(model=name).to_csv(
            out / f"{name}_parameters.csv", index=False
        )
        mean, sd, scales[strategy] = score_summary(latent)
        means[strategy], uncertainties[strategy] = mean, sd
        latent_summaries[strategy] = {
            label: dict(
                mean=float(a.mean()),
                sd=float(a.std(ddof=1)),
                min=float(a.min()),
                max=float(a.max()),
                quantiles=np.quantile(a, [0.025, 0.25, 0.5, 0.75, 0.975]).tolist(),
            )
            for label, a in [("posterior_means", mean), ("posterior_sds", sd)]
        }
        stability.append(dict(strategy=strategy, pearson_r=float(np.corrcoef(means["pc1"], mean)[0, 1])))
        model = build_outcome("logit", mean, sd, y, cov[:, :2])
        fit = sample(model, settings, ppc=True)
        fits[strategy] = fit
        diagnostics["logit_" + strategy] = sampling_diagnostics(fit)
        ppc["logit_" + strategy] = ppc_prevalence(fit, y)
        associations.append(
            coefficient_summary(fit, ["intercept", "beta_z", "beta_cov"], link="logit").assign(
                model="logit_" + strategy
            )
        )
    mean, sd = means["pc1"], uncertainties["pc1"]
    for label, indices in [
        ("plus_updrs", [0, 1, 2]),
        ("plus_hy", [0, 1, 3]),
        ("plus_ledd", [0, 1, 4]),
        ("plus_gait_speed", [0, 1, 5]),
        ("expanded", [0, 1, 2, 4, 5]),
    ]:
        fit = sample(build_outcome("logit", mean, sd, y, cov[:, indices]), settings, ppc=True)
        diagnostics[label] = sampling_diagnostics(fit)
        ppc[label] = ppc_prevalence(fit, y)
        associations.append(
            coefficient_summary(fit, ["intercept", "beta_z", "beta_cov"], link="logit").assign(model=label)
        )
    pc2 = zscore(pca.scores["complexity"][:, 1])[0]
    incremental_cov = np.column_stack([cov[:, :2], pc2])
    incremental = sample(build_outcome("logit", mean, sd, y, incremental_cov), settings, ppc=True)
    diagnostics["incremental_complexity_pc2"] = sampling_diagnostics(incremental)
    ppc["incremental_complexity_pc2"] = ppc_prevalence(incremental, y)
    associations.append(
        coefficient_summary(incremental, ["intercept", "beta_z", "beta_cov"], link="logit").assign(
            model="incremental_complexity_pc2"
        )
    )
    # Store a term map: beta_cov is never left as an unexplained positional coefficient.
    terms = {"logit_" + s: COVARIATES[:2] for s in means}
    terms.update(
        plus_updrs=[*COVARIATES[:2], COVARIATES[2]],
        plus_hy=[*COVARIATES[:2], COVARIATES[3]],
        plus_ledd=[*COVARIATES[:2], COVARIATES[4]],
        plus_gait_speed=[*COVARIATES[:2], COVARIATES[5]],
        expanded=[COVARIATES[i] for i in (0, 1, 2, 4, 5)],
        incremental_complexity_pc2=[*COVARIATES[:2], "complexity_PC2"],
    )
    write_json(out / "coefficient_terms.json", terms)
    pd.concat(associations, ignore_index=True).to_csv(out / "logistic_associations.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "component_stability.csv", index=False)
    grid = np.linspace(mean.min(), mean.max(), 1000)
    table, _ = probability_table(fits["pc1"], grid, "logit")
    table.to_csv(out / "primary_logistic_curve.csv", index=False)
    spec = spline_spec(mean)
    no_cov = np.empty((len(y), 0))
    forms, loos, regions, prior_tables, curve_checks = {}, {}, {}, [], {}
    for form in ("fixed", "free", "spline"):
        log.warning("Running functional form: %s", form)
        model = build_outcome(form, mean, sd, y, no_cov, spline=spec)
        prior_curves = prior_probability_curves(
            model, grid, form, spline=spec, draws=20 if smoke else 500, seed=settings.seed
        )
        prior_lo, prior_hi = np.quantile(prior_curves, [0.025, 0.975], axis=0)
        prior_tables.append(
            pd.DataFrame(dict(model=form, score=grid, mean=prior_curves.mean(0), lower=prior_lo, upper=prior_hi))
        )
        trace = sample(model, settings, ppc=True)
        forms[form] = trace
        diagnostics[form] = sampling_diagnostics(trace)
        ppc[form] = ppc_prevalence(trace, y)
        names = {
            "fixed": ["tau"],
            "free": ["tau", "slope"],
            "spline": ["intercept", "beta_linear", "beta_nonlinear", "sigma_nonlinear"],
        }[form]
        coefficient_summary(trace, names, link="probit").to_csv(out / f"{form}_parameters.csv", index=False)
        table, regions[form] = probability_table(trace, grid, form, spec)
        table.to_csv(out / f"{form}_probability_curve.csv", index=False)
        curve_checks[form] = curve_region_checks(trace, grid, form, spec)
        loos[form] = outcome_loo(trace, form, mean, sd, y, no_cov, spec, settings, exact=not smoke)
    loos["primary_logit"] = outcome_loo(fits["pc1"], "logit", mean, sd, y, cov[:, :2], None, settings, exact=not smoke)
    loos["incremental_pc2"] = outcome_loo(
        incremental, "logit", mean, sd, y, incremental_cov, None, settings, exact=not smoke
    )
    pairs = []
    for a, b in [("free", "fixed"), ("spline", "fixed"), ("spline", "free"), ("incremental_pc2", "primary_logit")]:
        row = paired_elpd(loos[a], loos[b])
        row["reliable"] &= (
            all(d["adequate"] for name, d in diagnostics.items() if name.startswith("measurement_")) and not smoke
        )
        pairs.append(dict(comparison=f"{a} minus {b}", **row))
    pd.DataFrame(pairs).to_csv(out / "paired_loo_comparisons.csv", index=False)
    write_json(
        out / "loo_diagnostics.json",
        {name: {k: v for k, v in value.items() if k != "pointwise"} for name, value in loos.items()},
    )
    curves, maxima, individual_summary = contrasts(
        forms["fixed"], grid, baseline_scores=mean, cumulative_rule=cfg["cumulative_rule"]
    )
    curves.to_csv(out / "hypothetical_contrast_curves.csv", index=False)
    maxima.to_csv(out / "hypothetical_contrast_maxima.csv", index=False)
    write_json(out / "aggregate_subject_contrasts.json", individual_summary)
    speed, speed_diag = gait_speed_model(frame)
    speed.to_csv(out / "descriptive_gait_speed_ols.csv", index=False)
    write_json(out / "gait_speed_diagnostics.json", speed_diag)
    write_json(out / "spline_design.json", spec)
    write_json(out / "latent_scaling.json", scales)
    write_json(out / "latent_distribution_summaries.json", latent_summaries)
    regions["common_intersection"] = common_region([r["segments"] for r in regions.values()])
    write_json(out / "derivative_regions.json", regions)
    write_json(out / "curve_region_sensitivity.json", curve_checks)
    pd.concat(prior_tables, ignore_index=True).to_csv(out / "prior_probability_curves.csv", index=False)
    write_json(out / "posterior_predictive_checks.json", ppc)
    write_json(out / "sampling_diagnostics.json", diagnostics)
    if figures:
        from .plots import fitted_figures

        fitted_figures(
            out, grid, fits["pc1"], forms, spec, mean, sd, cumulative_rule=cfg["cumulative_rule"], synthetic=smoke
        )
    return dict(sampling=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--input", required=True)
        if name == "run":
            p.add_argument("--output", required=True)
            p.add_argument("--figures", action="store_true")
    p = sub.add_parser("synthetic-check")
    p.add_argument("--output", required=True)
    p.add_argument("--figures", action="store_true")
    p = sub.add_parser("assemble")
    p.add_argument("--clinical", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("extract-features")
    p.add_argument("--manifest", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "synthetic-check":
        from .synthetic import synthetic_frame, synthetic_protocol

        frame, cfg = synthetic_frame(), synthetic_protocol()
        run(
            validate_frame(frame, expected=None),
            cfg,
            args.output,
            digest="generated_from_scratch",
            smoke=True,
            figures=args.figures,
        )
    elif args.command == "assemble":
        keys = {"participant_id": "string", "trial_id": "string"}
        clinical = pd.read_csv(external_path(args.clinical), dtype=keys)
        features = pd.read_csv(external_path(args.features), dtype=keys)
        joined = validate_frame(link_tables(clinical, features))
        destination = external_path(args.output)
        with destination.open("x") as handle:
            joined.to_csv(handle, index=False)
    elif args.command == "extract-features":
        from .signal_io import extract_manifest

        extract_manifest(external_path(args.manifest), external_path(args.protocol), external_path(args.output))
    else:
        cfg = yaml.safe_load(external_path(args.config).read_text())
        validate_protocol(cfg)
        cohort = load_cohort(external_path(args.input))
        if args.command == "validate":
            print("PASS: complete 269-participant cohort, 105/164 outcome counts, QC and protocol resolved.")
        else:
            run(cohort.frame, cfg, args.output, digest=cohort.digest, figures=args.figures)


if __name__ == "__main__":
    main()
