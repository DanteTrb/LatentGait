"""Aggregate summaries only. All writes are restricted to an external private directory."""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .models import posterior_values, probability_draws
from .functional_form import derivative_region


def coefficient_summary(trace, names, *, link=None):
    import arviz as az

    if link not in (None, "logit", "probit"):
        raise ValueError("Declare logit/probit link, or None for measurement parameters")
    rows = []
    for name in names:
        values = posterior_values(trace, name)
        for index in np.ndindex(values.shape[1:]):
            v = values[(slice(None), *index)]
            lo, hi = az.hdi(v, hdi_prob=0.95)
            row = dict(
                parameter=name,
                term=str(index),
                mean=v.mean(),
                hdi_lower=lo,
                hdi_upper=hi,
                probability_positive=float(np.mean(v > 0)),
                parameter_scale="log_odds" if link == "logit" else "probit" if link == "probit" else "measurement",
            )
            if link == "logit" and name.startswith("beta"):
                odds = np.exp(v)
                lo, hi = az.hdi(odds, hdi_prob=0.95)
                row.update(or_mean=odds.mean(), or_hdi_lower=lo, or_hdi_upper=hi)
            rows.append(row)
    return pd.DataFrame(rows)


def probability_table(trace, grid, form, spline=None):
    draws = probability_draws(trace, grid, form, spline=spline)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    table = pd.DataFrame(dict(score=grid, mean=draws.mean(0), lower=lo, upper=hi))
    region = derivative_region(grid, draws)
    table["derivative_of_mean"] = region.pop("derivative")
    return table, region


def curve_region_checks(trace, grid, form, spline=None):
    """Predeclared numerical/descriptive checks; no model/region selection by results."""
    rows = []
    for points in (len(grid), 2000):
        x = np.linspace(grid[0], grid[-1], points)
        draws = probability_draws(trace, x, form, spline=spline)
        for fraction in (0.7, 0.8, 0.9):
            region = derivative_region(x, draws, fraction=fraction)
            region.pop("derivative")
            rows.append(dict(grid_points=points, derivative_fraction=fraction, **region))
    primary = next(r for r in rows if r["grid_points"] == len(grid) and r["derivative_fraction"] == 0.8)
    refined = next(r for r in rows if r["grid_points"] == 2000 and r["derivative_fraction"] == 0.8)
    a, b = primary["peak_location"], refined["peak_location"]
    return dict(
        checks=rows,
        primary_grid_points=len(grid),
        refined_grid_points=2000,
        primary_fraction=0.8,
        peak_location_change=None if a is None or b is None else abs(a - b),
    )


def cumulative_contrast_interval(scores, weights, *, rule="lower_tail_90"):
    """Empirical weighted quantiles, combining tied baseline scores; no KDE."""
    scores, weights = np.asarray(scores), np.asarray(weights)
    if (
        scores.ndim != 1
        or scores.shape != weights.shape
        or not len(scores)
        or not np.isfinite(scores).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Finite baseline scores and nonnegative matching weights are required")
    if rule not in ("central_90", "lower_tail_90"):
        raise ValueError("Unknown cumulative convention")
    if weights.sum() <= 1e-15:
        return dict(interval=None, included_weight_fraction=None)
    order = np.argsort(scores, kind="stable")
    support, starts = np.unique(scores[order], return_index=True)
    grouped = np.add.reduceat(weights[order], starts)
    cumulative = np.cumsum(grouped) / grouped.sum()
    low = 0 if rule == "lower_tail_90" else int(np.searchsorted(cumulative, 0.05))
    high = int(np.searchsorted(cumulative, 0.90 if rule == "lower_tail_90" else 0.95))
    high = min(high, len(support) - 1)
    return dict(
        interval=[float(support[low]), float(support[high])],
        included_weight_fraction=float(grouped[low : high + 1].sum() / grouped.sum()),
    )


def contrasts(trace, grid, *, baseline_scores, cumulative_rule):
    """S6 grid contrasts and private aggregate summaries of subject-specific draws."""
    if cumulative_rule not in ("lower_tail_90", "central_90"):
        raise ValueError("Declare lower_tail_90 or central_90 cumulative contrast convention")
    tau = posterior_values(trace, "tau")[:, None]
    before = ndtr(grid[None, :] - tau)
    curves, maxima = [], []
    for delta in (0.1, 0.2, 0.3, 0.5):
        change = ndtr(grid[None, :] - delta - tau) - before
        mean = change.mean(0)
        lo, hi = np.quantile(change, [0.025, 0.975], axis=0)
        j = int(np.argmin(mean))
        curves.append(
            pd.DataFrame(
                dict(
                    shift=delta,
                    score=grid,
                    mean=mean,
                    lower=lo,
                    upper=hi,
                    shifted_outside_observed_support=grid - delta < grid[0],
                )
            )
        )
        maxima.append(
            dict(shift=delta, location=float(grid[j]), mean=float(mean[j]), lower=float(lo[j]), upper=float(hi[j]))
        )
    z = posterior_values(trace, "z_true")
    individual = ndtr(z - 0.3 - tau) - ndtr(z - tau)
    means = individual.mean(0)
    aggregate = individual.mean(1)
    scores = np.asarray(baseline_scores)
    if scores.shape != means.shape or not np.isfinite(scores).all():
        raise ValueError("Baseline score order must use the same locked upstream participants")
    cumulative = cumulative_contrast_interval(scores, abs(means), rule=cumulative_rule)
    alternative = cumulative_contrast_interval(scores, abs(means), rule="central_90")
    # No individual scores/contrasts, IDs or sorted ranks are exported.
    summary = dict(
        shift=0.3,
        mean_of_individual_posterior_means=float(means.mean()),
        between_subject_sd_of_posterior_means=float(means.std(ddof=1)),
        min_individual_posterior_mean=float(means.min()),
        max_individual_posterior_mean=float(means.max()),
        cohort_mean_cri=np.quantile(aggregate, [0.025, 0.975]).tolist(),
        cumulative_interval=cumulative["interval"],
        cumulative_interval_weight_fraction=cumulative["included_weight_fraction"],
        cumulative_rule=cumulative_rule,
        central_90_alternative=alternative,
    )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(maxima), summary


def gait_speed_model(frame):
    """S8: OLS log(speed) on intercept + mean-centered age and disease duration."""
    import statsmodels.api as sm

    x = frame[["age_years", "disease_duration_years"]].to_numpy(float)
    x = sm.add_constant(x - x.mean(0))
    y = np.log(frame.gait_speed_m_s.to_numpy(float))
    fit = sm.OLS(y, x).fit()
    ci = fit.conf_int()
    result = pd.DataFrame(
        dict(
            term=["intercept", "age_years_centered", "disease_duration_years_centered"],
            coefficient=fit.params,
            se=fit.bse,
            lower=ci[:, 0],
            upper=ci[:, 1],
            p=fit.pvalues,
        )
    )
    influence = fit.get_influence()
    from statsmodels.stats.stattools import jarque_bera, durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan

    jb, jb_p, skew, kurtosis = jarque_bera(fit.resid)
    bp, bp_p, _, _ = het_breuschpagan(fit.resid, x)
    diagnostics = dict(
        r_squared=float(fit.rsquared),
        adjusted_r_squared=float(fit.rsquared_adj),
        condition_number=float(fit.condition_number),
        max_cooks_distance=float(influence.cooks_distance[0].max()),
        jarque_bera=float(jb),
        jarque_bera_p=float(jb_p),
        residual_skew=float(skew),
        residual_kurtosis=float(kurtosis),
        breusch_pagan=float(bp),
        breusch_pagan_p=float(bp_p),
        durbin_watson=float(durbin_watson(fit.resid)),
    )
    return result, diagnostics


def measurement_ppc(trace, observed):
    replicated = np.asarray(trace.posterior_predictive.indicators).reshape(-1, *observed.shape)
    rows = []
    for j in range(observed.shape[1]):
        means, sds = replicated[:, :, j].mean(1), replicated[:, :, j].std(1)
        rows.append(
            dict(
                indicator=j,
                observed_mean=observed[:, j].mean(),
                observed_sd=observed[:, j].std(),
                replicated_mean=float(means.mean()),
                replicated_mean_cri=np.quantile(means, [0.025, 0.975]).tolist(),
                replicated_sd=float(sds.mean()),
                replicated_sd_cri=np.quantile(sds, [0.025, 0.975]).tolist(),
            )
        )
    return rows


def descriptive_tables(frame):
    """Main Table 1 / S1 summaries, generated from input; no group hypothesis tests."""
    from .cohort import COVARIATES, DESCRIPTIVES, FEATURES

    rows = []
    binary = {
        "sex_male": frame.sex_male,
        "hoehn_yahr_ge3": (frame.hoehn_yahr >= 3).astype(int),
        "fall_last_year_binary": frame.fall_last_year_binary,
    }
    for group, data in [
        ("all", frame),
        ("nonfallers", frame.loc[frame.fall_last_year_binary.eq(0)]),
        ("fallers", frame.loc[frame.fall_last_year_binary.eq(1)]),
    ]:
        for c in [*COVARIATES, *FEATURES, *[c for c in DESCRIPTIVES if c != "sex_male"]]:
            values = data[c]
            rows.append(
                dict(
                    group=group,
                    variable=c,
                    n=len(values),
                    mean=values.mean(),
                    sd=values.std(ddof=1),
                    median=values.median(),
                    q25=values.quantile(0.25),
                    q75=values.quantile(0.75),
                )
            )
        for c, values in binary.items():
            v = values.loc[data.index]
            rows.append(dict(group=group, variable=c, n=len(v), count=int(v.sum()), percent=float(100 * v.mean())))
    smds = []
    for c in [*COVARIATES, "onset_age_years", "cadence_steps_min", "stride_length_m"]:
        non = frame.loc[frame.fall_last_year_binary.eq(0), c]
        fall = frame.loc[frame.fall_last_year_binary.eq(1), c]
        pooled = np.sqrt((non.var(ddof=1) + fall.var(ddof=1)) / 2)
        smds.append(
            dict(variable=c, standardized_mean_difference=(fall.mean() - non.mean()) / pooled if pooled else np.nan)
        )
    p0 = frame.loc[frame.fall_last_year_binary.eq(0), "sex_male"].mean()
    p1 = frame.loc[frame.fall_last_year_binary.eq(1), "sex_male"].mean()
    denominator = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
    smds.append(
        dict(variable="sex_male", standardized_mean_difference=(p1 - p0) / denominator if denominator else np.nan)
    )
    return pd.DataFrame(rows), pd.DataFrame(smds)
