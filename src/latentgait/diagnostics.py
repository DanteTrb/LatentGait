"""Posterior checks and paired outcome-LOO, including exact problematic-case refits."""

import numpy as np
import xarray as xr
from scipy.special import logsumexp

from .models import build_outcome, marginal_probability, sample


def sampling_diagnostics(trace):
    import arviz as az

    s = az.summary(trace, kind="diagnostics")
    rh, bulk, tail = s.r_hat.to_numpy(), s.ess_bulk.to_numpy(), s.ess_tail.to_numpy()
    divergences = int(trace.sample_stats.diverging.sum())
    return dict(
        max_rhat=float(np.nanmax(rh)),
        min_ess_bulk=float(np.nanmin(bulk)),
        min_ess_tail=float(np.nanmin(tail)),
        divergences=divergences,
        bfmi_min=float(np.min(az.bfmi(trace))),
        adequate=bool(
            np.isfinite(rh).all()
            and np.isfinite(bulk).all()
            and np.isfinite(tail).all()
            and np.max(rh) <= 1.01
            and min(np.min(bulk), np.min(tail)) >= 400
            and divergences == 0
        ),
    )


def outcome_loo(trace, form, mean, sd, y, covariates, spline, settings, *, exact=True):
    import arviz as az

    p = marginal_probability(trace, mean, sd, form, covariates, spline)
    # Verify numerical integration where no analytical probit integral is available.
    if form in ("logit", "spline"):
        p2 = marginal_probability(trace, mean, sd, form, covariates, spline, nodes=63)
        if np.max(abs(p - p2)) > 1e-5:
            raise ValueError("Score integration has not converged; increase quadrature resolution")
    observed_p = np.where(np.asarray(y)[None, :] == 1, p, 1 - p)
    log_lik = np.log(np.clip(observed_p, 1e-15, 1))
    chains, draws = trace.posterior.sizes["chain"], trace.posterior.sizes["draw"]
    marginal = az.InferenceData(
        posterior=trace.posterior,
        log_likelihood=xr.Dataset({"fall": (("chain", "draw", "subject"), log_lik.reshape(chains, draws, -1))}),
    )
    loo = az.loo(marginal, var_name="fall", pointwise=True)
    pointwise = np.array(loo.loo_i).copy()
    bad = np.flatnonzero(np.asarray(loo.pareto_k) > min(0.7, float(loo.good_k)))
    refits_ok = True
    for i in bad if exact else []:
        keep = np.arange(len(y)) != i
        m = build_outcome(form, mean[keep], sd[keep], y[keep], covariates[keep], spline=spline)
        refit = sample(m, settings)
        refits_ok &= sampling_diagnostics(refit)["adequate"]
        pred = marginal_probability(refit, mean[i : i + 1], sd[i : i + 1], form, covariates[i : i + 1], spline)
        if form in ("logit", "spline"):
            pred2 = marginal_probability(
                refit, mean[i : i + 1], sd[i : i + 1], form, covariates[i : i + 1], spline, nodes=63
            )
            if np.max(abs(pred - pred2)) > 1e-5:
                raise ValueError("Exact LOO held-out integration has not converged")
        prob = pred[:, 0] if y[i] else 1 - pred[:, 0]
        pointwise[i] = logsumexp(np.log(np.clip(prob, 1e-15, 1))) - np.log(len(prob))
    return dict(
        pointwise=pointwise,
        elpd=float(pointwise.sum()),
        se=float(np.sqrt(len(y) * np.var(pointwise, ddof=1))),
        pareto_k_max=float(np.asarray(loo.pareto_k).max()),
        flagged=int(len(bad)),
        exact_refits=int(len(bad)) if exact else 0,
        reliable=bool((exact or not len(bad)) and refits_ok and sampling_diagnostics(trace)["adequate"]),
    )


def paired_elpd(first, second):
    difference = np.asarray(first["pointwise"]) - np.asarray(second["pointwise"])
    return dict(
        delta_elpd=float(difference.sum()),
        se=float(np.sqrt(len(difference) * np.var(difference, ddof=1))),
        reliable=bool(first["reliable"] and second["reliable"]),
    )


def ppc_prevalence(trace, y):
    draws = np.asarray(trace.posterior_predictive.fall).reshape(-1, len(y)).mean(-1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(observed=float(np.mean(y)), replicated_mean=float(draws.mean()), lower=float(lo), upper=float(hi))
