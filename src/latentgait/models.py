"""Models specified by the final manuscript; all outcomes share one locked cohort."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr
from numpy.polynomial.hermite import hermgauss

from .cohort import zscore
from .functional_form import rcs_basis, rcs_tensor


@dataclass(frozen=True)
class Sampling:
    draws: int = 3000
    tune: int = 3000
    chains: int = 4
    target_accept: float = 0.95
    seed: int = 42
    cores: int = 1

    def kwargs(self):
        return dict(
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            target_accept=self.target_accept,
            random_seed=self.seed,
            cores=self.cores,
            return_inferencedata=True,
            progressbar=False,
        )


def build_measurement(x, labels, *, anchor_sd, loading_prior_sd, residual_prior_sd):
    import pymc as pm
    import pytensor.tensor as pt

    if labels[0] != "kinematics_PC1" or min(anchor_sd, loading_prior_sd, residual_prior_sd) <= 0:
        raise ValueError("Invalid anchor/prior specification")
    with pm.Model(coords={"subject": np.arange(len(x)), "indicator": labels, "free_indicator": labels[1:]}) as model:
        latent = pm.Normal("latent", 0, 1, dims="subject")
        loading = pm.Normal("loading", 0, loading_prior_sd, dims="free_indicator")
        residual = pm.HalfNormal("residual", residual_prior_sd, dims="free_indicator")
        pm.Deterministic("residual_variance", residual**2, dims="free_indicator")
        load = pt.concatenate([pt.ones(1), loading])
        sigma = pt.concatenate([pt.ones(1) * anchor_sd, residual])
        pm.Normal("indicators", latent[:, None] * load, sigma, observed=x, dims=("subject", "indicator"))
    return model


def sample(model, settings, *, ppc=False):
    import pymc as pm

    with model:
        fit = pm.sample(**settings.kwargs(), idata_kwargs={"log_likelihood": True})
        if ppc:
            pm.sample_posterior_predictive(fit, random_seed=settings.seed, extend_inferencedata=True, progressbar=False)
    return fit


def score_summary(trace):
    draws = np.asarray(trace.posterior.latent).reshape(-1, trace.posterior.sizes["subject"])
    mean, sd = draws.mean(0), draws.std(0, ddof=0)
    standardized, loc, scale = zscore(mean)
    # BOTH the mean and posterior SD must be transformed by the same score scale.
    if np.any(sd <= 0):
        raise ValueError("Degenerate latent posterior SD")
    return standardized, sd / scale, dict(center=float(loc), sd=float(scale))


def spline_spec(mean):
    knots = np.quantile(mean, [0.05, 0.35, 0.65, 0.95])
    basis = rcs_basis(mean, knots)
    _, center, scale = zscore(basis)
    return dict(knots=knots, center=center, scale=scale)


def build_outcome(form, mean, sd, y, covariates, *, spline=None):
    """Normal score-error model. Probit models use no covariates (paper equations)."""
    import pymc as pm
    import pytensor.tensor as pt

    mean, sd, y, covariates = map(np.asarray, (mean, sd, y, covariates))
    if form not in ("logit", "fixed", "free", "spline"):
        raise ValueError("Unknown outcome model")
    if np.any(sd <= 0) or not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise ValueError("Invalid latent posterior summaries")
    if form != "logit" and covariates.shape[1]:
        raise ValueError("Paper probit specifications are unadjusted; do not silently add covariates")
    with pm.Model(coords={"subject": np.arange(len(y)), "covariate": np.arange(covariates.shape[1])}) as model:
        epsilon = pm.Normal("score_error", 0, 1, dims="subject")
        z = pm.Deterministic("z_true", mean + sd * epsilon, dims="subject")
        if form == "fixed":
            eta = z - pm.Normal("tau", 0, 1)
        elif form == "free":
            tau = pm.Normal("tau", 0, 1.5)
            beta = pm.Deterministic("slope", pt.exp(pm.Normal("log_slope", 0, 0.5)))
            eta = beta * (z - tau)
        else:
            intercept = pm.Normal("intercept", 0, 1.5)
            if form == "logit":
                beta = pm.Normal("beta_z", 0, 1)
                eta = intercept + beta * z
            else:
                if spline is None:
                    raise ValueError("Freeze spline knots/scaling from predictor summaries before fitting")
                b = rcs_tensor(z, **spline)
                linear = pm.Normal("beta_linear", 0, 1)
                sigma = pm.HalfNormal("sigma_nonlinear", 0.5)
                nonlinear = pm.Normal("beta_nonlinear", 0, sigma, shape=2)
                eta = intercept + linear * b[:, 0] + pt.dot(b[:, 1:], nonlinear)
            if covariates.shape[1]:
                coefficients = pm.Normal("beta_cov", 0, 1, dims="covariate")
                eta = eta + pt.dot(covariates, coefficients)
        probability = pm.math.sigmoid(eta) if form == "logit" else pm.math.invprobit(eta)
        p = pm.Deterministic("p_fall", probability, dims="subject")
        pm.Bernoulli("fall", p=p, observed=y, dims="subject")
    return model


def posterior_values(trace, name, *, group="posterior"):
    a = getattr(trace, group)[name]
    return np.asarray(a.stack(sample=("chain", "draw")).transpose("sample", ...))


def probability_draws(trace, score, form, covariates=None, spline=None, *, group="posterior"):
    """Score is one fixed vector, or a (posterior samples, subjects) matrix."""
    score = np.asarray(score)
    values = lambda name: posterior_values(trace, name, group=group)
    if form == "fixed":
        eta = score - values("tau")[:, None]
    elif form == "free":
        eta = values("slope")[:, None] * (score - values("tau")[:, None])
    else:
        eta = values("intercept")[:, None]
        if form == "logit":
            eta = eta + values("beta_z")[:, None] * score
        else:
            b = (rcs_basis(score, spline["knots"]) - spline["center"]) / spline["scale"]
            eta = eta + values("beta_linear")[:, None] * b[..., 0]
            eta = eta + np.sum(values("beta_nonlinear")[:, None, :] * b[..., 1:], axis=-1)
        if covariates is not None and np.asarray(covariates).shape[1]:
            eta = eta + values("beta_cov") @ np.asarray(covariates).T
    return expit(eta) if form == "logit" else ndtr(eta)


def prior_probability_curves(model, grid, form, *, spline=None, draws=500, seed=42):
    """Prior-only diagnostic at fixed grid scores; no outcome-derived prior tuning."""
    import pymc as pm

    names = {
        "fixed": ["tau"],
        "free": ["tau", "slope"],
        "logit": ["intercept", "beta_z"],
        "spline": ["intercept", "beta_linear", "beta_nonlinear"],
    }[form]
    with model:
        prior = pm.sample_prior_predictive(draws=draws, var_names=names, random_seed=seed)
    return probability_draws(prior, grid, form, spline=spline, group="prior")


def marginal_probability(trace, mean, sd, form, covariates=None, spline=None, nodes=31):
    """Integrate held-out score uncertainty for a conditional OUTCOME-LOO target.

    The augmented regression posterior can update each nuisance score; for LOO we
    integrate against its upstream N(mean, sd) distribution, avoiding optimistic
    conditional log likelihood evaluated at an outcome-updated individual score.
    """
    mean, sd = np.asarray(mean), np.asarray(sd)
    if form in ("fixed", "free"):
        tau = posterior_values(trace, "tau")[:, None]
        slope = 1 if form == "fixed" else posterior_values(trace, "slope")[:, None]
        return ndtr(slope * (mean - tau) / np.sqrt(1 + (slope * sd) ** 2))
    x, weights = hermgauss(nodes)
    p = None
    for node, weight in zip(x, weights / np.sqrt(np.pi)):
        draw = probability_draws(trace, mean + np.sqrt(2) * sd * node, form, covariates, spline)
        p = weight * draw if p is None else p + weight * draw
    return p
