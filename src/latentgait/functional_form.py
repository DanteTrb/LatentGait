"""Outcome-model functional-form sensitivity: fixed/free probit, RCS, PSIS-LOO."""
from __future__ import annotations
import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
from patsy import dmatrix
from .models import sampling_kwargs

def rcs_basis(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear + two restricted-cubic terms, evaluated numerically for scaling/tests."""
    x=np.asarray(z); k1,k2,k3,k4=np.asarray(knots)
    plus=lambda a: np.maximum(a, 0)**3
    tail=plus(x-k1)-plus(x-k4)
    h=lambda kj: (plus(x-kj)-plus(x-k4))/(k4-k1)-((k4-kj)/(k4-k1)**2)*tail
    return np.column_stack([x, h(k2), h(k3)])

def _rcs_tensor(z, knots, center, scale):
    """Same basis evaluated at sampled latent scores, retaining measurement error."""
    k1,k2,k3,k4=np.asarray(knots)
    plus=lambda a: pt.maximum(a, 0)**3
    tail=plus(z-k1)-plus(z-k4)
    h=lambda kj: (plus(z-kj)-plus(z-k4))/(k4-k1)-((k4-kj)/(k4-k1)**2)*tail
    return (pt.stack([z, h(k2), h(k3)], axis=1)-center)/scale

def fit_probit_form(form: str, z_mean, z_sd, y, age_z, duration_z, *, seed=42):
    """Fit one prespecified form; knots derive from predictor summaries, never outcome."""
    z_mean=np.asarray(z_mean); z_sd=np.maximum(np.asarray(z_sd), 1e-6); y=np.asarray(y)
    knots=np.quantile(z_mean, [.05,.35,.65,.95])
    with pm.Model(coords={"subject": np.arange(len(y))}) as model:
        z=pm.Normal("z_true", z_mean, z_sd, dims="subject")
        intercept=pm.Normal("intercept", 0, 1.5)
        ba=pm.Normal("beta_age", 0, 1); bd=pm.Normal("beta_duration", 0, 1)
        if form == "fixed": eta=z-pm.Normal("tau", 0, 1.5)
        elif form == "free": eta=intercept + pm.Normal("beta_z", 0, 1)*z
        elif form == "spline":
            B0=rcs_basis(z_mean, knots); center=B0.mean(0); scale=B0.std(0)
            B=_rcs_tensor(z, knots, center, scale)
            beta=pm.Normal("beta_spline", 0, [1,.75,.75], shape=3)
            eta=intercept + pm.math.dot(B, beta)
        else: raise ValueError("form must be fixed, free, or spline")
        p=pm.math.invprobit(eta+ba*np.asarray(age_z)+bd*np.asarray(duration_z))
        pm.Bernoulli("fall", p=p, observed=y, dims="subject")
        idata=pm.sample(**sampling_kwargs(seed)); ppc=pm.sample_posterior_predictive(idata, random_seed=seed)
    return idata, ppc, knots

def compare_psis_loo(idatas: dict):
    """Return LOO comparison/Pareto-k; refit any observation with k > .7 exactly."""
    loos={name: az.loo(x, pointwise=True) for name,x in idatas.items()}
    return az.compare(loos, ic="loo"), {k: np.asarray(v.pareto_k) for k,v in loos.items()}
