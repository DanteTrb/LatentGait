from __future__ import annotations
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def sampling_kwargs(seed: int = 42) -> dict:
    return dict(draws=3000, tune=3000, chains=4, target_accept=.95, random_seed=seed, return_inferencedata=True)


def latent_measurement(indicators: np.ndarray, *, anchor: int = 2, seed: int = 42):
    """Three-indicator model, standard-normal latent score, anchor loading=1."""
    X = (np.asarray(indicators) - np.mean(indicators, axis=0)) / np.std(indicators, axis=0, ddof=0)
    n, k = X.shape
    if k != 3: raise ValueError("Primary model requires the three prespecified PC1 indicators")
    with pm.Model(coords={"subject": np.arange(n), "indicator": np.arange(k)}) as m:
        z = pm.Normal("latent_instability", 0, 1, dims="subject")
        free = pm.Normal("loading_free", 0, 1, shape=k-1)
        load = pt.concatenate([free[:anchor], pt.ones(1), free[anchor:]])
        sigma_free = pm.HalfNormal("sigma_free", 1, shape=k-1)
        sigma = pt.concatenate([sigma_free[:anchor], pt.ones(1)*.001, sigma_free[anchor:]])
        pm.Normal("indicators", mu=z[:, None]*load, sigma=sigma, observed=X, dims=("subject", "indicator"))
        idata = pm.sample(**sampling_kwargs(seed))
        ppc = pm.sample_posterior_predictive(idata, random_seed=seed)
    return idata, ppc


def fixed_slope_probit(z_mean, z_sd, y, age_z, duration_z, *, seed=42):
    """Reference model propagating normal posterior-score uncertainty."""
    n = len(y)
    with pm.Model(coords={"subject": np.arange(n)}) as m:
        z = pm.Normal("z_true", mu=z_mean, sigma=np.maximum(z_sd, 1e-6), dims="subject")
        tau = pm.Normal("tau", 0, 1.5)
        b_age = pm.Normal("beta_age", 0, 1); b_duration = pm.Normal("beta_duration", 0, 1)
        p = pm.math.invprobit(z - tau + b_age*np.asarray(age_z) + b_duration*np.asarray(duration_z))
        pm.Bernoulli("fall", p=p, observed=np.asarray(y), dims="subject")
        idata = pm.sample(**sampling_kwargs(seed)); ppc = pm.sample_posterior_predictive(idata, random_seed=seed)
    return idata, ppc
