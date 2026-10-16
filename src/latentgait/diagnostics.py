from __future__ import annotations
import numpy as np

def sampling_diagnostics(idata):
    import arviz as az
    s = az.summary(idata, kind="diagnostics")
    return {"max_rhat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min()), "min_ess_tail": float(s["ess_tail"].min()), "divergences": int(idata.sample_stats["diverging"].sum())}

def derivative_region(grid, probabilities, fraction=.80):
    """Grid-wise derivative and >=80%-of-maximum region; descriptive only."""
    d = np.gradient(np.asarray(probabilities), np.asarray(grid), axis=-1)
    mean_d = d.mean(axis=0) if d.ndim == 2 else d
    maximum = np.max(mean_d)
    return mean_d, np.asarray(grid)[mean_d >= fraction*maximum]
