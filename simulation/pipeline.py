"""Synthetic-only experiments. No clinical files are read; no results are forced.

Run from the repository root: python -m simulation.pipeline --scenario gradual
Outputs go outside the repository by default. --smoke checks plumbing only.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import ndtr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DOMAINS = {'kinematics': 3, 'rhythmicity': 9, 'complexity': 3}
SCENARIOS = {'null': 0.0, 'gradual': 0.25, 'transition': 0.90}
DISCLAIMER = 'SYNTHETIC ONLY - NOT CLINICAL RESULTS - NO TREATMENT EFFECTS'


def generate_data(n=269, seed=20261012, scenario='gradual'):
    """Fixed generating mechanism; no seed search or outcome-based selection."""
    if n < 30 or scenario not in SCENARIOS:
        raise ValueError('Use n >= 30 and a documented scenario.')
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    age = rng.normal(size=n)
    duration = .2 * age + np.sqrt(.96) * rng.normal(size=n)
    features = {}
    for (domain, count), loading in zip(DOMAINS.items(), [.8, .6, .35]):
        axis = loading * truth + np.sqrt(1-loading**2) * rng.normal(size=n)
        local = rng.normal(size=n)
        features[domain] = np.column_stack([
            .8 * axis + .3 * (-1)**j * local + .5 * rng.normal(size=n)
            for j in range(count)
        ])
    clinical = np.column_stack([age, duration,
        .45*truth+rng.normal(size=n), .25*truth+rng.normal(size=n),
        -.35*truth+rng.normal(size=n)])
    clinical = StandardScaler().fit_transform(clinical)
    probability = ndtr(-.3 + SCENARIOS[scenario]*truth + .2*age + .1*duration)
    y = rng.binomial(1, probability)
    return dict(ids=np.array([f'SYNTHETIC_{i:04d}' for i in range(n)]),
                truth=truth, features=features, covariates=clinical, y=y,
                generating_probability=probability)


def domain_pca(features):
    """Standardize then fit separate PCAs; orientation uses features, never y."""
    scores, rows = {}, []
    for domain, x in features.items():
        xz = StandardScaler().fit_transform(x)
        pca = PCA().fit(xz)
        z = pca.transform(xz)
        for j in range(z.shape[1]):
            # Deterministic sign: largest absolute feature loading is positive.
            idx = np.argmax(np.abs(pca.components_[j]))
            sign = 1 if pca.components_[j, idx] >= 0 else -1
            z[:, j] *= sign
            rows.append(dict(domain=domain, component=j+1,
                             variance_ratio=pca.explained_variance_ratio_[j]))
        scores[domain] = StandardScaler().fit_transform(z)
    return scores, pd.DataFrame(rows)


def indicators(scores, variance, strategy='pc1'):
    result, labels = [], []
    for domain, z in scores.items():
        k = 1
        if strategy == 'complexity_pc2' and domain == 'complexity':
            k = 2
        elif strategy == 'all_pc2':
            k = 2
        elif strategy == 'variance80':
            v = variance.loc[variance.domain == domain, 'variance_ratio'].to_numpy()
            k = min(len(v), np.searchsorted(np.cumsum(v), .8)+1)
        elif strategy not in ('pc1', 'complexity_pc2', 'all_pc2', 'variance80'):
            raise ValueError(strategy)
        result.append(z[:, :k])
        labels.extend(f'{domain}_PC{j+1}' for j in range(k))
    return np.column_stack(result), labels


def sample_options(smoke, seed):
    return dict(draws=60 if smoke else 1500, tune=60 if smoke else 1500,
                chains=2 if smoke else 4, cores=1, target_accept=.95,
                random_seed=seed, progressbar=False)


def fit_measurement(x, tight, options):
    """Fall-blind one-factor model. Both fits use loading[0]=1, s~N(0,1).

    Tight anchor fixes its residual SD=.1; relaxed estimates all residual SDs.
    Small posterior SD under the tight constraint is not empirical precision.
    """
    import pymc as pm
    import pytensor.tensor as pt
    with pm.Model() as model:
        s = pm.Normal('latent', 0, 1, shape=x.shape[0])
        loading = pt.concatenate([pt.ones(1), pm.Normal('other_loadings', 0, 1, shape=x.shape[1]-1)])
        if tight:
            sigma = pt.concatenate([pt.ones(1)*.1, pm.HalfNormal('residual_sd', 1, shape=x.shape[1]-1)])
        else:
            sigma = pm.HalfNormal('residual_sd', 1, shape=x.shape[1])
        pm.Normal('indicators', s[:, None]*loading, sigma, observed=x)
        trace = pm.sample(**options)
    return trace


def score_nodes(trace, q=31):
    """Empirical posterior quantile integration, not plug-in posterior means.

    Marginal per-person uncertainty is propagated with a modular cut: outcomes
    do not update the measurement model. Cross-person posterior dependencies
    and uncertainty in estimated standardization constants are not propagated.
    """
    values = trace.posterior['latent'].values.reshape(-1, trace.posterior['latent'].shape[-1])
    means = values.mean(axis=0)
    loc, scale = means.mean(), means.std(ddof=0)
    nodes = np.quantile((values-loc)/scale, (np.arange(q)+.5)/q, axis=0).T
    return nodes, (means-loc)/scale


def curvature_basis(x):
    """Natural cubic spline: linear term plus two fixed nonlinear contrasts.

    Knots (-4,-1,1,4) are fixed before outcomes; linear tails outside boundaries.
    Contrasts are orthogonal to 1 and x on a fixed reference grid, not on y.
    """
    knots = np.array([-4., -1., 1., 4.])
    spline = CubicSpline(knots, np.eye(4), bc_type='natural', axis=0)
    ref = np.linspace(-4, 4, 401)
    linear = np.column_stack([np.ones(len(ref)), ref])
    projection = np.linalg.lstsq(linear, spline(ref), rcond=None)[0]
    residual = spline(ref)-linear@projection
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    rotation = vt[:2].T
    x = np.asarray(x)
    clipped = np.clip(x, -4, 4)
    raw = spline(clipped)+(x-clipped)[..., None]*spline(clipped, 1)
    return (raw-np.stack([np.ones_like(x), x], axis=-1)@projection)@rotation


def fit_outcome(nodes, covariates, y, form, options):
    """Bernoulli likelihood integrates over the fixed measurement posterior.

    LOO therefore leaves out each outcome, not a latent variable inferred from
    that outcome. This is conditional outcome LOO, NOT whole-pipeline validation.
    """
    import pymc as pm
    import pytensor.tensor as pt
    if form not in ('fixed', 'free', 'spline', 'logit'):
        raise ValueError(form)
    with pm.Model() as model:
        alpha = pm.Normal('alpha', 0, 1.5)
        beta = 1. if form == 'fixed' else pm.Normal('beta', 0, 1)
        gamma = pm.Normal('gamma', 0, 1, shape=covariates.shape[1])
        eta = alpha + beta*nodes + (covariates@gamma)[:, None]
        if form == 'spline':
            curvature = pm.Normal('curvature', 0, .5, shape=2)
            eta = eta + pt.sum(curvature_basis(nodes)*curvature, axis=-1)
        p = pm.math.sigmoid(eta) if form == 'logit' else .5*(1+pt.erf(eta/np.sqrt(2)))
        pm.Bernoulli('fall', p=pt.clip(p.mean(axis=1), 1e-9, 1-1e-9), observed=y)
        trace = pm.sample(**options, idata_kwargs={'log_likelihood': True})
    return trace


def probability_draws(trace, score, form, covariates=None):
    """Curves conditional on score; covariates default to centered means."""
    post = trace.posterior
    alpha = post.alpha.values.reshape(-1)
    beta = np.ones_like(alpha) if form == 'fixed' else post.beta.values.reshape(-1)
    score = np.asarray(score)
    eta = alpha[:, None]+beta[:, None]*score[None, :]
    if covariates is not None:
        gamma = post.gamma.values.reshape(len(alpha), -1)
        eta += gamma@np.asarray(covariates).T
    if form == 'spline':
        eta += post.curvature.values.reshape(len(alpha), 2)@curvature_basis(score).T
    if form == 'logit':
        from scipy.special import expit
        return expit(eta)
    return ndtr(eta)


def region_segments(grid, derivative, fraction=.8):
    """All connected intervals above a fraction of the positive maximum.

    Never label a decreasing/flat curve as a positive transition. Boundary
    contact and disconnected sets are explicit, not silently discarded.
    """
    grid, derivative = np.asarray(grid), np.asarray(derivative)
    if not 0 < fraction < 1 or len(grid) != len(derivative):
        raise ValueError('Invalid region arguments')
    peak = float(derivative.max())
    if peak <= 1e-10:
        return [], None, False
    indices = np.flatnonzero(derivative >= fraction*peak)
    groups = np.split(indices, np.where(np.diff(indices)>1)[0]+1)
    segments = [[float(grid[g[0]]), float(grid[g[-1]])] for g in groups if len(g)]
    return segments, float(grid[np.argmax(derivative)]), bool(indices[0]==0 or indices[-1]==len(grid)-1)


def diagnostic_row(trace, label):
    import arviz as az
    summary = az.summary(trace, kind='diagnostics')
    return dict(model=label, max_rhat=float(summary.r_hat.max()),
                min_ess_bulk=float(summary.ess_bulk.min()),
                divergences=int(trace.sample_stats.diverging.sum()))


def run(scenario='gradual', n=269, seed=20261012, smoke=False, output=None):
    """Run one declared scenario once; return summaries without selection."""
    import arviz as az
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    output = Path(output or tempfile.mkdtemp(prefix='LatentGait_SYNTHETIC_')).resolve()
    if output == root or root in output.parents:
        raise ValueError('Generated outputs must be outside the repository.')
    output.mkdir(parents=True, exist_ok=True)
    data = generate_data(n, seed, scenario)
    scores, variance = domain_pca(data['features'])
    variance.to_csv(output/'pca_variance.csv', index=False)
    options = sample_options(smoke, seed)
    x, labels = indicators(scores, variance)
    tight = fit_measurement(x, True, options)
    relaxed = fit_measurement(x, False, options)
    diagnostics = [diagnostic_row(tight, 'measurement_tight'), diagnostic_row(relaxed, 'measurement_relaxed')]
    nodes, means = score_nodes(relaxed)
    _, tight_means = score_nodes(tight)
    construct = [dict(strategy='tight_anchor', correlation=float(np.corrcoef(means,tight_means)[0,1])),
                 dict(strategy='kinematics_pc1',correlation=float(np.corrcoef(means,scores['kinematics'][:,0])[0,1]))]
    latent=relaxed.posterior.latent.values.reshape(-1,n)
    scale=latent.mean(axis=0).std(ddof=0)
    pd.DataFrame(dict(synthetic_id=data['ids'],posterior_mean=means,
                     posterior_sd=latent.std(axis=0)/scale)).to_csv(output/'latent_scores.csv',index=False)
    loadings=relaxed.posterior.other_loadings.values.reshape(-1,len(labels)-1)
    loading_rows=[dict(indicator=labels[0],mean=1.,lower=1.,upper=1.,fixed=True)]
    for j,label in enumerate(labels[1:]):
        loading_rows.append(dict(indicator=label,mean=float(loadings[:,j].mean()),
            lower=float(np.quantile(loadings[:,j],.025)),upper=float(np.quantile(loadings[:,j],.975)),fixed=False))
    pd.DataFrame(loading_rows).to_csv(output/'measurement_loadings.csv',index=False)
    fits = {}
    for form in ('fixed', 'free', 'spline'):
        fits[form] = fit_outcome(nodes, data['covariates'][:, :2], data['y'], form, options)
        diagnostics.append(diagnostic_row(fits[form], form))
    # Covariates and PCA strategies are exploratory, never chosen by outcome fit.
    logit = fit_outcome(nodes, data['covariates'][:, :2], data['y'], 'logit', options)
    expanded = fit_outcome(nodes, data['covariates'], data['y'], 'logit', options)
    associations = []
    for name, trace in [('primary_logit',logit), ('expanded_logit',expanded)]:
        beta = trace.posterior.beta.values.ravel()
        associations.append(dict(model=name, mean_or=float(np.exp(beta).mean()),
            lower=float(np.quantile(np.exp(beta),.025)), upper=float(np.quantile(np.exp(beta),.975)),
            probability_positive=float((beta>0).mean())))
        diagnostics.append(diagnostic_row(trace,name))
    for strategy in ('complexity_pc2','all_pc2','variance80'):
        extended, _ = indicators(scores, variance, strategy)
        measure = fit_measurement(extended, False, options)
        other_nodes, other_means = score_nodes(measure)
        association = fit_outcome(other_nodes, data['covariates'][:, :2], data['y'], 'logit', options)
        beta = association.posterior.beta.values.ravel()
        construct.append(dict(strategy=strategy, correlation=float(np.corrcoef(means,other_means)[0,1])))
        associations.append(dict(model=strategy, mean_or=float(np.exp(beta).mean()),
            lower=float(np.quantile(np.exp(beta),.025)), upper=float(np.quantile(np.exp(beta),.975)),
            probability_positive=float((beta>0).mean())))
        diagnostics += [diagnostic_row(measure,strategy+'_measurement'), diagnostic_row(association,strategy+'_logit')]
    loo = {name: az.loo(trace, pointwise=True) for name,trace in fits.items()}
    comparisons=[]
    diag_by_name={d['model']:d for d in diagnostics}
    def sampling_ok(name):
        d=diag_by_name[name]
        return bool(not smoke and d['max_rhat']<=1.01 and d['min_ess_bulk']>=400 and d['divergences']==0)
    for a,b in [('free','fixed'),('spline','fixed'),('spline','free')]:
        difference=(loo[a].loo_i-loo[b].loo_i).values
        comparisons.append(dict(comparison=a+' minus '+b, delta_elpd=float(difference.sum()),
             se=float(np.sqrt(n*np.var(difference,ddof=1))),
             reliable=bool(sampling_ok(a) and sampling_ok(b) and sampling_ok('measurement_relaxed')
                           and not loo[a].warning and not loo[b].warning)))
    pd.DataFrame(comparisons).to_csv(output/'conditional_loo.csv',index=False)
    pd.DataFrame([dict(model=k,max_pareto_k=float(v.pareto_k.max()),
                          threshold=float(v.good_k),warning=bool(v.warning)) for k,v in loo.items()]).to_csv(output/'loo_diagnostics.csv',index=False)
    grid=np.linspace(np.quantile(means,.025),np.quantile(means,.975),301)
    regions=[];ppc=[];curve_rows=[];contrast_rows=[]
    logit_curves=probability_draws(logit,grid,'logit')
    pd.DataFrame(dict(score=grid,mean=logit_curves.mean(axis=0),
        lower=np.quantile(logit_curves,.025,axis=0),upper=np.quantile(logit_curves,.975,axis=0))).to_csv(output/'association_curve.csv',index=False)
    rng=np.random.default_rng(seed+1)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,trace in fits.items():
        curves=probability_draws(trace,grid,name)
        derivative=np.gradient(curves,grid,axis=1)
        for i,s in enumerate(grid):
            curve_rows.append(dict(model=name,score=s,mean=float(curves[:,i].mean()),
                lower=float(np.quantile(curves[:,i],.025)),upper=float(np.quantile(curves[:,i],.975)),
                derivative_mean=float(derivative[:,i].mean()),
                derivative_lower=float(np.quantile(derivative[:,i],.025)),
                derivative_upper=float(np.quantile(derivative[:,i],.975))))
        for shift in (.1,.2,.3,.5):
            contrasts=probability_draws(trace,grid-shift,name)-curves
            for i,s in enumerate(grid):
                if s-shift < grid[0]:
                    continue  # Do not depict shifts outside the declared support.
                contrast_rows.append(dict(model=name,shift=shift,score=s,
                    mean=float(100*contrasts[:,i].mean()),lower=float(100*np.quantile(contrasts[:,i],.025)),
                    upper=float(100*np.quantile(contrasts[:,i],.975))))
        for fraction in (.7,.8,.9):
            for draw,d in enumerate(derivative):
                segments,peak,boundary=region_segments(grid,d,fraction)
                regions.append(dict(model=name,draw=draw,fraction=fraction,segments=segments,
                                    peak=peak,boundary=boundary))
        axes[0].plot(grid,curves.mean(axis=0),label=name)
        axes[0].fill_between(grid,*np.quantile(curves,[.025,.975],axis=0),alpha=.12)
        axes[1].plot(grid,derivative.mean(axis=0),label=name)
        shifted=probability_draws(trace,grid-.3,name)-curves
        axes[2].plot(grid,shifted.mean(axis=0)*100,label=name)
        # Integrate same score nodes used by the Bernoulli likelihood.
        predicted=np.mean([probability_draws(trace,nodes[:,j],name,data['covariates'][:,:2])
                           for j in range(nodes.shape[1])],axis=0)
        replicated=rng.binomial(1,predicted)
        bins=pd.qcut(means,4,labels=False,duplicates='drop')
        for group in [-1,*np.unique(bins)]:
            idx=np.ones(n,dtype=bool) if group==-1 else bins==group
            prevalence=replicated[:,idx].mean(axis=1)
            ppc.append(dict(model=name,group=int(group),n=int(idx.sum()),observed=float(data['y'][idx].mean()),
                replicated_mean=float(prevalence.mean()),lower=float(np.quantile(prevalence,.025)),upper=float(np.quantile(prevalence,.975))))
    for ax,title in zip(axes,['Conditional fall probability','dP/ds','Shift contrast (percentage points)']):
        ax.set(title=title,xlabel='Synthetic latent instability (SD)');ax.legend()
    fig.suptitle(DISCLAIMER);fig.tight_layout();fig.savefig(output/'synthetic_curves.pdf');plt.close(fig)
    (output/'regions.json').write_text(json.dumps(regions,indent=2))
    pd.DataFrame(curve_rows).to_csv(output/'probability_curves.csv',index=False)
    pd.DataFrame(contrast_rows).to_csv(output/'shift_contrasts.csv',index=False)
    pd.DataFrame(ppc).to_csv(output/'ppc.csv',index=False)
    pd.DataFrame(construct).to_csv(output/'construct_sensitivity.csv',index=False)
    pd.DataFrame(associations).to_csv(output/'associations.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'sampling_diagnostics.csv',index=False)
    report=dict(disclaimer=DISCLAIMER,scenario=scenario,n=n,fallers=int(data['y'].sum()),seed=seed,
         smoke=smoke,measurement_labels=labels,
         clinical_data_read=False,whole_pipeline_validation=False,
         software_versions={package:version(package) for package in ['numpy','scipy','pandas','pymc','arviz','scikit-learn']},
         diagnostic_gate='not smoke; R-hat <=1.01; bulk ESS >=400; zero divergences; no PSIS warning',
         region_definition='70/80/90% of draw-specific positive maximum derivative within central 95% score support',
         notes='Smoke results are invalid for inference. Flagged PSIS requires exact refits or K-fold; no rankings should be claimed from unreliable comparisons.')
    (output/'provenance.json').write_text(json.dumps(report,indent=2))
    print(DISCLAIMER, '\nOutput:',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=SCENARIOS,default='gradual')
    parser.add_argument('--n',type=int,default=269)
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    run(**vars(parser.parse_args()))
