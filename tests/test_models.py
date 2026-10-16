"""Bayesian graph and integration checks on independently generated software fixtures."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from latentgait.cohort import zscore
from latentgait.functional_form import rcs_basis, rcs_tensor
from latentgait.models import (
    Sampling,
    build_measurement,
    build_outcome,
    marginal_probability,
    prior_probability_curves,
    score_summary,
    spline_spec,
)
from latentgait.synthetic import synthetic_frame


class ModelTests(unittest.TestCase):
    def test_odds_ratios_are_only_reported_for_logit_coefficients(self):
        import arviz as az
        from latentgait.reporting import coefficient_summary

        values = np.linspace(-0.3, 0.6, 40).reshape(2, 20)
        trace = az.from_dict(posterior={"beta_linear": values, "beta_z": values})
        probit = coefficient_summary(trace, ["beta_linear"], link="probit")
        self.assertNotIn("or_mean", probit.columns)
        self.assertEqual(probit.parameter_scale.iloc[0], "probit")
        measurement = coefficient_summary(trace, ["beta_z"])
        self.assertNotIn("or_mean", measurement.columns)
        logistic = coefficient_summary(trace, ["beta_z"], link="logit")
        self.assertAlmostEqual(logistic.or_mean.iloc[0], np.exp(values).mean())
        self.assertEqual(logistic.parameter_scale.iloc[0], "log_odds")

    def test_spline_prior_curves_are_prior_only_probability_draws(self):
        f = synthetic_frame()
        mean = zscore(f.pelvis_tilt_range)[0]
        spec = spline_spec(mean)
        model = build_outcome(
            "spline", mean, np.full(len(f), 0.1), f.fall_last_year_binary.to_numpy(), np.empty((len(f), 0)), spline=spec
        )
        p = prior_probability_curves(model, np.linspace(-2, 2, 21), "spline", spline=spec, draws=10, seed=19)
        self.assertEqual(p.shape, (10, 21))
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_production_nuts_settings(self):
        s = Sampling()
        self.assertEqual((s.chains, s.tune, s.draws, s.target_accept), (4, 3000, 3000, 0.95))

    def test_tensor_spline_agrees_with_numeric_and_has_gradient(self):
        import pytensor
        import pytensor.tensor as pt

        x = pt.dvector("x")
        knots = np.array([-2.0, -0.4, 0.6, 2.0])
        b = rcs_tensor(x, knots, np.zeros(3), np.ones(3))
        fn = pytensor.function([x], [b, pt.grad(pt.sum(b), x)])
        samples = np.linspace(-4, 4, 101)
        values, gradient = fn(samples)
        np.testing.assert_allclose(values, rcs_basis(samples, knots))
        self.assertTrue(np.isfinite(gradient).all())

    def test_all_paper_graphs_have_finite_logdensity(self):
        f = synthetic_frame()
        mean = zscore(f.pelvis_tilt_range)[0]
        sd = np.repeat(0.15, len(f))
        y = f.fall_last_year_binary.to_numpy()
        for form in ("fixed", "free", "spline", "logit"):
            c = np.zeros((len(f), 2 if form == "logit" else 0))
            model = build_outcome(form, mean, sd, y, c, spline=spline_spec(mean))
            self.assertTrue(np.isfinite(model.compile_logp()(model.initial_point())), msg=form)
            if form == "fixed":
                self.assertNotIn("intercept", model.named_vars)
                self.assertNotIn("beta_cov", model.named_vars)
            if form == "free":
                self.assertIn("log_slope", model.named_vars)
                self.assertNotIn("beta_z", model.named_vars)
        with self.assertRaises(ValueError):
            build_outcome("fixed", mean, sd, y, np.zeros((len(y), 2)))

    def test_measurement_anchor_sd_is_point_one_not_point_zero_zero_one(self):
        import pymc as pm

        model = build_measurement(
            np.zeros((12, 3)),
            ["kinematics_PC1", "rhythmicity_PC1", "complexity_PC1"],
            anchor_sd=0.1,
            loading_prior_sd=1,
            residual_prior_sd=0.5,
        )
        with model:
            prior = pm.sample_prior_predictive(draws=200, random_seed=17)
        latent = np.asarray(prior.prior.latent)
        observed = np.asarray(prior.prior_predictive.indicators)[..., 0]
        self.assertAlmostEqual(float((observed - latent).std()), 0.1, delta=0.01)

    def test_measurement_sd_is_rescaled(self):
        import arviz as az

        draws = np.stack([np.array([-4.0, 0.0, 4.0]) + 0.1 * i for i in range(8)]).reshape(2, 4, 3)
        fit = az.from_dict(posterior={"latent": draws}, coords={"subject": np.arange(3)}, dims={"latent": ["subject"]})
        mean, sd, scale = score_summary(fit)
        self.assertAlmostEqual(float(mean.std()), 1)
        np.testing.assert_allclose(sd, draws.reshape(-1, 3).std(0) / scale["sd"])

    def test_marginal_probit_matches_independent_numerical_integration(self):
        import arviz as az
        from scipy.integrate import quad
        from scipy.stats import norm

        tau, slope = 0.4, 1.3
        trace = az.from_dict(posterior={"tau": np.full((2, 20), tau), "slope": np.full((2, 20), slope)})
        p = marginal_probability(trace, np.array([0.2]), np.array([0.3]), "free")
        reference = quad(lambda x: ndtr(slope * (x - tau)) * norm.pdf(x, 0.2, 0.3), -np.inf, np.inf)[0]
        np.testing.assert_allclose(p, reference, rtol=1e-10)

    def test_exact_loo_excludes_outcome_and_replaces_flagged_point(self):
        import arviz as az
        from latentgait.diagnostics import outcome_loo

        n = 12
        mean, sd, y = np.linspace(-1, 1, n), np.repeat(0.2, n), np.arange(n) % 2
        trace = az.from_dict(posterior={"tau": np.zeros((2, 20))})
        fake_loo = SimpleNamespace(loo_i=np.repeat(-1.0, n), pareto_k=np.r_[0.9, np.repeat(0.1, n - 1)], good_k=0.7)
        observed_sizes = []

        def check_refit(model, settings):
            observed_sizes.append(model.rvs_to_values[model["fall"]].data.shape[0])
            return trace

        with (
            patch("arviz.loo", return_value=fake_loo),
            patch("latentgait.diagnostics.sample", side_effect=check_refit),
            patch("latentgait.diagnostics.sampling_diagnostics", return_value={"adequate": True}),
        ):
            result = outcome_loo(trace, "fixed", mean, sd, y, np.empty((n, 0)), None, Sampling(), exact=True)
        self.assertEqual(observed_sizes, [n - 1])
        self.assertEqual(result["exact_refits"], 1)
        self.assertAlmostEqual(result["pointwise"][0], np.log(1 - ndtr(mean[0] / np.sqrt(1 + sd[0] ** 2))))

    def test_hypothetical_contrast_peak_matches_link_identity(self):
        import arviz as az
        from latentgait.reporting import contrasts

        # For fixed tau=0 and shift delta, |Phi(s-delta)-Phi(s)| peaks at s=delta/2.
        trace = az.from_dict(posterior={"tau": np.zeros((2, 20)), "z_true": np.zeros((2, 20, 3))})
        grid = np.linspace(-2, 2, 4001)
        _, maxima, summary = contrasts(
            trace, grid, baseline_scores=np.array([-1.0, 0.0, 1.0]), cumulative_rule="lower_tail_90"
        )
        for row in maxima.itertuples():
            self.assertAlmostEqual(row.location, row.shift / 2, delta=0.0011)
            self.assertAlmostEqual(row.mean, 1 - 2 * ndtr(row.shift / 2), places=7)
        self.assertEqual(summary["cumulative_rule"], "lower_tail_90")

    def test_curve_region_checks_preserve_the_predeclared_primary(self):
        import arviz as az
        from latentgait.reporting import curve_region_checks

        trace = az.from_dict(posterior={"tau": np.zeros((2, 20))})
        checks = curve_region_checks(trace, np.linspace(-3, 3, 1000), "fixed")
        self.assertEqual(len(checks["checks"]), 6)
        self.assertEqual(checks["primary_fraction"], 0.8)
        self.assertEqual(checks["primary_grid_points"], 1000)
        self.assertEqual(checks["refined_grid_points"], 2000)


if __name__ == "__main__":
    unittest.main()
