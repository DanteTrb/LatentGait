"""Generated mathematical/software fixtures; no participant data or fitted paper values."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from latentgait.cohort import link_tables, validate_frame, zscore
from latentgait.features import (
    ami_delay,
    event_candidates,
    fnn_dimension,
    harmonic_features,
    multiscale_entropy,
    percent_determinism,
    resample_rbf,
    sample_entropy,
    trim_boundaries,
)
from latentgait.functional_form import common_region, derivative_region, rcs_basis
from latentgait.pca import LARGEST_LOADING_POSITIVE, _largest_loading_sign, fit_domain_pcas, indicator_matrix
from latentgait.synthetic import synthetic_frame, synthetic_protocol
from latentgait.workflow import external_path, validate_protocol


class CohortTests(unittest.TestCase):
    def test_default_lock_rejects_other_cohort(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_frame(synthetic_frame())

    def test_synthetic_count_override_is_explicit(self):
        self.assertEqual(len(validate_frame(synthetic_frame(), expected=(40, 20, 20))), 40)

    def test_whitespace_duplicate_and_missing_ids(self):
        for value in (" SOFTWARE_0000 ", " ", None):
            frame = synthetic_frame()
            frame.loc[1, "participant_id"] = value
            with self.assertRaises(ValueError):
                validate_frame(frame, expected=None)

    def test_no_missing_or_invalid_outcome_or_qc(self):
        for field, value in [
            ("mse_v", np.nan),
            ("age_years", np.inf),
            ("fall_last_year_binary", 2),
            ("qc_pass", 0),
            ("steady_state_confirmed", 0),
            ("retained_strides", 19),
            ("retained_strides", 20.5),
            ("gait_speed_m_s", 0),
            ("det_v", 101),
        ]:
            frame = synthetic_frame()
            if field == "retained_strides":
                frame[field] = frame[field].astype(float)
            frame.loc[0, field] = value
            with self.assertRaises(ValueError, msg=field):
                validate_frame(frame, expected=None)

    def test_constant_feature_rejected(self):
        frame = synthetic_frame()
        frame.hr_v = 1
        with self.assertRaisesRegex(ValueError, "Constant"):
            validate_frame(frame, expected=None)

    def test_protocol_is_fail_closed(self):
        cfg = synthetic_protocol()
        validate_protocol(cfg)
        cfg["orientation"]["rhythmicity"] = None
        with self.assertRaisesRegex(ValueError, "orientation"):
            validate_protocol(cfg)

    def test_final_orientation_cannot_be_overridden_or_silently_migrated(self):
        for name, value in [
            ("kinematics", [-1, -1, -1]),
            ("rhythmicity", [1] * 9),
            ("complexity", "outcome_selected"),
        ]:
            cfg = synthetic_protocol()
            cfg["orientation"][name] = value
            with self.assertRaisesRegex(ValueError, "orientation"):
                validate_protocol(cfg)
            with self.assertRaisesRegex(ValueError, "orientation"):
                fit_domain_pcas(synthetic_frame(), cfg["orientation"], n_bootstrap=2)
        cfg = synthetic_protocol()
        cfg["protocol_version"] = 2
        with self.assertRaisesRegex(ValueError, "version-3"):
            validate_protocol(cfg)

    def test_public_example_has_complete_pca_orientation(self):
        import yaml
        from latentgait.workflow import ROOT

        cfg = yaml.safe_load((ROOT / "configs/analysis.example.yaml").read_text())
        self.assertEqual(cfg["orientation"]["rhythmicity"], LARGEST_LOADING_POSITIVE)
        self.assertEqual(cfg["orientation"]["complexity"], LARGEST_LOADING_POSITIVE)
        self.assertTrue(cfg["orientation_rationale"])
        cfg["qc_protocol_reference"] = "Generated software-test QC reference"
        cfg["measurement"]["acknowledge_archived_priors"] = True
        validate_protocol(cfg)

    def test_final_choices_are_fixed_in_protocol(self):
        for field, value in [("bootstrap_scaling", "fixed"), ("cumulative_rule", "lower_tail_90")]:
            cfg = synthetic_protocol()
            cfg[field] = value
            with self.assertRaises(ValueError):
                validate_protocol(cfg)

    def test_private_output_guard(self):
        from latentgait.workflow import ROOT

        with self.assertRaisesRegex(ValueError, "outside"):
            external_path(ROOT / "results")

    def test_linkage_uses_keys_and_rejects_partial_matches(self):
        frame = synthetic_frame()
        clinical = frame.drop(columns=["hr_v"])
        features = frame[["participant_id", "trial_id", "hr_v"]].sample(frac=1, random_state=2)
        linked = link_tables(clinical, features)
        np.testing.assert_allclose(linked.hr_v, frame.hr_v)
        with self.assertRaisesRegex(ValueError, "key sets"):
            link_tables(clinical, features.iloc[:-1])


class NumericalTests(unittest.TestCase):
    def test_largest_loading_sign_breaks_exact_ties_by_schema_order(self):
        for component, expected in [([-0.8, 0.8, 0.1], -1), ([0.8, -0.8, 0.1], 1), ([0.1, -0.9, 0.2], -1)]:
            component = np.asarray(component)
            self.assertEqual(_largest_loading_sign(component), expected)
            np.testing.assert_array_equal(
                component * _largest_loading_sign(component),
                -component * _largest_loading_sign(-component),
            )

    def test_pca_orientation_preserves_anchor_and_canonical_nonanchor_loadings(self):
        from sklearn.decomposition import PCA
        from latentgait.cohort import DOMAINS

        frame = synthetic_frame()
        pca = fit_domain_pcas(frame, synthetic_protocol()["orientation"], n_bootstrap=6)
        z = zscore(frame[DOMAINS["kinematics"]])[0]
        component = PCA(svd_solver="full").fit(z).components_[0]
        component *= np.sign(component @ np.ones(3))
        np.testing.assert_allclose(pca.scores["kinematics"][:, 0], z @ component)
        for (domain, number), rows in pca.loadings.groupby(["domain", "component"], sort=False):
            values = rows.loading.to_numpy()
            if domain == "kinematics" and number == 1:
                self.assertGreater(values.sum(), 0)
            else:
                self.assertGreater(values[np.argmax(abs(values))], 0)

    def test_pca_and_bootstrap_are_invariant_to_solver_sign_reversals(self):
        from sklearn.decomposition import PCA

        class ReversedPCA(PCA):
            def fit(self, x, y=None):
                super().fit(x, y)
                self.components_ *= -1
                return self

        frame = synthetic_frame()
        cfg = synthetic_protocol()
        reference = fit_domain_pcas(frame, cfg["orientation"], n_bootstrap=6)
        with patch("latentgait.pca.PCA", ReversedPCA):
            reversed_fit = fit_domain_pcas(frame, cfg["orientation"], n_bootstrap=6)
        for domain in reference.scores:
            np.testing.assert_allclose(reference.scores[domain], reversed_fit.scores[domain])
        np.testing.assert_allclose(reference.loadings.loading, reversed_fit.loadings.loading)
        np.testing.assert_allclose(
            reference.bootstrap[["mean", "lower", "upper"]], reversed_fit.bootstrap[["mean", "lower", "upper"]]
        )

    def test_domains_share_draws_and_stability_extends_primary_draws(self):
        frame = synthetic_frame()
        from latentgait.cohort import DOMAINS

        frame[DOMAINS["complexity"]] = frame[DOMAINS["kinematics"]].to_numpy()
        directions = synthetic_protocol()["orientation"]
        pca = fit_domain_pcas(frame, directions, n_bootstrap=6, n_bootstrap_check=12)
        a = pca.bootstrap.loc[pca.bootstrap.domain.eq("kinematics"), ["mean", "lower", "upper"]]
        b = pca.bootstrap.loc[pca.bootstrap.domain.eq("complexity"), ["mean", "lower", "upper"]]
        np.testing.assert_allclose(a, b)
        baseline = fit_domain_pcas(frame, directions, n_bootstrap=6)
        self.assertTrue(pca.bootstrap.equals(baseline.bootstrap))
        self.assertTrue(pca.bootstrap_stability.n_resamples.eq(12).all())
        self.assertEqual(len(pca.component_diagnostics), 3)

    def test_weighted_central_interval_is_empirical_and_tie_invariant(self):
        from latentgait.reporting import cumulative_contrast_interval

        r = cumulative_contrast_interval(np.arange(100), np.ones(100))
        self.assertEqual(r["interval"], [4.0, 94.0])
        self.assertGreaterEqual(r["included_weight_fraction"], 0.90)
        tied = cumulative_contrast_interval(np.array([0.0, 1.0, 0.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        shuffled = cumulative_contrast_interval(np.array([2.0, 0.0, 1.0, 0.0]), np.array([4.0, 3.0, 2.0, 1.0]))
        self.assertEqual(tied, shuffled)
        self.assertIsNone(cumulative_contrast_interval(np.arange(3), np.zeros(3))["interval"])

    def test_separate_domain_pca_bootstrap_and_retention(self):
        frame = synthetic_frame()
        directions = synthetic_protocol()["orientation"]
        pca = fit_domain_pcas(frame, directions, n_bootstrap=12)
        self.assertEqual(len(pca.bootstrap), 15)
        self.assertTrue((pca.bootstrap.lower <= pca.bootstrap.upper).all())
        for strategy, k in [("pc1", 3), ("complexity_pc2", 4), ("all_pc2", 6)]:
            x, names = indicator_matrix(pca, strategy)
            self.assertEqual(x.shape, (40, k))
            self.assertEqual(names[0], "kinematics_PC1")
            np.testing.assert_allclose(x.std(0), 1)
        x, _ = indicator_matrix(pca, "variance80")
        self.assertGreaterEqual(x.shape[1], 3)

    def test_pca_never_reads_falls(self):
        f = synthetic_frame()
        cfg = synthetic_protocol()
        p = fit_domain_pcas(f, cfg["orientation"], n_bootstrap=4)
        f.fall_last_year_binary = 1 - f.fall_last_year_binary
        q = fit_domain_pcas(f, cfg["orientation"], n_bootstrap=4)
        for domain in p.scores:
            np.testing.assert_allclose(p.scores[domain], q.scores[domain])

    def test_standardized_measurement_sd_must_share_mean_scale(self):
        x = np.array([-3.0, 0.0, 3.0])
        _, _, scale = zscore(x)
        self.assertAlmostEqual(float(0.2 / scale), 0.2 / np.sqrt(6))

    def test_spline_linear_tails_and_continuity(self):
        knots = np.array([-2.0, -0.4, 0.6, 2.0])
        for x in ([-5.0, -4.0, -3.0], [3.0, 4.0, 5.0]):
            b = rcs_basis(x, knots)
            np.testing.assert_allclose(b[2] - 2 * b[1] + b[0], 0, atol=1e-12)
        for k in knots:
            b = rcs_basis([k - 1e-7, k, k + 1e-7], knots)
            self.assertLess(abs(b[0] - b[2]).max(), 1e-5)

    def test_probit_80pct_geometry_and_absent_region(self):
        x = np.linspace(-3, 3, 6001)
        r = derivative_region(x, ndtr(x))
        self.assertAlmostEqual(r["peak_location"], 0, places=3)
        self.assertAlmostEqual(r["segments"][0][1], np.sqrt(-2 * np.log(0.8)), delta=0.002)
        self.assertEqual(derivative_region(x, ndtr(-x))["segments"], [])

    def test_region_intersection_does_not_fill_gaps(self):
        self.assertEqual(common_region([[[0, 1], [2, 3]], [[0.5, 2.5]], [[0, 4]]]), [[0.5, 1], [2, 2.5]])
        self.assertEqual(common_region([[[0, 1]], []]), [])

    def test_ihr_uses_power_parity_not_amplitude(self):
        t = np.arange(3000) / 100
        x = np.cos(2 * np.pi * t) + 2 * np.cos(4 * np.pi * t)
        boundaries = np.arange(0, 3001, 100)
        hr, ihr = harmonic_features(x, 100, boundaries, "v")
        self.assertAlmostEqual(hr, 2, delta=0.01)
        self.assertAlmostEqual(ihr, 80, delta=0.05)
        hr_ml, ihr_ml = harmonic_features(x, 100, boundaries, "ml")
        self.assertAlmostEqual(hr_ml, 0.5, delta=0.01)
        self.assertAlmostEqual(ihr_ml, 20, delta=0.05)

    def test_trim_requires_twenty_strides_after_removal(self):
        x = np.zeros(2500)
        self.assertEqual(len(trim_boundaries(x, np.arange(0, 2401, 100))) - 1, 20)
        with self.assertRaises(ValueError):
            trim_boundaries(x, np.arange(0, 2301, 100))

    def test_entropy_matches_bruteforce(self):
        x = np.array([0.0, 0.1, 0.2, 0.1, 0.0, 0.2, 0.1, 0.3, 0.1, 0.0])
        templates = np.lib.stride_tricks.sliding_window_view(x, 3)
        a = b = 0
        for i in range(len(templates)):
            for j in range(i):
                b += np.max(abs(templates[i, :2] - templates[j, :2])) <= 0.11
                a += np.max(abs(templates[i] - templates[j])) <= 0.11
        self.assertAlmostEqual(sample_entropy(x, tolerance=0.11), -np.log(a / b))

    def test_mse_and_recurrence_are_finite_on_generated_signal(self):
        rng = np.random.default_rng(22)
        self.assertTrue(np.isfinite(multiscale_entropy(rng.normal(size=2000))))
        t = np.linspace(0, 20 * np.pi, 300)
        det = percent_determinism(np.sin(t), 3, 5, radius_fraction=0.1, lmin=2, theiler=0)
        self.assertTrue(0 <= det <= 100)

    def test_rbf_and_documented_parameter_searches(self):
        x = np.linspace(-1, 1, 64)
        resampled = resample_rbf(x, kernel="linear", smoothing=0, neighbors=None, epsilon=None)
        np.testing.assert_allclose(resampled, np.linspace(-1, 1, 2000), atol=1e-10)
        rng = np.random.default_rng(23)
        periodic = np.sin(np.arange(2000) * 0.13) + 0.001 * rng.normal(size=2000)
        delay = ami_delay(periodic, bins=10)
        self.assertTrue(5 <= delay <= 19)
        dimension = fnn_dimension(periodic, delay, rtol=17, atol=3, theiler=0)
        self.assertTrue(3 <= dimension <= 10)
        peaks, wavelet = event_candidates(periodic, 100, fir_numtaps=21, fir_window="hamming")
        self.assertGreater(len(peaks), 0)
        self.assertEqual(len(wavelet), len(periodic))


if __name__ == "__main__":
    unittest.main()
