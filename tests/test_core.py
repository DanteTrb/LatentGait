import numpy as np
import pandas as pd
import unittest
from latentgait.cohort import CohortError, lock_subject_cohort
from latentgait.diagnostics import derivative_region
from latentgait.pca import fit_domain_pcas

class CoreTests(unittest.TestCase):
    def test_subject_lock_rejects_duplicate_synthetic_rows(self):
        d = pd.DataFrame({"id": ["a", "a"], "fall": [0, 1]})
        with self.assertRaisesRegex(CohortError, "one row"):
            lock_subject_cohort(d, id_col="id", fall_col="fall", expected_n=None)

    def test_pca_is_oriented_without_outcome_data(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(30, 3))
        r=fit_domain_pcas(pd.DataFrame(x, columns=list("abc")), {"x": {"features": list("abc"), "directions": [1,1,1]}}, n_bootstrap=8)
        self.assertEqual(r.scores.shape, (30, 1))
        self.assertEqual(len(r.bootstrap), 3)

    def test_derivative_region_synthetic_sigmoid(self):
        grid=np.linspace(-4,4,401); p=1/(1+np.exp(-grid))
        _, region=derivative_region(grid,p)
        self.assertLess(region.min(), 0); self.assertGreater(region.max(), 0)
