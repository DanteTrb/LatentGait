import unittest
import numpy as np
from simulation.pipeline import (generate_data, domain_pca, indicators,
                                  curvature_basis, region_segments)


class SyntheticTests(unittest.TestCase):
    def test_deterministic_unique_ids(self):
        a,b=generate_data(),generate_data()
        self.assertEqual(len(np.unique(a['ids'])),269)
        np.testing.assert_array_equal(a['y'],b['y'])
        self.assertTrue(all(i.startswith('SYNTHETIC_') for i in a['ids']))

    def test_scenarios_share_features_not_selected_outcomes(self):
        a=generate_data(scenario='null');b=generate_data(scenario='transition')
        np.testing.assert_array_equal(a['features']['kinematics'],b['features']['kinematics'])
        self.assertFalse(np.array_equal(a['generating_probability'],b['generating_probability']))

    def test_pca_strategies(self):
        z,v=domain_pca(generate_data()['features'])
        for strategy,k in [('pc1',3),('complexity_pc2',4),('all_pc2',6)]:
            x,labels=indicators(z,v,strategy)
            self.assertEqual(x.shape,(269,k));self.assertEqual(len(labels),k)
            np.testing.assert_allclose(x.mean(axis=0),0,atol=1e-12)
        x,_=indicators(z,v,'variance80');self.assertGreaterEqual(x.shape[1],3)

    def test_spline_shape_and_linear_tails(self):
        self.assertEqual(curvature_basis(np.ones((4,5))).shape,(4,5,2))
        y=curvature_basis(np.array([5.,6.,7.]))
        np.testing.assert_allclose(y[2]-2*y[1]+y[0],0,atol=1e-12)

    def test_region_definition_not_tau_interval(self):
        x=np.linspace(-3,3,1201)
        derivative=np.exp(-x*x/2)/np.sqrt(2*np.pi)
        segments,peak,boundary=region_segments(x,derivative,.8)
        self.assertAlmostEqual(peak,0)
        self.assertFalse(boundary)
        self.assertAlmostEqual(segments[0][1],np.sqrt(-2*np.log(.8)),delta=.006)
        self.assertEqual(region_segments(x,-derivative)[0],[])
        self.assertTrue(region_segments(x,np.ones_like(x))[2])

    def test_disconnected_regions_remain_separate(self):
        x=np.linspace(-3,3,601)
        d=np.exp(-30*(x-1)**2)+np.exp(-30*(x+1)**2)
        self.assertEqual(len(region_segments(x,d)[0]),2)


if __name__=='__main__':
    unittest.main()
