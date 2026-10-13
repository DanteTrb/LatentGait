"""Software fixtures only; no clinical dataset or inferential results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from review_analysis.inputs import DOMAINS, COVARIATES, load_participant_table
from review_analysis.pipeline import domain_pca, indicators, curvature_basis, region_segments


def contract_fixture():
    # Deterministic test values exercise the schema, not disease relationships.
    x=np.arange(40,dtype=float)
    frame=pd.DataFrame({'ID':[f'TEST_{i:03d}' for i in range(40)],'target_bin':x.astype(int)%2})
    for j,c in enumerate(COVARIATES): frame[c]=x+j+1
    for j,c in enumerate([c for cols in DOMAINS.values() for c in cols]):
        frame[c]=np.sin(x*(j+1)*.137)+.01*x
    return frame


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'fixture.csv'
    def tearDown(self):self.temp.cleanup()
    def read(self,frame):
        frame.to_csv(self.path,index=False)
        return load_participant_table(self.path)
    def test_schema_and_size_from_file(self):
        d=self.read(contract_fixture())
        self.assertEqual(len(d['y']),40)
        self.assertEqual(len(d['input_sha256']),64)
        self.assertEqual(d['covariates'].shape,(40,5))
    def test_missing_file_no_fallback(self):
        with self.assertRaises(FileNotFoundError):load_participant_table(self.path)
    def test_repeated_id_rejected(self):
        f=contract_fixture();f.loc[1,'ID']=f.loc[0,'ID']
        with self.assertRaisesRegex(ValueError,'Repeated'):self.read(f)
    def test_missing_id_rejected(self):
        f=contract_fixture();f.loc[0,'ID']=' '
        with self.assertRaisesRegex(ValueError,'Missing/empty'):self.read(f)
    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):self.read(contract_fixture().drop(columns='Age'))
    def test_missing_measurement_rejected(self):
        f=contract_fixture();f.loc[0,'Age']=np.nan
        with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):self.read(f)
    def test_constant_rejected(self):
        f=contract_fixture();f['Age']=1
        with self.assertRaisesRegex(ValueError,'Constant'):self.read(f)
    def test_invalid_outcome_rejected(self):
        f=contract_fixture();f.loc[0,'target_bin']=2
        with self.assertRaisesRegex(ValueError,'binary'):self.read(f)
    def test_pca_strategies(self):
        d=self.read(contract_fixture());z,v=domain_pca(d['features'])
        for strategy,k in [('pc1',3),('complexity_pc2',4),('all_pc2',6)]:
            matrix,labels=indicators(z,v,strategy)
            self.assertEqual(matrix.shape,(40,k))
            self.assertEqual(len(labels),k)
    def test_spline_linear_tails(self):
        y=curvature_basis(np.array([5.,6.,7.]))
        np.testing.assert_allclose(y[2]-2*y[1]+y[0],0,atol=1e-12)
    def test_region_not_location_interval(self):
        x=np.linspace(-3,3,1201);d=np.exp(-x*x/2)
        segments,peak,boundary=region_segments(x,d,.8)
        self.assertAlmostEqual(peak,0);self.assertFalse(boundary)
        self.assertAlmostEqual(segments[0][1],np.sqrt(-2*np.log(.8)),delta=.006)
        self.assertEqual(region_segments(x,-d)[0],[])


if __name__=='__main__':unittest.main()
