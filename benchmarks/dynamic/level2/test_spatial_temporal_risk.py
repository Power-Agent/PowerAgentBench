import unittest
import numpy as np
import pandas as pd
from spatial_temporal_risk import spatial_temporal_risk

class SpatialTemporalRiskTests(unittest.TestCase):
    def test_localized_and_widespread_disturbance_are_distinguished(self):
        t = pd.Series([0.0, 1.0, 2.0])
        local = spatial_temporal_risk(pd.DataFrame({"A": [0.90]*3, "B": [1.0]*3}), t)
        wide = spatial_temporal_risk(pd.DataFrame({"A": [0.90]*3, "B": [0.90]*3}), t)
        self.assertLess(local["mean_iov_undervoltage"], wide["mean_iov_undervoltage"])
        self.assertEqual(local["affected_bus_fraction_undervoltage"], 0.5)
        self.assertEqual(wide["affected_bus_fraction_undervoltage"], 1.0)

    def test_tail_detects_brief_severe_excursion(self):
        t = pd.Series([0., 1., 2., 3., 4.])
        v = pd.DataFrame({"A": [1., 1., .90, 1., 1.], "B": [1.]*5})
        result = spatial_temporal_risk(v, t, tail_fraction=.2)
        self.assertGreater(result["tail_mean_undervoltage"], result["mean_iov_undervoltage"])

    def test_irregular_sample_times(self):
        result = spatial_temporal_risk(pd.DataFrame({"A": [.95, .90, .95]}), pd.Series([0., 1., 3.]))
        self.assertAlmostEqual(result["mean_iov_undervoltage"], .5)

    def test_under_over_metrics_are_independent(self):
        t = pd.Series([0., 1., 2.])
        under = spatial_temporal_risk(pd.DataFrame({"A": [.90]*3}), t)
        over = spatial_temporal_risk(pd.DataFrame({"A": [1.10]*3}), t)
        self.assertEqual(under["worst_bus_iov_overvoltage"], 0.0)
        self.assertEqual(over["worst_bus_iov_undervoltage"], 0.0)
        self.assertEqual(over["worst_bus_iov_overvoltage"], 1.0)

    def test_empty_and_single_sample_inputs(self):
        empty = spatial_temporal_risk(pd.DataFrame(), pd.Series(dtype=float))
        one = spatial_temporal_risk(pd.DataFrame({"A": [.90]}), pd.Series([0.]))
        self.assertEqual(empty["bus_count"], 0)
        self.assertEqual(one["mean_iov_undervoltage"], 0.0)

    def test_invalid_contract(self):
        v = pd.DataFrame({"A": [1., 1.]})
        with self.assertRaisesRegex(ValueError, "same number"):
            spatial_temporal_risk(v, pd.Series([0.]))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            spatial_temporal_risk(v, pd.Series([1., 1.]))
        with self.assertRaisesRegex(ValueError, "finite"):
            spatial_temporal_risk(pd.DataFrame({"A": [1., np.nan]}), pd.Series([0., 1.]))
        with self.assertRaisesRegex(ValueError, "tail_fraction"):
            spatial_temporal_risk(v, pd.Series([0., 1.]), tail_fraction=0.)

if __name__ == "__main__":
    unittest.main()
