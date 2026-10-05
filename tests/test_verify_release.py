"""Invalid outputs must never receive a release-verification PASS."""
import json
from pathlib import Path
import unittest

import pandas as pd

from verify_release import validate_outputs

ROOT = Path(__file__).resolve().parents[1]


class OutputVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tr = pd.read_csv(ROOT / 'config/transects.csv')
        self.m = pd.read_csv(ROOT / 'results/tables/measurements.csv')
        self.s = pd.read_csv(ROOT / 'results/tables/summary.csv')
        self.rb = pd.read_csv(ROOT / 'results/tables/robustness_summary.csv')
        self.manifest = json.loads((ROOT / 'results/tables/analysis_manifest.json').read_text())

    def verify(self):
        validate_outputs(self.tr, self.m, self.s, self.rb, self.manifest)

    def test_released_outputs_pass(self):
        self.verify()

    def test_nan_measurements_rejected(self):
        self.m['FWHM'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.verify()

    def test_infinite_measurements_rejected(self):
        self.m.loc[0, 'FWHM'] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.verify()

    def test_nan_robustness_rejected(self):
        self.rb['Gx'] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.verify()

    def test_duplicate_measurements_rejected(self):
        self.m.loc[1] = self.m.loc[0]
        with self.assertRaisesRegex(RuntimeError, 'keys'):
            self.verify()

    def test_stale_summary_rejected(self):
        self.m['FWHM'] += 1
        self.m['x50_right'] += 1
        with self.assertRaisesRegex(RuntimeError, 'measurement/summary'):
            self.verify()

    def test_duplicate_robustness_rejected(self):
        self.rb.loc[1] = self.rb.loc[0]
        with self.assertRaisesRegex(RuntimeError, 'duplicated'):
            self.verify()

    def test_bad_leave_parent_values_rejected(self):
        index = self.rb.index[-1]
        self.rb.loc[index, 'UV_mean_W50_reference_px'] += 1
        self.rb.loc[index, 'Gx'] = (self.rb.loc[index, 'UV_mean_W50_reference_px']
                                  / self.rb.loc[index, 'NIR_mean_W50_reference_px'])
        with self.assertRaisesRegex(RuntimeError, 'Robustness UV mean'):
            self.verify()

    def test_bad_manifest_rejected(self):
        self.manifest['Gx_ratio_of_means'] += .1
        with self.assertRaisesRegex(RuntimeError, 'manifest'):
            self.verify()
