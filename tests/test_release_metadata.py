"""Integrity checks for the sole frozen-geometry input."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import build_release_metadata as metadata


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transects = self.root / 'config' / 'transects.csv'
        self.expected = self.root / 'provenance' / 'input_checksums.sha256'
        self.observed = self.root / 'results' / 'input_checksums_observed.sha256'
        self.transects.parent.mkdir()
        self.expected.parent.mkdir()
        self.transects.write_bytes(metadata.TRAN.read_bytes())
        self.pin_hash()
        for name, value in [('ROOT', self.root), ('TRAN', self.transects),
                            ('EXPECTED', self.expected), ('OBSERVED', self.observed)]:
            context = patch.object(metadata, name, value)
            context.start()
            self.addCleanup(context.stop)

    def pin_hash(self):
        digest = hashlib.sha256(self.transects.read_bytes()).hexdigest()
        self.expected.write_text(f'{digest}  config/transects.csv\n', encoding='utf-8')

    def test_actual_input_alone_passes(self):
        metadata.main()
        self.assertEqual(self.expected.read_text(), self.observed.read_text())

    def test_outcome_column_rejected(self):
        table = pd.read_csv(self.transects)
        table['UV_W50'] = 1.0
        table.to_csv(self.transects, index=False)
        self.pin_hash()
        with self.assertRaisesRegex(RuntimeError, 'geometry schema'):
            metadata.main()

    def test_missing_geometry_rejected(self):
        table = pd.read_csv(self.transects)
        table.loc[0, 'center_x'] = float('nan')
        table.to_csv(self.transects, index=False)
        self.pin_hash()
        with self.assertRaisesRegex(RuntimeError, 'missing values'):
            metadata.main()

    def test_wrong_transect_count_rejected(self):
        table = pd.read_csv(self.transects).iloc[:-1]
        table.to_csv(self.transects, index=False)
        self.pin_hash()
        with self.assertRaisesRegex(RuntimeError, 'five fixed transects'):
            metadata.main()

    def test_modified_geometry_checksum_rejected(self):
        table = pd.read_csv(self.transects)
        table.loc[0, 'center_x'] += 1
        table.to_csv(self.transects, index=False)
        with self.assertRaisesRegex(RuntimeError, 'checksum failure'):
            metadata.main()


if __name__ == '__main__':
    unittest.main()
