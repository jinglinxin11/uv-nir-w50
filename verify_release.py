"""Verify immutable inputs and cross-check regenerated numerical outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
T = ROOT / 'results' / 'tables'
ABS_TOL = 1e-10
EXPECTED = {
    'UV_mean_W50_reference_px': 14.526104873392228,
    'NIR_mean_W50_reference_px': 6.050997585336955,
    'Gx_ratio_of_means': 2.400613232534174,
    'reduction_percent': 58.34397701189191,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_columns(frame, columns, label):
    if not set(columns).issubset(frame.columns):
        raise RuntimeError(f'{label}: missing required numerical columns')
    try:
        values = frame[list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f'{label}: invalid numerical values') from error
    if not np.isfinite(values).all():
        raise RuntimeError(f'{label}: non-finite numerical values')


def equal(observed, expected, label):
    if not np.isclose(float(observed), float(expected), rtol=0, atol=ABS_TOL):
        raise RuntimeError(f'{label} mismatch: {observed} vs {expected}')


def means(measurements):
    uv = float(measurements.loc[measurements.condition.eq('UV'), 'FWHM'].mean())
    nir = float(measurements.loc[measurements.condition.eq('NIR'), 'FWHM'].mean())
    return uv, nir


def validate_outputs(transects, measurements, summary, robustness, manifest):
    if (len(transects) != 5 or transects.ROI_ID.nunique() != 5
            or transects.parent_ID.nunique() != 4):
        raise RuntimeError('Fixed transect check failed')
    expected_keys = {(condition, roi) for condition in ('UV', 'NIR')
                     for roi in transects.ROI_ID}
    keys = list(zip(measurements.condition, measurements.ROI_ID))
    if len(keys) != 10 or len(set(keys)) != 10 or set(keys) != expected_keys:
        raise RuntimeError('Measurement condition/transect keys are invalid or duplicated')
    finite_columns(measurements, ('FWHM', 'peak_to_noise'), 'Measurements')
    if (not measurements.valid.eq(True).all() or (measurements.FWHM <= 0).any()
            or (measurements.peak_to_noise < 3).any()):
        raise RuntimeError('Measurement validity check failed')
    if {'x50_left', 'x50_right'}.issubset(measurements.columns):
        finite_columns(measurements, ('x50_left', 'x50_right'), 'Half-maximum crossings')
        for row in measurements.itertuples():
            equal(row.FWHM, row.x50_right - row.x50_left, 'Width/crossing consistency')
    if len(summary) != 1:
        raise RuntimeError('Expected exactly one primary summary record')
    finite_columns(summary, tuple(EXPECTED), 'Primary summary')
    uv, nir = means(measurements)
    computed = dict(zip(EXPECTED, (uv, nir, uv / nir, 100 * (1 - nir / uv))))
    record = summary.iloc[0]
    for key, expected in EXPECTED.items():
        equal(record[key], computed[key], f'{key} measurement/summary')
        equal(record[key], expected, key)
        if key not in manifest:
            raise RuntimeError(f'Analysis manifest missing {key}')
        equal(manifest[key], computed[key], f'{key} analysis manifest')
    for key, expected in (('technical_transects', 5), ('parent_ROIs', 4)):
        equal(record[key], expected, key)
        equal(manifest[key], expected, f'{key} analysis manifest')
    if record['measurement_channel'] != 'CIE Lab a*' or manifest.get('measurement_channel') != 'CIE Lab a*':
        raise RuntimeError('Analysis channel mismatch')
    expected_analyses = {f'Gaussian sigma = {sigma:g} ref px': 5
                         for sigma in (0, .5, .75, 1.0)}
    expected_analyses.update({f'Leave parent {parent} out': int(transects.parent_ID.ne(parent).sum())
                              for parent in transects.parent_ID.unique()})
    if (len(robustness) != 8 or robustness.analysis.duplicated().any()
            or set(robustness.analysis) != set(expected_analyses)):
        raise RuntimeError('Robustness analyses are missing, unexpected or duplicated')
    finite_columns(robustness, ('paired_transects', 'UV_mean_W50_reference_px',
                               'NIR_mean_W50_reference_px', 'Gx'), 'Robustness')
    for row in robustness.itertuples():
        equal(row.paired_transects, expected_analyses[row.analysis], 'Robustness transect count')
        if row.UV_mean_W50_reference_px <= 0 or row.NIR_mean_W50_reference_px <= 0:
            raise RuntimeError('Robustness widths must be positive')
        equal(row.Gx, row.UV_mean_W50_reference_px / row.NIR_mean_W50_reference_px,
              'Robustness ratio')
        subset = None
        if row.analysis == 'Gaussian sigma = 0.75 ref px':
            subset = measurements
        elif row.analysis.startswith('Leave parent '):
            parent = row.analysis[len('Leave parent '):-len(' out')]
            keep = transects.loc[transects.parent_ID.ne(parent), 'ROI_ID']
            subset = measurements.loc[measurements.ROI_ID.isin(keep)]
        if subset is not None:
            expected_uv, expected_nir = means(subset)
            equal(row.UV_mean_W50_reference_px, expected_uv, 'Robustness UV mean')
            equal(row.NIR_mean_W50_reference_px, expected_nir, 'Robustness NIR mean')


def main():
    checks = {}
    for line in (ROOT / 'provenance' / 'input_checksums.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        checks[relative] = sha(ROOT / relative) == digest
    if not checks or not all(checks.values()):
        raise RuntimeError(f'Input hashes failed: {checks}')
    registration = json.loads((T / 'registration_reconstruction.json').read_text(encoding='utf-8'))
    if (not registration.get('overall_pass') or not registration.get('checks')
            or not all(registration['checks'].values())):
        raise RuntimeError('Registration reconstruction/validation failed')
    validate_outputs(
        pd.read_csv(ROOT / 'config' / 'transects.csv'),
        pd.read_csv(T / 'measurements.csv'),
        pd.read_csv(T / 'summary.csv'),
        pd.read_csv(T / 'robustness_summary.csv'),
        json.loads((T / 'analysis_manifest.json').read_text(encoding='utf-8')),
    )
    print('Input hashes: PASS\nRegistration reconstruction: PASS\nRegistration quality controls: PASS\n'
          'Fixed transects: 5\nValid measurements: 10/10\nNumerical validation: PASS\n'
          'Measurement/summary consistency: PASS\nRobustness outputs: PASS\n\nReproduction verified.')


if __name__ == '__main__':
    main()
