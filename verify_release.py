"""Verify released inputs, primary results, and robustness outputs after reproduction."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent; T=ROOT/'results'/'tables'; ABS_TOL=1e-10
EXPECTED={'UV_mean_W50_reference_px':14.526104873392228,'NIR_mean_W50_reference_px':6.050997585336955,'Gx_ratio_of_means':2.400613232534174,'reduction_percent':58.34397701189191}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks={}
 for line in (ROOT/'provenance'/'input_checksums.sha256').read_text().splitlines():
  h,rel=line.split('  ',1); checks[rel]=sha(ROOT/rel)==h
 if not all(checks.values()): raise RuntimeError(f'Input hashes failed: {checks}')
 tr=pd.read_csv(ROOT/'config'/'transects.csv'); m=pd.read_csv(T/'measurements.csv'); s=pd.read_csv(T/'summary.csv').iloc[0]; rb=pd.read_csv(T/'robustness_summary.csv')
 if len(tr)!=5 or tr.parent_ID.nunique()!=4: raise RuntimeError('Fixed transect check failed')
 if len(m)!=10 or not m.valid.all() or not (m.peak_to_noise>=3).all(): raise RuntimeError('Measurement validity check failed')
 for k,v in EXPECTED.items():
  if not np.isclose(float(s[k]),v,rtol=0,atol=ABS_TOL): raise RuntimeError(f'{k} mismatch: {s[k]} vs {v}')
 if len(rb)!=8: raise RuntimeError('Robustness summary must contain 4 smoothing and 4 leave-one-parent rows')
 print('Input hashes: PASS\nFixed transects: 5\nValid measurements: 10/10\nNumerical validation: PASS\nRobustness outputs: PASS\n\nReproduction verified.')
if __name__=='__main__': main()
