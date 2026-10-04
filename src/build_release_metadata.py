"""Verify frozen input-only geometry and immutable release checksums."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROV=ROOT/'provenance'; TRAN=ROOT/'config'/'transects.csv'
EXPECTED=PROV/'input_checksums.sha256'; OBSERVED=ROOT/'results'/'tables'/'input_checksums_observed.sha256'
# Fields required to recreate the frozen transect geometry.  `candidate_ID` is a
# geometry/provenance identifier consumed by the measurement routine; no
# condition-specific outcome is retained here.
GEOMETRY=['ROI_ID','parent_ID','candidate_ID','center_x','center_y','tangent_angle_deg','normal_angle_deg','tangent_offset_ref_px','direction','selected_ID']
def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main():
 frozen=pd.read_csv(TRAN)
 if list(frozen.columns)!=GEOMETRY: raise RuntimeError('Transect input must contain only the frozen geometry schema')
 if len(frozen)!=5 or frozen.ROI_ID.nunique()!=5 or frozen.parent_ID.nunique()!=4: raise RuntimeError('Expected five fixed transects in four parent ROIs')
 if frozen.isna().any().any(): raise RuntimeError('Frozen transect geometry contains missing values')
 observed=[]; failed={}
 for line in EXPECTED.read_text(encoding='utf-8').splitlines():
  expected_hash,rel=line.split('  ',1); actual=sha(ROOT/rel); observed.append(f'{actual}  {rel}\n'); failed[rel]=actual!=expected_hash
 OBSERVED.parent.mkdir(parents=True,exist_ok=True); OBSERVED.write_text(''.join(observed),encoding='utf-8')
 if any(failed.values()): raise RuntimeError(f'Immutable input checksum failure: {failed}')
 print(f'Frozen geometry: PASS\nImmutable input checksums: PASS ({len(observed)} files)')
if __name__=='__main__': main()
