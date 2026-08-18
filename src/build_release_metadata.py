"""Derive input-only transect configuration and release provenance records."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROV=ROOT/'provenance'; FULL=PROV/'ROI_selection_records.csv'; TRAN=ROOT/'config'/'transects.csv'
# Fields required to recreate the frozen transect geometry.  `candidate_ID` is a
# geometry/provenance identifier consumed by the measurement routine; no
# condition-specific outcome is retained here.
GEOMETRY=['ROI_ID','parent_ID','candidate_ID','center_x','center_y','tangent_angle_deg','normal_angle_deg','tangent_offset_ref_px','direction','selected_ID']
def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main():
 PROV.mkdir(exist_ok=True)
 full=pd.read_csv(FULL); full[GEOMETRY].to_csv(TRAN,index=False)
 checks=[ROOT/'data'/'UV_248T.tif',ROOT/'data'/'NIR_highres_unsharpened_original.jpg',TRAN,ROOT/'config'/'registration.json',FULL]
 (PROV/'input_checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in checks),encoding='utf-8')
if __name__=='__main__': main()
