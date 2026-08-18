"""Recompute the released smoothing and leave-one-parent-ROI-out summaries."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent)); import measurement_core as core
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; CFG=ROOT/'config'; OUT=ROOT/'results'/'tables'
def img(p): return np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)
def main():
 r=json.loads((CFG/'registration.json').read_text()); b={**r['uv_mapping'],'common_grid_width_px':r['common_grid_width_px'],'common_grid_height_px':r['common_grid_height_px']}; q=r['nir_similarity_common_to_cropped_native']; t=core.Similarity(q['scale'],q['rotation_deg'],q['tx'],q['ty']); rois=pd.read_csv(CFG/'transects.csv'); us=1/math.sqrt(b['resize_scale_x']*b['resize_scale_y']); um=core.rgb_channel_maps(img(DATA/'UV_248T.tif'),8*us)['Lab_a']; nm=core.rgb_channel_maps(img(DATA/'NIR_highres_cropped_to_UV_FOV.tif'),8*t.scale)['Lab_a']; d=np.arange(-core.PROFILE_HALF_LENGTH_REF,core.PROFILE_HALF_LENGTH_REF+core.PROFILE_STEP_REF/2,core.PROFILE_STEP_REF); rows=[]
 def measure(label, subset, sigma):
  rr,_,_=core.measure_rois(subset.to_dict('records'),um,nm,'high',d,t,b,sigma_ref=sigma,half_level=.5,filter_method='gaussian',channel='Lab_a'); z=pd.DataFrame(rr); uv=float(z[z.condition.eq('UV')].FWHM.mean()); ni=float(z[z.condition.eq('NIR')].FWHM.mean()); rows.append({'analysis':label,'paired_transects':len(subset),'UV_mean_W50_reference_px':uv,'NIR_mean_W50_reference_px':ni,'Gx':uv/ni})
 for sigma in [0,.5,.75,1.0]: measure(f'Gaussian sigma = {sigma:g} ref px',rois,sigma)
 for parent in sorted(rois.parent_ID.unique()): measure(f'Leave parent {parent} out',rois[rois.parent_ID.ne(parent)],.75)
 out=pd.DataFrame(rows); out.to_csv(OUT/'robustness_summary.csv',index=False); print(out.to_string(index=False))
if __name__=='__main__': main()
