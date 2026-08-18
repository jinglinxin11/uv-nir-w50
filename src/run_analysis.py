"""Recompute the frozen five-transect Lab-a apparent-W50 result from two inputs."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_core as core

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'results'/'tables'; CFG=ROOT/'config'
def load(path): return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg=json.loads((CFG/'registration.json').read_text(encoding='utf-8'))
    t=reg['nir_similarity_common_to_cropped_native']; transform=core.Similarity(t['scale'],t['rotation_deg'],t['tx'],t['ty'])
    baseline={**reg['uv_mapping'], 'common_grid_width_px': reg['common_grid_width_px'], 'common_grid_height_px': reg['common_grid_height_px']}
    uv=load(DATA/'UV_248T.tif'); nir=load(DATA/'NIR_highres_cropped_to_UV_FOV.tif'); rois=pd.read_csv(CFG/'transects.csv').to_dict('records')
    uv_scale=1/math.sqrt(baseline['resize_scale_x']*baseline['resize_scale_y'])
    uvmap=core.rgb_channel_maps(uv,8*uv_scale)['Lab_a']; nirmap=core.rgb_channel_maps(nir,8*transform.scale)['Lab_a']
    d=np.arange(-core.PROFILE_HALF_LENGTH_REF,core.PROFILE_HALF_LENGTH_REF+core.PROFILE_STEP_REF/2,core.PROFILE_STEP_REF)
    core.MIN_CNR=3.0
    rows, profiles, _=core.measure_rois(rois,uvmap,nirmap,'high',d,transform,baseline,sigma_ref=.75,half_level=.5,filter_method='gaussian',channel='Lab_a')
    m=pd.DataFrame(rows); p=pd.DataFrame(profiles)
    if len(m)!=10 or not m.valid.all(): raise RuntimeError('Expected 10 valid condition-by-transect measurements')
    uvmean=float(m.loc[m.condition.eq('UV'),'FWHM'].mean()); nirmean=float(m.loc[m.condition.eq('NIR'),'FWHM'].mean())
    summary={'UV_mean_W50_reference_px':uvmean,'NIR_mean_W50_reference_px':nirmean,'Gx_ratio_of_means':uvmean/nirmean,'reduction_percent':100*(1-nirmean/uvmean),'technical_transects':5,'parent_ROIs':4,'measurement_channel':'CIE Lab a*'}
    m.to_csv(OUT/'measurements.csv',index=False); p.to_csv(OUT/'profiles.csv',index=False); pd.DataFrame([summary]).to_csv(OUT/'summary.csv',index=False)
    (OUT/'analysis_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
