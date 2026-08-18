"""Generate released standalone supplementary and main-text candidate figures.

All quantitative curves use the released measurements/profiles. RGB and local
Lab-a maps are display aids; W50 and Gx remain from native-grid sampling.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; CFG=ROOT/'config'; TAB=ROOT/'results'/'tables'; OUT=ROOT/'results'/'figures'
sys.path.insert(0,str(Path(__file__).resolve().parent)); import measurement_core as core
UV='#D55E00'; NIR='#0072B2'; CYAN='#009ECA'; GRID='#D9D9D9'
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True); fig.savefig(OUT/f'{name}.png',dpi=450,bbox_inches='tight',facecolor='white'); plt.close(fig)
def rgb(path): return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)
def setup(): plt.rcParams.update({'font.family':'Arial','font.size':9})
def transform():
    r=json.loads((CFG/'registration.json').read_text()); t=r['nir_similarity_common_to_cropped_native']; base={**r['uv_mapping'],'common_grid_width_px':r['common_grid_width_px'],'common_grid_height_px':r['common_grid_height_px']}; return r,base,core.Similarity(t['scale'],t['rotation_deg'],t['tx'],t['ty'])
def uv_common(im,base):
    h,w=base['common_grid_height_px'],base['common_grid_width_px']; yy,xx=np.mgrid[:h,:w].astype(float); ux,uy=core.common_to_uv_native(xx,yy,base,im.shape[:2]); return np.stack([core.sample_array(im[...,c],ux,uy) for c in range(3)],-1)
def draw_rois(ax,rois):
    for i,r in rois.iterrows():
      a=np.deg2rad(r.normal_angle_deg); x,y=r.center_x,r.center_y; dx,dy=8*np.cos(a),8*np.sin(a); ax.plot([x-dx,x+dx],[y-dy,y+dy],c=CYAN,lw=1.8); ax.plot(x,y,'o',mfc='white',mec=CYAN,mew=1.2); ax.text(x+4,y-5,f'X{i+1:02d}',c=CYAN,fontsize=7,weight='bold')
def main():
  setup(); _,base,t=transform(); rois=pd.read_csv(CFG/'transects.csv'); m=pd.read_csv(TAB/'measurements.csv'); p=pd.read_csv(TAB/'profiles.csv'); s=pd.read_csv(TAB/'summary.csv').iloc[0]
  uv=uv_common(rgb(DATA/'UV_248T.tif'),base); nir=rgb(DATA/'NIR_976-T_display.tif')
  # RGB and matching local Lab-a display panels
  for name,img,title in [('Fig_Main_UV_RGB_ROIs',uv,'248 nm UV'),('Fig_Main_NIR_RGB_ROIs',nir,'976 nm NIR')]:
    fig,ax=plt.subplots(figsize=(4,5)); ax.imshow(np.clip(img,0,255).astype('uint8')); draw_rois(ax,rois); ax.set_title(title); ax.axis('off'); save(fig,name)
  maps=[]
  for im in [uv,nir]:
    a=core.rgb_channel_maps(im,8)['Lab_a']; maps.append(a-ndimage.gaussian_filter(a,20,mode='nearest'))
  lim=float(np.percentile(np.concatenate([np.abs(x).ravel() for x in maps]),99)); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
  for name,mp,title in [('Fig_Main_UV_local_Lab_a',maps[0],'UV local Lab a*'),('Fig_Main_NIR_local_Lab_a',maps[1],'NIR local Lab a*')]:
    fig,ax=plt.subplots(figsize=(4.3,5)); im=ax.imshow(mp,cmap='RdBu_r',norm=norm); ax.set_title(title); ax.axis('off'); cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.set_label('Local Δa*'); save(fig,name)
  # Center-aligned mean profile
  fig,ax=plt.subplots(figsize=(5.3,4))
  for cond,col,label in [('UV',UV,'UV (248 nm)'),('NIR',NIR,'NIR (976 nm)')]:
    arr=[]
    for _,g in p[p.condition.eq(cond)].groupby('ROI_ID'):
      g=g.sort_values('distance_reference_px'); x=g.distance_reference_px.to_numpy(float); y=g.normalized.to_numpy(float); x=x-x[np.argmax(y)]; common=np.arange(-16,16.001,.2); arr.append(np.interp(common,x,y))
    arr=np.array(arr); mean=arr.mean(0); sd=arr.std(0,ddof=1); ax.fill_between(common,np.maximum(0,mean-sd),np.minimum(1.15,mean+sd),color=col,alpha=.14); ax.plot(common,mean,c=col,lw=2,label=label)
  ax.axhline(.5,c='0.45',ls='--'); ax.set(xlabel='Peak-centered distance (reference px)',ylabel='Normalized ridge signal',title='Center-aligned writing profiles',xlim=(-16,16),ylim=(0,1.15)); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color=GRID); save(fig,'Fig_Main_center_aligned_profiles')
  # S12: recalculate smoothing sensitivity with fixed transects
  d=np.arange(-core.PROFILE_HALF_LENGTH_REF,core.PROFILE_HALF_LENGTH_REF+core.PROFILE_STEP_REF/2,core.PROFILE_STEP_REF); uvraw=rgb(DATA/'UV_248T.tif'); nirraw=rgb(DATA/'NIR_highres_cropped_to_UV_FOV.tif'); us=1/math.sqrt(base['resize_scale_x']*base['resize_scale_y']); umap=core.rgb_channel_maps(uvraw,8*us)['Lab_a']; nmap=core.rgb_channel_maps(nirraw,8*t.scale)['Lab_a']; rows=[]
  for sig in [0,.5,.75,1.0]:
    rr,_,_=core.measure_rois(rois.to_dict('records'),umap,nmap,'high',d,t,base,sigma_ref=sig,half_level=.5,filter_method='gaussian',channel='Lab_a')
    for z in rr: rows.append({'sigma_ref_px':sig,**z})
  sens=pd.DataFrame(rows); sens.to_csv(TAB/'smoothing_sensitivity.csv',index=False)
  # S11: primary ten measurements and all 40 smoothing-setting measurements.
  fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3.7),sharey=True); order=list(rois.ROI_ID)
  for cond,col,off in [('UV',UV,-.12),('NIR',NIR,.12)]:
    q=m[m.condition.eq(cond)].set_index('ROI_ID').loc[order]; x=np.arange(1,6)+off; ax1.scatter(x,q.peak_to_noise,c=col,label=cond,s=36); ax1.plot(x,q.peak_to_noise,c=col,alpha=.5)
  for cond,col in [('UV',UV),('NIR',NIR)]:
    for i,roi in enumerate(order,1):
      q=sens[(sens.condition.eq(cond))&(sens.ROI_ID.eq(roi))].sort_values('sigma_ref_px'); ax2.plot(q.sigma_ref_px,q.peak_to_noise,c=col,alpha=.38); ax2.scatter(q.sigma_ref_px,q.peak_to_noise,c=col,s=12)
  for ax,title in [(ax1,'(a) Primary measurements'),(ax2,'(b) Smoothing settings')]:
    ax.axhline(3,c='0.35',ls='--'); ax.set_title(title); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color=GRID)
  ax1.set(xticks=np.arange(1,6),xticklabels=[f'X{i:02d}' for i in range(1,6)],ylabel='Peak-to-noise ratio'); ax1.legend(frameon=False,ncol=2,fontsize=8); ax2.set(xlabel='Gaussian σ (reference px)'); fig.suptitle('Figure S11. PNR validity checks',y=1.02); fig.tight_layout(); save(fig,'Figure_S11_PNR_validity')
  baseline=sens[sens.sigma_ref_px.eq(.75)].set_index(['condition','ROI_ID']).FWHM
  for cond,col in [('UV',UV),('NIR',NIR)]:
    fig,ax=plt.subplots(figsize=(5.1,3.7))
    for i,roi in enumerate(order,1):
      q=sens[(sens.condition.eq(cond))&(sens.ROI_ID.eq(roi))].sort_values('sigma_ref_px'); y=[100*(v/baseline[(cond,roi)]-1) for v in q.FWHM]; ax.plot(q.sigma_ref_px,y,marker='o',label=f'X{i:02d}')
    ax.axvline(.75,c='0.35',ls='--'); ax.axhline(0,c='0.55',lw=.8); ax.set(xlabel='Gaussian σ (reference px)',ylabel='Change from σ = 0.75 (%)',title=f'Figure S12. {cond} smoothing sensitivity'); ax.legend(frameon=False,ncol=3,fontsize=8); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',color=GRID); save(fig,f'Figure_S12_smoothing_{cond}')
  summary=sens.groupby(['sigma_ref_px','condition']).FWHM.mean().unstack(); summary['Gx']=summary.UV/summary.NIR; summary.to_csv(TAB/'smoothing_summary.csv')
if __name__=='__main__': main()
