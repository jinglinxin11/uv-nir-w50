"""Export standalone publication figures; each output is an individual file."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'results'/'figures'; TAB=ROOT/'results'/'tables'; CFG=ROOT/'config'
UV='#D55E00'; NIR='#0072B2'; CYAN='#009ECA'
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.png', dpi=450, bbox_inches='tight', facecolor='white')
    plt.close(fig)
def style(): plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none'})
def main():
    style(); m=pd.read_csv(TAB/'measurements.csv'); p=pd.read_csv(TAB/'profiles.csv'); rois=pd.read_csv(CFG/'M4_5X_ROIs.csv'); s=pd.read_csv(TAB/'summary.csv').iloc[0]
    # Standalone ROI-location image
    img=np.asarray(Image.open(DATA/'NIR_976-T_display.tif').convert('RGB')); fig,ax=plt.subplots(figsize=(5,6)); ax.imshow(img)
    for i,r in rois.iterrows():
      x,y=r.center_x,r.center_y; a=np.deg2rad(r.normal_angle_deg); dx,dy=22*np.cos(a),22*np.sin(a); ax.plot([x-dx,x+dx],[y-dy,y+dy],color=CYAN,lw=2); ax.plot(x,y,'o',mfc='white',mec=CYAN,mew=1.5); ax.text(x+5,y-7,f'X{i+1:02d}',color=CYAN,weight='bold')
    ax.set(title='Registered-image ROI locations'); ax.set_xlabel('Reference-grid X (px)'); ax.set_ylabel('Reference-grid Y (px)'); save(fig,'Fig_S9_ROI_locations')
    # Every technical profile separately
    for i,roi in enumerate(rois.ROI_ID,1):
      fig,ax=plt.subplots(figsize=(5.2,3.6))
      for cond,c in [('UV',UV),('NIR',NIR)]:
        q=p[(p.ROI_ID==roi)&(p.condition==cond)].sort_values('distance_reference_px'); mm=m[(m.ROI_ID==roi)&(m.condition==cond)].iloc[0]; ax.plot(q.distance_reference_px,q.normalized,color=c,lw=2,label=f'{cond}: W50={mm.FWHM:.2f} px'); ax.axvline(mm.x50_left,color=c,ls=':',lw=1); ax.axvline(mm.x50_right,color=c,ls=':',lw=1)
      ax.axhline(.5,color='0.4',ls='--',lw=1); ax.set(xlabel='Distance along profile normal (reference px)',ylabel='Normalized Lab a* intensity',title=f'Transect X{i:02d}'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); save(fig,f'Fig_S9_profile_X{i:02d}')
    # Representative W50 extraction
    roi=rois.ROI_ID.iloc[3]; fig,ax=plt.subplots(figsize=(5.4,3.8))
    for cond,c in [('UV',UV),('NIR',NIR)]:
      q=p[(p.ROI_ID==roi)&(p.condition==cond)].sort_values('distance_reference_px'); mm=m[(m.ROI_ID==roi)&(m.condition==cond)].iloc[0]; ax.plot(q.distance_reference_px,q.normalized,c=c,lw=2,label=f'{cond} W50={mm.FWHM:.2f} px'); ax.annotate('',(mm.x50_left,.45),(mm.x50_right,.45),arrowprops={'arrowstyle':'<->','color':c})
    ax.axhline(.5,color='0.4',ls='--'); ax.set(xlabel='Transverse position (reference px)',ylabel='Normalized Lab a* intensity',title='Representative W50 extraction (X04)'); ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False); save(fig,'Fig_S10_representative_W50_X04')
    # Paired width plot
    fig,ax=plt.subplots(figsize=(5.2,4)); piv=m.pivot(index='ROI_ID',columns='condition',values='FWHM');
    label_offset=[0.70,-0.55,0.85,0.15,0.60]
    for i,(idx,row) in enumerate(piv.iterrows(),1):
      ax.plot([0,1],[row.UV,row.NIR],color='0.65')
      ax.scatter([0,1],[row.UV,row.NIR],c=[UV,NIR],s=35,zorder=3)
      ax.text(.50,(row.UV+row.NIR)/2+label_offset[i-1],f'X{i:02d}',ha='center',va='center',fontsize=8,
              bbox={'boxstyle':'round,pad=0.08','fc':'white','ec':'none','alpha':0.82})
    ax.hlines(s.UV_mean_W50_reference_px,-.12,.12,color='k',lw=3); ax.hlines(s.NIR_mean_W50_reference_px,.88,1.12,color='k',lw=3); ax.set(xlim=(-.3,1.3),xticks=[0,1],xticklabels=['248 nm UV','976 nm NIR'],ylabel='Apparent W50 (reference px)',title='All five paired technical transects'); ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.25); save(fig,'Fig_Main_paired_W50')
    # Gx
    fig,ax=plt.subplots(figsize=(4.5,3.8)); ax.bar([0,1],[1,s.Gx_ratio_of_means],color=['#FBE9DF','#D8EAF5'],edgecolor=[UV,NIR],linewidth=2); ax.set(xticks=[0,1],xticklabels=['248 nm UV\nreference','976 nm NIR'],ylabel='Relative X-direction width ratio',title='Conditional X-direction confinement proxy'); ax.text(1,s.Gx_ratio_of_means/2,f'{s.Gx_ratio_of_means:.4f}×',ha='center',weight='bold',color=NIR); ax.text(.5,.05,'Ratio of mean apparent W50 values',transform=ax.transAxes,ha='center',fontsize=8); ax.spines[['top','right']].set_visible(False); save(fig,'Fig_Main_Gx_proxy')
if __name__=='__main__': main()
