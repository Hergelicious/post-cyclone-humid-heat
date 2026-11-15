# Save per-cell CMIP6 fields for reuse: (a) warming-level slopes of monthly Tmax and RH on smoothed GMST (26 models, SSP5-8.5 with its own historical realisation);
# (b) scaled human-induced change (historical minus hist-nat, 1995-2014, scaled to 0.85 C observed human-induced warming) for the 15 attribution realisations.
import pandas as pd, numpy as np, glob, os
cells=np.load('data/cmip6/cells4276.npy'); OBS_REF=0.85
def load(path):
    d=dict(np.load(path)); n=min(len(d['gmst']),len(d['tasmax']),len(d['hurs']),len(d['yr'])); return {k:v[:n] for k,v in d.items()}
S={}
for f in sorted(glob.glob('data/cmip6/*_ssp585_*.npz')):
    m=os.path.basename(f).split('_')[0]; mem=os.path.basename(f).split('_')[2][:-4]
    hp=f'data/cmip6/{m}_historical_{mem}.npz'
    if not os.path.exists(hp): print('no matching historical',m); continue
    h=load(hp); s=load(f)
    yr=np.r_[h['yr'],s['yr']]; mo=np.r_[h['mo'],s['mo']]; gm=np.r_[h['gmst'],s['gmst']]; tx=np.r_[h['tasmax'],s['tasmax']]; hu=np.r_[h['hurs'],s['hurs']]
    gy=pd.Series(gm).groupby(yr).mean().rolling(11,center=True,min_periods=5).mean(); ga=gy-gy.loc[1995:2014].mean()+OBS_REF
    bT=np.zeros((12,len(cells)),'float32'); bH=np.zeros((12,len(cells)),'float32')
    for k in range(1,13):
        sel=mo==k; x=ga.loc[yr[sel]].values; xc=x-x.mean()
        bT[k-1]=(xc@(tx[sel]-tx[sel].mean(0)))/(xc@xc); bH[k-1]=(xc@(hu[sel]-hu[sel].mean(0)))/(xc@xc)
    S[m]=(bT,bH)
np.savez_compressed('data/cmip6/slopes26.npz',models=np.array(list(S)),bT=np.stack([v[0] for v in S.values()]),bH=np.stack([v[1] for v in S.values()]))
print('slopes',len(S))
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
J=pd.read_csv('pilot/attr_jobs.csv'); keys=[]; dT=[]; dH=[]
for (m,r),_ in J.groupby(['m','r']):
    hp=f'data/cmip6/{m}_historical_{r}.npz'; npth=f'data/cmip6/{m}_hist-nat_{r}.npz'
    if not (os.path.exists(hp) and os.path.exists(npth)): continue
    h=load(hp); n=load(npth); sh=(h['yr']>=1995)&(h['yr']<=2014); sn=(n['yr']>=1995)&(n['yr']<=2014)
    sc=gsm.loc[1995:2014].mean()/(h['gmst'][sh].mean()-n['gmst'][sn].mean())
    dT.append(np.stack([h['tasmax'][sh&(h['mo']==k)].mean(0)-n['tasmax'][sn&(n['mo']==k)].mean(0) for k in range(1,13)])*sc)
    dH.append(np.stack([h['hurs'][sh&(h['mo']==k)].mean(0)-n['hurs'][sn&(n['mo']==k)].mean(0) for k in range(1,13)])*sc); keys.append(f'{m}|{r}')
np.savez_compressed('data/cmip6/humandelta15.npz',keys=np.array(keys),dT=np.stack(dT).astype('float32'),dH=np.stack(dH).astype('float32'))
print('attribution members',len(keys))
