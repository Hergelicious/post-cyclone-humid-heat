# County-point CMIP6 fields (same runs and definitions as cmip6_cell_deltas.py): warming-level slopes of monthly Tmax and RH on
# smoothed GMST for 26 models, and scaled human-induced change for the 15 attribution realisations. GMST from the cell files.
import pandas as pd, numpy as np, glob, os
OBS_REF=0.85
def load(m,e,r):
    a=dict(np.load(f'data/cmip6/{m}_{e}_{r}.npz')); b=dict(np.load(f'data/cmip6cty/{m}_{e}_{r}.npz'))
    n=min(len(a['gmst']),len(b['tasmax']),len(b['hurs']),len(a['yr']))
    assert (a['yr'][:n]==b['yr'][:n]).all()
    return dict(gmst=a['gmst'][:n],tasmax=b['tasmax'][:n],hurs=b['hurs'][:n],yr=a['yr'][:n],mo=a['mo'][:n])
S={}
for f in sorted(glob.glob('data/cmip6/*_ssp585_*.npz')):
    m,_,mem=os.path.basename(f)[:-4].split('_')
    if not (os.path.exists(f'data/cmip6cty/{m}_historical_{mem}.npz') and os.path.exists(f'data/cmip6cty/{m}_ssp585_{mem}.npz')): print('missing',m); continue
    h=load(m,'historical',mem); s=load(m,'ssp585',mem)
    yr=np.r_[h['yr'],s['yr']]; mo=np.r_[h['mo'],s['mo']]; gm=np.r_[h['gmst'],s['gmst']]; tx=np.r_[h['tasmax'],s['tasmax']]; hu=np.r_[h['hurs'],s['hurs']]
    gy=pd.Series(gm).groupby(yr).mean().rolling(11,center=True,min_periods=5).mean(); ga=gy-gy.loc[1995:2014].mean()+OBS_REF
    n=tx.shape[1]; bT=np.zeros((12,n),'float32'); bH=np.zeros((12,n),'float32')
    for k in range(1,13):
        sel=mo==k; x=ga.loc[yr[sel]].values; xc=x-x.mean()
        bT[k-1]=(xc@(tx[sel]-tx[sel].mean(0)))/(xc@xc); bH[k-1]=(xc@(hu[sel]-hu[sel].mean(0)))/(xc@xc)
    S[m]=(bT,bH)
np.savez_compressed('data/cmip6cty/slopes26.npz',models=np.array(list(S)),bT=np.stack([v[0] for v in S.values()]),bH=np.stack([v[1] for v in S.values()]))
print('slopes',len(S))
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
old=np.load('data/cmip6/humandelta15.npz'); keys=[]; dT=[]; dH=[]
for key in old['keys']:
    m,r=key.split('|'); h=load(m,'historical',r); n_=load(m,'hist-nat',r)
    sh=(h['yr']>=1995)&(h['yr']<=2014); sn=(n_['yr']>=1995)&(n_['yr']<=2014)
    sc=gsm.loc[1995:2014].mean()/(h['gmst'][sh].mean()-n_['gmst'][sn].mean())
    dT.append(np.stack([h['tasmax'][sh&(h['mo']==k)].mean(0)-n_['tasmax'][sn&(n_['mo']==k)].mean(0) for k in range(1,13)])*sc)
    dH.append(np.stack([h['hurs'][sh&(h['mo']==k)].mean(0)-n_['hurs'][sn&(n_['mo']==k)].mean(0) for k in range(1,13)])*sc); keys.append(key)
np.savez_compressed('data/cmip6cty/humandelta15.npz',keys=np.array(keys),dT=np.stack(dT).astype('float32'),dH=np.stack(dH).astype('float32'))
print('attribution members',len(keys))
