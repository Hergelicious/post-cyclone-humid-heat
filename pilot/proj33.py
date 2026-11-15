import pandas as pd, numpy as np, glob, os
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
rng=np.random.default_rng(3); rp406=0.953
cells=np.load('data/cmip6/cells4276.npy'); cidx={(round(a,1),round(b,1)):i for i,(a,b) in enumerate(cells)}
def load(m,e):
    f=sorted(glob.glob(f'data/cmip6/{m}_{e}_*.npz'))
    if not f: return None
    if e=='historical':  # use the realisation that continues into SSP5-8.5
        s5=sorted(glob.glob(f'data/cmip6/{m}_ssp585_*.npz'))
        if s5:
            mem=os.path.basename(s5[0]).split('_')[2]; same=[x for x in f if os.path.basename(x).split('_')[2]==mem]
            if same: f=same
    d=dict(np.load(f[0])); n=min(len(d['gmst']),len(d['tasmax']),len(d['hurs']),len(d['yr']))
    return {k:v[:n] for k,v in d.items()}
models=sorted(set(os.path.basename(f).split('_')[0] for f in glob.glob('data/cmip6/*_ssp585_*.npz')))
OBS_REF=0.85  # warming 1850-1900 to 1995-2014 (IPCC AR6)
# ---- records 2005-2024, 100 km
R=pd.read_pickle('pilot/recon_records.pkl'); A=R[(R.foot=='100')&(R.season>=2005)].reset_index(drop=True)
N=len(A); T=np.zeros((N,30),'float32'); RH=np.zeros((N,30),'float32'); HI=np.zeros((N,30),'float32')
for i,row in enumerate(A.itertuples()):
    s=series(*row.cell); w=slice(row.t0+1,row.t0+31); T[i]=s['T'][w]; RH[i]=s['RH'][w]; HI[i]=s['HI'][w]
ci=np.array([cidx[c] for c in A.cell]); mo=mon[A.t0.values]; g_obs=gpi.loc[A.season.values].values
# unique cell-day key
dayidx=A.t0.values[:,None]+1+np.arange(30)[None]; ckey=pd.factorize(pd.Series([f'{a}_{b}' for a,b in A.cell]))[0]
key=(ckey[:,None].astype(np.int64)*20000+dayidx).ravel(); ukey,inv=np.unique(key,return_inverse=True)
popw=np.repeat(A['pop'].values[:,None],30,1).ravel(); lm=np.repeat(A.lmic.values[:,None],30,1).ravel(); yrs=np.repeat(A.season.values[:,None],30,1).ravel()
def uniq(mask):
    v=np.where(mask.ravel(),popw*rp406,0.0); best=np.zeros(len(ukey)); np.maximum.at(best,inv,v)
    # attribute each unique key to LMIC flag/year of first element
    first=np.zeros(len(ukey),int); first[inv[::-1]]=np.arange(len(inv))[::-1]
    return pd.DataFrame(dict(v=best,lmic=lm[first],year=yrs[first]))
def tot(df,lmic=True): return df[df.lmic==lmic].v.sum()/20/1e6
def peryear(df,lmic=True): return df[df.lmic==lmic].groupby('year').v.sum().reindex(range(2005,2025),fill_value=0)/1e6
obs=uniq(HI>=40.6); print('observed 2005-2024 LMIC',round(tot(obs),1),'HIC',round(tot(obs,False),1))
# ---- per model pattern scaling
WL=[1.5,2.8,3.3]; rows=[]; peryr={}
for m in models:
    h=load(m,'historical'); f=load(m,'ssp585')
    if h is None or f is None: continue
    yr_=np.concatenate([h['yr'],f['yr']]); mo_=np.concatenate([h['mo'],f['mo']])
    gm=np.concatenate([h['gmst'],f['gmst']]); tx=np.concatenate([h['tasmax'],f['tasmax']]); hu=np.concatenate([h['hurs'],f['hurs']])
    gy=pd.Series(gm).groupby(yr_).mean(); gy=gy.rolling(11,center=True,min_periods=5).mean(); gref=gy.loc[1995:2014].mean(); ganom=gy-gref+OBS_REF
    bT=np.zeros((12,len(cells)),'float32'); bH=np.zeros((12,len(cells)),'float32')
    for k in range(1,13):
        sel=mo_==k; x=ganom.loc[yr_[sel]].values; X=np.c_[x-x.mean()]
        bT[k-1]=(X[:,0]@(tx[sel]-tx[sel].mean(0)))/(X[:,0]@X[:,0]); bH[k-1]=(X[:,0]@(hu[sel]-hu[sel].mean(0)))/(X[:,0]@X[:,0])
    ecs_like=ganom.loc[2081:2100].mean()
    for wl in WL:
        dg=(wl-g_obs)[:,None]; dT=bT[mo-1,ci][:,None]*dg; dH=bH[mo-1,ci][:,None]*dg
        hi=hi_c(T+dT,np.clip(RH+dH,1,100)); u=uniq(hi>=40.6)
        rows.append(dict(model=m,wl=wl,lmic=tot(u),hic=tot(u,False),bT_med=float(np.median(bT[mo-1,ci])),bH_med=float(np.median(bH[mo-1,ci])),g2100=ecs_like))
        peryr[(m,wl)]=peryear(u)
    print(m,[round(r['lmic'],0) for r in rows if r['model']==m],flush=True)
P=pd.DataFrame(rows); P.to_csv('pilot/cmip6_proj33.csv',index=False)
print(P.groupby('wl').lmic.describe(percentiles=[.1,.5,.9]).round(0))
pv=P.pivot(index='model',columns='wl',values='lmic'); print('ratio 3.3/2.8 median',round((pv[3.3]/pv[2.8]).median(),3),'range',round((pv[3.3]/pv[2.8]).min(),3),round((pv[3.3]/pv[2.8]).max(),3))
print('ratio 3.3/1.5 median',round((pv[3.3]/pv[1.5]).median(),2))
comb={}
for wl in WL:
    vals=[]
    for b in range(400):
        m=rng.choice(P.model.unique()); ys=rng.choice(range(2005,2025),20); vals.append(peryr[(m,wl)].loc[ys].mean())
    comb[wl]=np.percentile(vals,[5,50,95])
print('combined',{k:v.round(0) for k,v in comb.items()})
pd.DataFrame(comb,index=['p5','p50','p95']).to_csv('pilot/cmip6_combined33.csv')
