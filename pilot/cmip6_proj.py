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
WL=[1.5,2.0,2.8]; rows=[]; peryr={}
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
P=pd.DataFrame(rows); P.to_csv('pilot/cmip6_proj.csv',index=False)
pd.to_pickle(peryr,'pilot/cmip6_peryear.pkl'); pd.to_pickle(peryear(obs),'pilot/obs_peryear_2005.pkl')
print(P.groupby('wl')[['lmic','hic','bT_med','bH_med']].describe(percentiles=[.1,.5,.9]).round(2).T)
# ---- combined uncertainty: model x year-block bootstrap
comb={}
for wl in WL:
    vals=[]
    for b in range(400):
        m=rng.choice(P.model.unique()); ys=rng.choice(range(2005,2025),20); vals.append(peryr[(m,wl)].loc[ys].mean())
    comb[wl]=np.percentile(vals,[5,50,95])
po=peryear(obs); ob=[po.loc[rng.choice(range(2005,2025),20)].mean() for _ in range(400)]
print('combined 5-50-95 LMIC:',{k:v.round(0) for k,v in comb.items()},' obs',np.percentile(ob,[5,50,95]).round(0))
pd.DataFrame(comb,index=['p5','p50','p95']).to_csv('pilot/cmip6_combined.csv')
# ---- attribution with hist-nat
att=[]
for m in models:
    h=load(m,'historical'); n=load(m,'hist-nat')
    if n is None: continue
    sh=(h['yr']>=1995)&(h['yr']<=2014); sn=(n['yr']>=1995)&(n['yr']<=2014)
    if sn.sum()<12*20: print('hist-nat short',m,n['yr'].max()); 
    dT=np.zeros((12,len(cells)),'float32'); dH=np.zeros((12,len(cells)),'float32')
    for k in range(1,13):
        dT[k-1]=h['tasmax'][sh&(h['mo']==k)].mean(0)-n['tasmax'][sn&(n['mo']==k)].mean(0)
        dH[k-1]=h['hurs'][sh&(h['mo']==k)].mean(0)-n['hurs'][sn&(n['mo']==k)].mean(0)
    dG=pd.Series(h['gmst'][sh]).mean()-pd.Series(n['gmst'][sn]).mean()
    gsm=gpi.rolling(11,center=True,min_periods=5).mean(); fac=(gsm.loc[A.season.values].values/gsm.loc[1995:2014].mean())[:,None]
    for var in ['T+RH','T only']:
        cfT=T-dT[mo-1,ci][:,None]*fac; cfH=np.clip(RH-(dH[mo-1,ci][:,None]*fac if var=='T+RH' else 0),1,100)
        hi=hi_c(cfT,cfH); u=uniq(hi>=40.6); d35f=(HI>=35).sum(1); d35c=(hi>=35).sum(1)
        L=A.lmic.values
        att.append(dict(model=m,var=var,dGMST=dG,dT_med=float(np.median(dT[mo-1,ci])),dH_med=float(np.median(dH[mo-1,ci])),lmic_fact=tot(obs),lmic_cf=tot(u),hic_fact=tot(obs,False),hic_cf=tot(u,False),d35_fact=d35f[L].mean(),d35_cf=d35c[L].mean()))
AT=pd.DataFrame(att); AT['frac_attr']=1-AT.lmic_cf/AT.lmic_fact; AT['d35_frac']=1-AT.d35_cf/AT.d35_fact
pd.set_option('display.width',250); print(AT.round(3).to_string()); AT.to_csv('pilot/attribution_single_member.csv',index=False)
