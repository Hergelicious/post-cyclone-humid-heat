import pandas as pd, numpy as np, json, itertools
exec(open('pilot/spatial_exposure.py').read().split('foot={')[0])   # reuse loaders: lf, series, pop_cell, cells_within, dates
rng=np.random.default_rng(5)
yrs={'E':range(1981,2001),'L':range(2018,2025)}
def clim_days(cell,t0,years,thr):
    s=series(*cell); m=s[0] if thr=='35' else s[1]; d=dates[t0]; out=[]
    for y in years:
        try: tt=np.searchsorted(dates.values,pd.Timestamp(year=y,month=d.month,day=min(d.day,28)).to_datetime64())
        except Exception: continue
        out.append(m[tt+1:tt+31].sum())
    return np.mean(out)
def popfix(cell,epoch): return pop_cell(*cell, 1990 if epoch=='E' else 2021)
# precompute per landfall (100 km) the needed quantities
rec=[]
for r in lf[lf.lmic].itertuples():
    t0=np.searchsorted(dates.values,r.time.normalize().to_datetime64()); S='E' if r.season<=2000 else ('L' if r.season>=2018 else None)
    if S is None: continue
    for cell in cells_within(r.lat,r.lon,100):
        if series(*cell) is None: continue
        d=dict(idx=r.index,S=S,cell=cell)
        for H in ('E','L'):
            for thr in ('35','406'): d[f'h{thr}_{H}']=clim_days(cell,t0,yrs[H],thr)
        for Pp in ('E','L'): d[f'p_{Pp}']=popfix(cell,Pp)
        d['obs35']=series(*cell)[0][t0+1:t0+31].sum(); d['obs406']=series(*cell)[1][t0+1:t0+31].sum(); d['p_act']=pop_cell(*cell,r.season)
        rec.append(d)
R=pd.DataFrame(rec); R.to_pickle('pilot/decomp_rec.pkl')
ny={'E':20,'L':7}
def E(S,P,H,thr): 
    s=R[R.S==S]; return (s[f'p_{P}']*s[f'h{thr}_{H}']).sum()/ny[S]/1e6
out={}
for thr in ('35','406'):
    obsE=(R[R.S=='E'].p_act*R[R.S=='E'][f'obs{thr}']).sum()/20/1e6; obsL=(R[R.S=='L'].p_act*R[R.S=='L'][f'obs{thr}']).sum()/7/1e6
    base=E('E','E','E',thr); full=E('L','L','L',thr)
    # Shapley over factors S,P,H
    fac=['S','P','H']; sh={f:0.0 for f in fac}
    for perm in itertools.permutations(fac):
        state={'S':'E','P':'E','H':'E'}; prev=E(**state,thr=thr)
        for f in perm:
            state[f]='L'; cur=E(state['S'],state['P'],state['H'],thr); sh[f]+=(cur-prev)/6; prev=cur
    out[thr]=dict(observed_early=obsE,observed_late=obsL,model_early=base,model_late=full,**{f'shapley_{f}':v for f,v in sh.items()},
                  **{f'share_{f}':v/(full-base) for f,v in sh.items()},
                  climate_only=E('E','E','L',thr),pop_only=E('E','L','E',thr),storms_only=E('L','E','E',thr))
O=pd.DataFrame(out).T; print(O.round(2).to_string()); O.to_csv('pilot/decomposition.csv')
