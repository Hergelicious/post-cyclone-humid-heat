import pandas as pd, numpy as np, os
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def es(T): return 6.112*np.exp(17.67*T/(T+243.5))
C=pd.read_csv('pilot/era5_cells1.csv'); off=np.round(((C.lon.values+180)%360-180)/15).astype(int)  # local = UTC + off
D=pd.read_pickle('pilot/era5_decomp_records.pkl'); base=pd.Timestamp('1981-01-01')
need=set()
for r in D.itertuples():
    for j in range(30):
        dt=base+pd.Timedelta(days=int(r.t0)+1+j); need.add((dt.year,dt.month))
os.makedirs('data/era5h/nightloc',exist_ok=True)
def load(y,m):
    a=np.load(f'data/era5h/Tp_{y}{m:02d}.npz'); T=a['T']; P=a['P']; Tw=np.load(f'data/era5h/Tw_{y}{m:02d}.npy'); n=min(len(T),len(Tw)); return T[:n],Tw[:n]
for y,m in sorted(need):
    out=f'data/era5h/nightloc/{y}{m:02d}.npz'
    if os.path.exists(out): continue
    T,Tw=load(y,m); ny,nm=(y,m+1) if m<12 else (y+1,1)
    if os.path.exists(f'data/era5h/Tp_{ny}{nm:02d}.npz'):
        T2,Tw2=load(ny,nm); T=np.concatenate([T,T2[:36]]); Tw=np.concatenate([Tw,Tw2[:36]])
    nd=pd.Timestamp(y,m,1).days_in_month; NC=T.shape[1]
    Tmin=np.full((nd,NC),np.nan,'float32'); Twmin=np.full((nd,NC),np.nan,'float32')
    for o in np.unique(off):
        c=np.where(off==o)[0]
        for d in range(nd):
            s=d*24+18-o; e=s+13   # local 18:00 on day d to 06:00 next day inclusive
            if s<0 or e>len(T): continue
            Tmin[d,c]=T[s:e,c].min(0); Twmin[d,c]=np.minimum(Tw[s:e,c],T[s:e,c]).min(0)
    np.savez(out,Tmin=Tmin,Twmin=Twmin)
print('nights done',flush=True)
cache={}
def get(kind,y,m):
    k=(kind,y,m)
    if k not in cache: cache[k]=dict(np.load(f'data/era5h/{kind}/{y}{m:02d}.npz'))
    return cache[k]
cols={k:[] for k in ['n26','n28','tw24','comp','nvalid']}
for r in D.itertuples():
    c={k:0 for k in cols}
    for j in range(30):
        dt=base+pd.Timedelta(days=int(r.t0)+1+j); o=get('nightloc',dt.year,dt.month); hm=get('night',dt.year,dt.month)['HImax']; i=dt.day-1
        tmin=o['Tmin'][i,r.ci]
        if not np.isfinite(tmin): continue
        c['nvalid']+=1; c['n26']+=tmin>=26; c['n28']+=tmin>=28; c['tw24']+=o['Twmin'][i,r.ci]>=24; c['comp']+=(tmin>=26)&(hm[i,r.ci]>=40.6)
    for k in cols: cols[k].append(c[k])
for k in cols: D[k]=cols[k]
D['day406']=D.e_obs
rows=[]
for k,lab in [('day406','Days with maximum HI ≥ 40.6 °C'),('n26','Nights with minimum temperature ≥ 26 °C'),('n28','Nights with minimum temperature ≥ 28 °C'),('tw24','Nights with minimum wet‑bulb temperature ≥ 24 °C'),('comp','Days with maximum HI ≥ 40.6 °C followed by a night ≥ 26 °C')]:
    e=(D[D.S=='E'].pE*D[D.S=='E'][k]).sum()/20/1e6; l=(D[D.S=='L'].pL*D[D.S=='L'][k]).sum()/4/1e6
    rows.append(dict(metric=lab,early=e,late=l,ratio=l/e,per_cell_early=D[D.S=='E'][k].mean(),per_cell_late=D[D.S=='L'][k].mean()))
O=pd.DataFrame(rows); print(O.round(2).to_string()); O.to_csv('pilot/night_heat_local.csv',index=False)
for S in 'EL': d=D[D.S==S]; print(S,'share of >=40.6 days followed by night >=26',round(d.comp.sum()/max(d.day406.sum(),1),3),'valid nights frac',round(d.nvalid.mean()/30,3))
