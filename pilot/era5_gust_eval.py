# Compare LMIC exposure in the 100 km circle with exposure in the part of it that ERA5 10 m gusts mark as storm-affected
# (maximum gust from 6 h before to 18 h after landfall >= 17.5, 20 or 25 m/s), and report how much of the gust footprint
# population lies within 100 km.
import numpy as np, pandas as pd
G=pd.read_parquet('pilot/era5_gust_cells.parquet')
R=pd.read_pickle('pilot/era5_records_2024.pkl'); Z=np.load('pilot/era5_dmax_2024.npz'); HImax=Z['HImax']
R=R.reset_index(drop=True); R['d406']=(HImax>=40.6).sum(1); R['cla']=[c[0] for c in R.cell]; R['clo']=[c[1] for c in R.cell]
# gust cells are centred on x.25/x.75; a record cell (x.0/x.5) takes the mean of the four gust cells that overlap it
gd={(i,a,b):g for i,a,b,g in zip(G.idx,G.cla,G.clo,G.g)}
def gcell(i,a,b):
    v=[gd.get((i,a+da,b+db)) for da in (-0.25,0.25) for db in (-0.25,0.25)]; v=[x for x in v if x is not None]
    return np.mean(v) if v else np.nan
R['gust']=[gcell(i,a,b) for i,a,b in zip(R.idx,R.cla,R.clo)]
rc=open('pilot/reconcile.py').read(); _ns={'np':np,'pd':pd}; exec(rc[rc.index('# population 0.5 deg'):rc.index('def series')],_ns); pop_cell=_ns['pop_cell']
per={'1981-2000':(1981,2000),'2001-2017':(2001,2017),'2018-2024':(2018,2024)}
V=R[R.valid&R.lmic].copy(); V['pd']=V['pop']*V.d406
print('cells with gust value',round(100*V.gust.notna().mean(),1),'%')
for thr in (None,17.5,20,25):
    X=V if thr is None else V[V.gust>=thr]
    vals=[X[(X.season>=a)&(X.season<=b)].pd.sum()/(b-a+1)/1e6 for a,b in per.values()]
    print(f'{"all cells in 100 km" if thr is None else f"gust >= {thr} m/s":22s} LMIC person-days HI>=40.6 (summed, M/yr): '+', '.join(f'{v:.0f}' for v in vals)+f'; ratio late/early {vals[2]/vals[0]:.2f}; share of 100 km total {100*X.pd.sum()/V.pd.sum():.0f}%')
# population of the gust footprint within and beyond 100 km (2018-2024 LMIC landfalls)
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
inc=set(R.idx[R.lmic]); G2=G[G.idx.isin(inc)].merge(L[['season','lat','lon']].rename(columns={'lat':'la0','lon':'lo0'}),left_on='idx',right_index=True)
G2['dist']=np.hypot(G2.cla-G2.la0,(G2.clo-G2.lo0)*np.cos(np.radians(G2.la0)))*111
for thr in (17.5,20,25):
    X=G2[(G2.g>=thr)&(G2.season>=2005)]; p=np.array([pop_cell(a,b,y) for a,b,y in zip(X.cla,X.clo,X.season)])
    print(f'gust >= {thr}: footprint population within 100 km {100*p[X.dist.values<=100].sum()/p.sum():.0f}% ; median footprint radius (km, population-weighted) {np.median(np.repeat(X.dist.values,np.maximum(1,(p/1e4).astype(int)))):.0f}')
