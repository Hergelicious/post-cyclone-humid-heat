# Station check of ERA5 and NASA POWER at the same 0.5 deg cells: daily maximum wet-bulb temperature (Stull) from hourly ERA5 and
# hourly NASA POWER at the cell nearest each GSDM-WBT station (within 40 km), against the station daily maximum, in the 30 days
# after LMIC landfalls of 2001-2020.
import pandas as pd, numpy as np, xarray as xr, json, os, requests, math
from concurrent.futures import ThreadPoolExecutor
D0=pd.Timestamp('1981-01-01'); NB=44
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
def stull(T,RH): return T*np.arctan(0.151977*np.sqrt(RH+8.313659))+np.arctan(T+RH)-np.arctan(RH-1.676331)+0.00391838*RH**1.5*np.arctan(0.023101*RH)-4.686035
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index(drop=True)
st=pd.read_csv('data/gsdm/stations.csv',dtype={'Code':str}).set_index('Code')
pairs=pd.read_csv('data/gsdm/pairs.csv',dtype={'code':str})
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[(R.foot=='100')&R.lmic&(R.season>=2001)&(R.season<=2020)].reset_index(drop=True)
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); pid={k:p for k,p in zip(pts.key,pts.pid)}
jobs=[]
for p in pairs.itertuples():
    e=lf.iloc[p.lf_idx]
    if e.season<2001 or e.season>2020: continue
    t0=(e.time.normalize()-D0).days; c=R[R.t0==t0]
    if c.empty or p.code not in st.index: continue
    sla,slo=st.loc[p.code,'Latitude'],st.loc[p.code,'Longitude']
    d=[math.hypot(a-sla,(b-slo)*math.cos(math.radians(sla)))*111 for a,b in c.cell]; k=int(np.argmin(d))
    if d[k]>40: continue
    jobs.append(dict(code=p.code,t0=t0,cell=c.cell.iloc[k],km=d[k],iso2=e.iso2))
J=pd.DataFrame(jobs).drop_duplicates(['code','t0']); print('station-landfall pairs',len(J))
def get(r):
    (la,lo),t0=r.cell,int(r.t0); fn=f'data/power_td/{la}_{lo}_{t0}.csv'
    if os.path.exists(fn): return 1
    a=(D0+pd.Timedelta(days=t0-11)).strftime('%Y%m%d'); b=(D0+pd.Timedelta(days=t0+32)).strftime('%Y%m%d')
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=UTC"
    for _ in range(3):
        try:
            t=requests.get(u,timeout=120).text
            if t.startswith('YEAR'): open(fn,'w').write(t); return 1
        except Exception: pass
    return 0
with ThreadPoolExecutor(6) as ex: list(ex.map(get,list(J.itertuples())))
dates=pd.date_range('1981-01-01','2020-12-31'); out=[]
for r in J.itertuples():
    (la,lo),t0=r.cell,int(r.t0); fn=f'data/power_td/{la}_{lo}_{t0}.csv'
    if not os.path.exists(fn): continue
    h=pd.read_csv(fn).replace(-999,np.nan)
    if len(h)<NB*24: continue
    Tn=h.T2M.values[:NB*24]; RHn=np.clip(h.RH2M.values[:NB*24],1,100)
    Te=np.full(NB*24,np.nan); Tde=Te.copy(); p_=pid[f'{la}_{lo}']
    for k in range(NB):
        f=f'data/era5q/{(D0+pd.Timedelta(days=t0-11+k)):%Y%m%d}.npz'
        if not os.path.exists(f): continue
        z=np.load(f); ix=dict(zip(z['pid'],range(len(z['pid']))))
        if p_ in ix: Te[k*24:(k+1)*24]=z['T'][:,ix[p_]]/100; Tde[k*24:(k+1)*24]=z['Td'][:,ix[p_]]/100
    RHe=np.clip(100*es(Tde)/es(Te),1,100); o=int(round(lo/15)); s0=12*24-o
    twe=np.nanmax(stull(Te,RHe)[s0:s0+720].reshape(30,24),1); twn=np.nanmax(stull(Tn,RHn)[s0:s0+720].reshape(30,24),1)
    tme=np.nanmax(Te[s0:s0+720].reshape(30,24),1); tmn=np.nanmax(Tn[s0:s0+720].reshape(30,24),1)
    d=xr.open_dataset(f'data/gsdm/nc/{r.code}.nc'); s=pd.Series(d.WBT.values,index=dates); fl=pd.Series(d.flag.values,index=dates); s[fl==1]=np.nan
    w=s.loc[D0+pd.Timedelta(days=t0+1):D0+pd.Timedelta(days=t0+30)].values
    if len(w)<30: continue
    for k in range(30):
        if np.isfinite(w[k]) and np.isfinite(twe[k]) and np.isfinite(twn[k]): out.append(dict(code=r.code,iso2=r.iso2,t0=t0,tw_st=w[k],tw_e=twe[k],tw_n=twn[k]))
O=pd.DataFrame(out); O.to_csv('pilot/station_arbiter.csv',index=False)
print('station-days',len(O),'stations',O.code.nunique(),'landfalls',O.t0.nunique())
for nm,c in [('ERA5','tw_e'),('NASA POWER','tw_n')]:
    dd=O[c]-O.tw_st; print(f'{nm}: daily max Tw bias vs station {dd.mean():.2f} C (median {dd.median():.2f}), MAE {dd.abs().mean():.2f}, r {np.corrcoef(O[c],O.tw_st)[0,1]:.2f}')
for thr in (26,27,28):
    print(f'days Tw>={thr} per 30: station {(O.tw_st>=thr).mean()*30:.2f}, ERA5 {(O.tw_e>=thr).mean()*30:.2f}, NASA POWER {(O.tw_n>=thr).mean()*30:.2f}')
