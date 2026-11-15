# ERA5 hourly 0.25 deg (ARCO, Google Cloud) point extraction for the ERA5-led rebuild (Nov 2026).
# Targets: (i) every 0.5 deg land cell within 100 km of each 1981-2024 landfall, days -11..+32 around landfall;
#          (ii) every US county / PR municipio point, days -14..+60 around each 2014-2024 US/PR landfall (+180 for Maria).
# Variables: 2 m temperature and 2 m dewpoint every hour; surface pressure at 12 UTC (for specific humidity).
# Output per UTC day: data/era5q/YYYYMMDD.npz with pid (point ids), T and Td (hour x point, int16, 0.01 C) and P (hPa x10, uint16).
import numpy as np, pandas as pd, os, sys, time, urllib.request, numcodecs
from concurrent.futures import ThreadPoolExecutor
B='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'
T0=pd.Timestamp('1900-01-01'); D0=pd.Timestamp('1981-01-01'); OUT='data/era5q'; os.makedirs(OUT,exist_ok=True)
blosc=numcodecs.Blosc()
# ---- points
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[R.foot=='100']
cells=sorted(set(R.cell)); C=pd.read_csv('data/geo/county_points.csv',dtype={'fips':str})
pts=pd.DataFrame([dict(kind='cell',key=f'{a}_{b}',lat=a,lon=b) for a,b in cells]+[dict(kind='county',key=r.fips,lat=r.lat,lon=r.lon) for r in C.itertuples()])
pts['j']=np.round((90-pts.lat)/0.25).astype(int); pts['i']=np.round((pts.lon%360)/0.25).astype(int)%1440
pts['pid']=np.arange(len(pts)); pts.to_csv(f'{OUT}/points.csv',index=False)
flat=(pts.j*1440+pts.i).values
cid={k:p for k,p in zip(pts.key,pts.pid)}
# ---- days needed -> point ids
need={}
for c,t0 in zip(R.cell,R.t0):
    p=cid[f'{c[0]}_{c[1]}']
    for k in range(-11,33): need.setdefault(int(t0)+k,set()).add(p)
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
US=L[L.iso2.isin(['US','PR'])&(L.season>=2014)]
cty=set(pts.pid[pts.kind=='county']); usdays=set()
for r in US.itertuples():
    d0=(r.time.normalize()-D0).days; hi=180 if r.track_id=='2017260N12310' else 60
    for k in range(-14,hi+1): need.setdefault(d0+k,set()).update(cty); usdays.add(d0+k)
order=sorted(usdays)+sorted(set(need)-usdays)
print('points',len(pts),'days',len(order),'US days',len(usdays),flush=True)
def get(var,h):
    u=f'{B}{var}/{h}.0.0'
    for k in range(6):
        try:
            b=urllib.request.urlopen(u,timeout=90).read()
            return np.frombuffer(blosc.decode(b),'<f4')
        except Exception as e:
            time.sleep(2*(k+1))
    raise RuntimeError(u)
pool=ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 96)
def day(d):
    date=D0+pd.Timedelta(days=d); out=f'{OUT}/{date:%Y%m%d}.npz'
    if os.path.exists(out): return 0
    if date>pd.Timestamp('2026-06-30'): return 0
    p=np.array(sorted(need[d])); fl=flat[p]; h0=int((date-T0)/pd.Timedelta(hours=1))
    fT=[pool.submit(get,'2m_temperature',h0+h) for h in range(24)]; fD=[pool.submit(get,'2m_dewpoint_temperature',h0+h) for h in range(24)]; fP=pool.submit(get,'surface_pressure',h0+12)
    T=np.stack([f.result()[fl] for f in fT])-273.15; Td=np.stack([f.result()[fl] for f in fD])-273.15; P=fP.result()[fl]/100
    tmp=out+'.tmp.npz'; np.savez(tmp,pid=p.astype('int32'),T=np.round(T*100).astype('int16'),Td=np.round(Td*100).astype('int16'),P=np.round(P*10).astype('uint16')); os.replace(tmp,out)
    return 1
t=time.time(); n=0
with ThreadPoolExecutor(int(os.environ.get('DP','3'))) as dp:
    for k,r in enumerate(dp.map(day,order)):
        n+=r
        if k%100==0: print(k,len(order),'new',n,'elapsed',round(time.time()-t),'s',flush=True)
print('ALL DONE',flush=True)
