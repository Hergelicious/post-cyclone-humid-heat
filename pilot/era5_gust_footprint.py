# Physically based footprint check: maximum ERA5 10 m wind gust (hourly maximum since previous post-processing, 0.25 deg ARCO store)
# from 6 h before to 18 h after each landfall, every 2 h, aggregated to the 0.5 deg cells within 400 km of the landfall point.
import numpy as np, pandas as pd, urllib.request, numcodecs, time, os, sys
from concurrent.futures import ThreadPoolExecutor
B='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'; V='10m_wind_gust_since_previous_post_processing'
T0=pd.Timestamp('1900-01-01'); blosc=numcodecs.Blosc()
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
def get(h):
    for k in range(6):
        try: return np.frombuffer(blosc.decode(urllib.request.urlopen(f'{B}{V}/{h}.0.0',timeout=90).read()),'<f4').reshape(721,1440)
        except Exception: time.sleep(2*(k+1))
    return None
lat=90-0.25*np.arange(721); lon=0.25*np.arange(1440)
def one(i):
    r=L.loc[i]; h0=int((r.time.floor('h')-T0)/pd.Timedelta(hours=1))
    # 0.25 deg points within 400 km
    jj=np.where(np.abs(lat-r.lat)<=4.0)[0]; dlon=4.0/max(np.cos(np.radians(r.lat)),0.2)
    ii=np.where(np.abs(((lon-r.lon%360)+180)%360-180)<=dlon)[0]
    G=None
    for h in range(-6,19,2):
        a=get(h0+h)
        if a is None: continue
        s=a[np.ix_(jj,ii)]; G=s if G is None else np.maximum(G,s)
    if G is None: return i,None
    LA,LO=np.meshgrid(lat[jj],((lon[ii]+180)%360)-180,indexing='ij'); d=np.hypot(LA-r.lat,(LO-r.lon)*np.cos(np.radians(r.lat)))*111
    m=d<=400; cla=np.floor(LA[m]/0.5)*0.5+0.25; clo=np.floor(LO[m]/0.5)*0.5+0.25
    df=pd.DataFrame(dict(cla=cla,clo=clo,g=G[m])).groupby(['cla','clo']).g.max().reset_index(); df['idx']=i
    return i,df
t=time.time(); out=[]
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 24) as ex:
    for k,(i,df) in enumerate(ex.map(one,L.index)):
        if df is not None: out.append(df)
        if k%100==0: print(k,len(L),round(time.time()-t),'s',flush=True)
pd.concat(out).to_parquet('pilot/era5_gust_cells.parquet',index=False); print('ALL DONE',flush=True)
