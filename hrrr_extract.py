# Archived HRRR forecasts (Google Cloud public archive) at county points for US landfalls 2018-2024 with hazardous outage-heat exposure:
# 2 m temperature and dewpoint for leads 1-48 h (1-36 h where only 36 h runs exist) from the 00 UTC run on each of days 0-6 after landfall.
import urllib.request, numpy as np, pandas as pd, eccodes, os, time
from concurrent.futures import ThreadPoolExecutor
B='https://storage.googleapis.com/high-resolution-rapid-refresh'
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl'); R=pd.read_csv('pilot/us_v2_storms_main.csv'); R=R[R.counties.notna()]
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('NAMES='):src.index('# ---- EAGLE-I rows')])
lf={NAMES.get(t,t):tm for t,tm in zip(L.track_id,L.time) if True}
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county'].copy(); cty['fips']=cty.key.str.zfill(5)
LL=np.load('data/hrrr/latlon.npy'); la=LL[0].ravel(); lo=(LL[1].ravel()+180)%360-180
from scipy.spatial import cKDTree
tree=cKDTree(np.c_[la,lo*np.cos(np.radians(30))])
sel=R[(R.season>=2018)&(R['hour35|obs']>=1e5)].storm.tolist(); print('storms',sel,flush=True)
os.makedirs('data/hrrr',exist_ok=True)
def get(date,cyc,fh,var):
    base=f'{B}/hrrr.{date}/conus/hrrr.t{cyc:02d}z.wrfsfcf{fh:02d}.grib2'
    for a_ in range(4):
        try:
            idx=urllib.request.urlopen(base+'.idx',timeout=60).read().decode().splitlines()
            for i,l in enumerate(idx):
                p=l.split(':')
                if p[3]==var and p[4]=='2 m above ground':
                    a=int(p[1]); b=int(idx[i+1].split(':')[1])-1
                    data=urllib.request.urlopen(urllib.request.Request(base,headers={'Range':f'bytes={a}-{b}'}),timeout=120).read()
                    g=eccodes.codes_new_from_message(data); v=eccodes.codes_get_values(g); eccodes.codes_release(g); return v
            return None
        except urllib.error.HTTPError as e:
            if e.code==404: return None
            time.sleep(3)
        except Exception: time.sleep(3)
    return None
for nm in sel:
    out=f'data/hrrr/{nm}.npz'
    if os.path.exists(out): continue
    r=[x for x in recs if x['storm']==nm][0]; fl=r['fips']; c=cty.set_index('fips').loc[fl]
    _,gi=tree.query(np.c_[c.lat.values,c.lon.values*np.cos(np.radians(30))])
    d0=lf[nm].normalize(); inits=[d0+pd.Timedelta(days=k) for k in range(7)]
    T=np.full((7,48,len(fl)),np.nan,'float32'); D=T.copy(); t=time.time()
    jobs=[(k,h,var) for k in range(7) for h in range(1,49) for var in ('TMP','DPT')]
    def run(j):
        k,h,var=j; v=get(inits[k].strftime('%Y%m%d'),0,h,var); return j,(None if v is None else v[gi])
    with ThreadPoolExecutor(12) as ex:
        for (k,h,var),v in ex.map(run,jobs):
            if v is not None: (T if var=='TMP' else D)[k,h-1]=v-273.15
    np.savez_compressed(out,T=T,D=D,fips=np.array(fl),inits=np.array([str(x) for x in inits]))
    print(nm,'done',round(time.time()-t),'s; leads available',int(np.isfinite(T[0,:,0]).sum()),flush=True)
print('ALL DONE',flush=True)
