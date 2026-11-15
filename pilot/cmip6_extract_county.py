# Extract monthly tasmax and hurs at US/PR county points (3,221) for every CMIP6 run already used (historical, ssp585, hist-nat),
# so that county outage counterfactuals use the model change at the county itself rather than at the nearest landfall cell.
import pandas as pd, numpy as np, fsspec, zarr, os, glob, time
from concurrent.futures import ThreadPoolExecutor
d=pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')  # Pangeo CMIP6 catalogue; d=d[d.table_id=='Amon']
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county']
P=np.c_[cty.lat.values,cty.lon.values%360]; np.save('data/cmip6cty/points.npy',P); np.save('data/cmip6cty/fips.npy',cty.key.str.zfill(5).values)
todo=[os.path.basename(f)[:-4].split('_') for f in sorted(glob.glob('data/cmip6/*_*_*.npz')) if os.path.basename(f).count('_')==2]
print(len(todo),flush=True)
import cftime
def run(job):
    m,e,r=job; out=f'data/cmip6cty/{m}_{e}_{r}.npz'
    if os.path.exists(out): return
    for att in range(3):
        try:
            res={}
            for v in ['tasmax','hurs']:
                z=d[(d.source_id==m)&(d.experiment_id==e)&(d.member_id==r)&(d.variable_id==v)].sort_values('version').zstore.iloc[-1]
                g=zarr.open_consolidated(fsspec.get_mapper(z.replace('gs://','https://storage.googleapis.com/')),mode='r')
                t=g['time']; dt=cftime.num2date(t[:],t.attrs['units'],t.attrs.get('calendar','standard'))
                yr=np.array([x.year for x in dt]); mo=np.array([x.month for x in dt]); sel=np.where(yr>=1950)[0]
                lat=g['lat'][:]; lon=g['lon'][:]%360
                ii=np.abs(lat[:,None]-P[:,0][None]).argmin(0); jj=np.abs(((lon[:,None]-P[:,1][None])+180)%360-180).argmin(0)
                a=g[v]; out_=[]
                for k0 in range(sel[0],sel[-1]+1,120):
                    data=np.asarray(a[k0:min(k0+120,sel[-1]+1)],dtype='float32'); out_.append(data[:,ii,jj]); del data
                res[v]=np.concatenate(out_); res['yr']=yr[sel]; res['mo']=mo[sel]
            np.savez_compressed(out,**res); print('ok',job,flush=True); return
        except Exception as ex:
            print('err',job,repr(ex)[:200],flush=True); time.sleep(5)
with ThreadPoolExecutor(4) as ex: list(ex.map(run,todo))
print('ALL DONE',flush=True)
