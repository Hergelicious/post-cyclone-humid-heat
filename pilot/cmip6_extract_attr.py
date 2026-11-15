import pandas as pd, numpy as np, fsspec, zarr, os, glob, sys, time
from concurrent.futures import ThreadPoolExecutor
d=pd.read_csv('https://storage.googleapis.com/cmip6/pangeo-cmip6.csv')  # Pangeo CMIP6 catalogue; d=d[d.table_id=='Amon']
cells=np.load('data/cmip6/cells4276.npy')
V=['tas','tasmax','hurs']
def members(m,e):
    s=d[(d.source_id==m)&(d.experiment_id==e)&(d.variable_id.isin(V))]
    g=s.groupby('member_id').variable_id.nunique(); return sorted(g[g==3].index)
todo=[tuple(x) for x in pd.read_csv('pilot/attr_jobs.csv').values]
print(len(todo),flush=True)
def tdecode(g,v):
    t=g['time']; units=t.attrs['units']; cal=t.attrs.get('calendar','standard')
    import cftime; dt=cftime.num2date(t[:],units,cal); return np.array([x.year for x in dt]),np.array([x.month for x in dt])
def run(job):
    m,e,r=job; out=f'data/cmip6/{m}_{e}_{r}.npz'
    if os.path.exists(out): return
    for att in range(3):
        try:
            res={}
            for v in V:
                z=d[(d.source_id==m)&(d.experiment_id==e)&(d.member_id==r)&(d.variable_id==v)].sort_values('version').zstore.iloc[-1]
                g=zarr.open_consolidated(fsspec.get_mapper(z.replace('gs://','https://storage.googleapis.com/')),mode='r')
                yr,mo=tdecode(g,v); sel=np.where(yr>=1950)[0]
                lat=g['lat'][:]; lon=g['lon'][:]%360
                a=g[v]
                ii=np.abs(lat[:,None]-cells[:,0][None]).argmin(0); jj=np.abs(((lon[:,None]-cells[:,1][None]%360)+180)%360-180).argmin(0)
                w=np.cos(np.radians(lat))[:,None]*np.ones(len(lon))[None]
                pts=[];gm=[]
                for k0 in range(sel[0],sel[-1]+1,120):
                    data=np.asarray(a[k0:min(k0+120,sel[-1]+1)],dtype='float32')
                    if v=='tas': gm.append((data*w).sum((1,2))/w.sum())
                    pts.append(data[:,ii,jj]); del data
                if v=='tas': res['gmst']=np.concatenate(gm)
                res[v]=np.concatenate(pts); res['yr']=yr[sel]; res['mo']=mo[sel]
            np.savez_compressed(out,**res); print('ok',job,flush=True); return
        except Exception as ex:
            print('err',job,repr(ex)[:200],flush=True); time.sleep(5)
with ThreadPoolExecutor(3) as ex: list(ex.map(run,todo))
print('ALL DONE',flush=True)
