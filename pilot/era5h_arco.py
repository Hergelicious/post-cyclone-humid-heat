import fsspec, zarr, numpy as np, pandas as pd, os, time
from concurrent.futures import ThreadPoolExecutor
u='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr'
g=zarr.open_consolidated(fsspec.get_mapper(u),mode='r')
lat=g['latitude'][:]; lon=g['longitude'][:]
c=pd.read_csv('pilot/era5_cells1.csv')
ii=np.array([np.abs(lon-(x%360)).argmin() for x in c.lon]); jj=np.array([np.abs(lat-y).argmin() for y in c.lat])
print('dims',g['2m_temperature'].shape,lat[:3],lon[:3],flush=True)
t0=pd.Timestamp('1959-01-01')
def do(ym):
    y,m=ym; out=f'data/era5h/Tp_{y}{m:02d}.npz'
    if os.path.exists(out): return
    a=int((pd.Timestamp(y,m,1)-t0)/pd.Timedelta(hours=1)); b=int((pd.Timestamp(y,m,1)+pd.offsets.MonthBegin(1)-t0)/pd.Timedelta(hours=1))
    for k in range(4):
        try:
            T=g['2m_temperature'][a:b][:,ii,jj]-273.15; P=g['surface_pressure'][a:b][:,ii,jj]/100
            np.savez(out,T=T.astype('float32'),P=P.astype('float32')); return
        except Exception as e: print('retry',ym,repr(e)[:80],flush=True); time.sleep(5)
months=[(y,m) for y in range(1981,2023) for m in range(1,13)]
t=time.time()
with ThreadPoolExecutor(8) as ex:
    for i,_ in enumerate(ex.map(do,months)):
        if i%24==0: print(i,round(time.time()-t),flush=True)
print('ALL DONE',flush=True)
