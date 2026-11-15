import fsspec, zarr, numpy as np, pandas as pd, time
u='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr'
g=zarr.open_consolidated(fsspec.get_mapper(u),mode='r'); tp=g['total_precipitation']
cells=pd.read_csv('pilot/wbt_cells.csv')
li=((cells.wlon.values%360)).astype(int); la=(cells.wlat.values+90).astype(int)
h0=int((pd.Timestamp('1981-01-01')-pd.Timestamp('1959-01-01'))/pd.Timedelta(hours=1)); h1=tp.shape[0]
out=np.zeros(((h1-h0)//24,len(cells)),'float32'); step=24*40; t=time.time()
for s in range(h0,h1,step):
    e=min(s+step,h1); a=tp[s:e]; v=a[:,li,la]  # hours x cells
    d=v.reshape(-1,24,len(cells)).sum(1)*1000
    out[(s-h0)//24:(e-h0)//24]=d
    if ((s-h0)//step)%40==0: print(s,round(time.time()-t),flush=True)
np.save('data/wbt/era5_precip_daily_cells.npy',out); print('done',out.shape,flush=True)
