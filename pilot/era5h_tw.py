import pandas as pd, numpy as np, xarray as xr, os, sys, time
from remotezip import RemoteZip
from concurrent.futures import ProcessPoolExecutor
c=pd.read_csv('pilot/era5_cells1.csv')
zips={**{y:f'wbt_{s}-{s+4}.zip' for s in range(1980,2020,5) for y in range(s,s+5)},**{y:'wbt_2020-2022.zip' for y in (2020,2021,2022)}}
def do(ym):
    y,m=ym; out=f'data/era5h/Tw_{y}{m:02d}.npy'
    if os.path.exists(out): return 1
    zn=zips[y]; base=zn[:-4] if y<2020 else '2020-2022'; mem=f'{base}/wbt_{y}-{m:02d}.nc'
    tmp=f'data/era5h/tmp_{y}{m:02d}'
    for k in range(5):
        try:
            with RemoteZip(f'https://zenodo.org/api/records/10418938/files/{zn}/content') as z: z.extract(mem,tmp)
            d=xr.open_dataset(f'{tmp}/{mem}')
            v=d.wbt.sel(latitude=xr.DataArray(c.lat.values.astype('float32'),dims='cell'),longitude=xr.DataArray((c.lon.values%360 if float(d.longitude.max())>180 else c.lon.values).astype('float32'),dims='cell'),method='nearest').values.astype('float32')
            if np.nanmean(v)>200: v=v-273.15
            np.save(out,v); d.close(); os.remove(f'{tmp}/{mem}'); return 1
        except Exception as e:
            print('retry',ym,repr(e)[:100],flush=True); time.sleep(10*(k+1))
    return 0
months=[(y,m) for m in [9,8,10,7,11,6,5,12,4,1,2,3] for y in range(1981,2023)]
t=time.time()
with ProcessPoolExecutor(11) as ex:
    for i,r in enumerate(ex.map(do,months)):
        if i%12==0: print(i,round(time.time()-t),flush=True)
print('ALL DONE',flush=True)
