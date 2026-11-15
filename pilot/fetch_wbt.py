import pandas as pd, numpy as np, xarray as xr, os, sys, time
from remotezip import RemoteZip
from concurrent.futures import ProcessPoolExecutor as ThreadPoolExecutor
cells=pd.read_csv('pilot/wbt_cells.csv')
lat=xr.DataArray(cells.wlat.values.astype('float32'),dims='cell'); lon=xr.DataArray(cells.wlon.values.astype('float32'),dims='cell')
zips={**{1979:'wbt_1979.zip'},**{y:f'wbt_{s}-{s+4}.zip' for s in range(1980,2020,5) for y in range(s,s+5)},**{y:'wbt_2020-2022.zip' for y in (2020,2021,2022)}}
os.makedirs('data/wbt/daily',exist_ok=True)
def do(ym):
    y,m=ym; out=f'data/wbt/daily/{y}-{m:02d}.npy'
    if os.path.exists(out): return 1
    zn=zips[y]; base=zn[:-4] if y<2020 else '2020-2022'; mem=f'{base}/wbt_{y}-{m:02d}.nc'
    url=f'https://zenodo.org/api/records/10418938/files/{zn}/content'
    tmp=f'data/wbt/tmp_{y}{m:02d}'
    for k in range(4):
        try:
            with RemoteZip(url) as z: z.extract(mem,tmp)
            d=xr.open_dataset(f'{tmp}/{mem}')
            v=d.wbt.sel(latitude=lat,longitude=lon).load()
            dm=v.resample(time='1D').max().values.astype('float32')  # days x cells
            if np.nanmean(dm)>200: dm=dm-273.15
            np.save(out,dm); d.close(); os.remove(f'{tmp}/{mem}'); return 1
        except Exception as e:
            print('retry',ym,e,flush=True); time.sleep(10*(k+1))
    return 0
months=[(y,m) for y in range(1981,2023) for m in range(1,13)]
t=time.time()
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv)>1 else 4) as ex:
    for i,r in enumerate(ex.map(do,months)):
        if i%12==0: print(i,round(time.time()-t),flush=True)
print('done',flush=True)
