import pandas as pd, os, time
from remotezip import RemoteZip
m=pd.read_csv('pilot/hdntl_matches.csv',parse_dates=['time'])
need={}
for _,r in m.iterrows():
    yrs={r.time.year}
    if r.time.dayofyear<45 and r.time.year>2012: yrs.add(r.time.year-1)
    if r.time.dayofyear>300 and r.time.year<2024: yrs.add(r.time.year+1)
    for y in yrs: need.setdefault(int(r.serial),set()).add(y)
byzip={}
for s,ys in need.items():
    lo=(s-1)//10*10+1; z=f"{lo:03d}-{min(lo+9,653):03d}.zip"; byzip.setdefault(z,{})[s]=ys
print(len(byzip),sum(len(v) for d in byzip.values() for v in d.values()),flush=True)
from concurrent.futures import ThreadPoolExecutor
def work(item):
    z,d=item
    for att in range(5):
        try:
            with RemoteZip(f'https://zenodo.org/api/records/17079409/files/{z}/content') as rz:
                for i in rz.infolist():
                    fn=i.filename; b=os.path.basename(fn)
                    if not b.endswith('.tif') or 'Flag' in b: continue
                    s=int(b.split('_')[0]); y=int(b.split('_')[-1][:4])
                    if s in d and y in d[s]:
                        out=f'data/hdntl/tif/{b}'
                        if not os.path.exists(out):
                            open(out,'wb').write(rz.read(fn))
            print('done',z,flush=True); break
        except Exception as e:
            print('retry',z,e,flush=True); time.sleep(10)

with ThreadPoolExecutor(4) as ex: list(ex.map(work,sorted(byzip.items())))
