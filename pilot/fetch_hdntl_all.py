import pandas as pd, os, time
from remotezip import RemoteZip
from concurrent.futures import ThreadPoolExecutor
m=pd.read_csv('pilot/hdntl_matches.csv'); m=m[m.dist<=100]; S=set(m.serial.astype(int))
byzip={}
for s in S:
    lo=(s-1)//10*10+1; byzip.setdefault(f"{lo:03d}-{min(lo+9,653):03d}.zip",set()).add(s)
def work(item):
    z,ss=item
    for att in range(6):
        try:
            with RemoteZip(f'https://zenodo.org/api/records/17079409/files/{z}/content') as rz:
                for i in rz.infolist():
                    b=os.path.basename(i.filename)
                    if not b.endswith('.tif') or 'Flag' in b: continue
                    if int(b.split('_')[0]) in ss:
                        out=f'data/hdntl/tif/{b}'
                        if not os.path.exists(out):
                            tmp=out+'.part'; open(tmp,'wb').write(rz.read(i.filename)); os.rename(tmp,out)
            print('done',z,flush=True); return
        except Exception as e: print('retry',z,repr(e)[:100],flush=True); time.sleep(15)
print(len(S),'cities',len(byzip),'zips',flush=True)
with ThreadPoolExecutor(5) as ex: list(ex.map(work,sorted(byzip.items())))
print('ALL DONE',flush=True)
