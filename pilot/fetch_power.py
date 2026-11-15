import pandas as pd, numpy as np, os, io, time, requests
from concurrent.futures import ThreadPoolExecutor
lf=pd.read_csv('pilot/landfalls.csv')
lf['clat']=(np.round(lf.lat*2)/2); lf['clon']=(np.round(lf.lon*2)/2)
cells=lf[['clat','clon']].drop_duplicates().values.tolist()
print(len(cells),'cells',flush=True)
def get(c):
    la,lo=c; fn=f'pilot/power/{la}_{lo}.csv'
    if os.path.exists(fn): return 1
    u=("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,QV2M,PS,PRECTOTCORR"
       f"&community=AG&longitude={lo}&latitude={la}&start=19810101&end=20251231&format=CSV&header=false")
    for k in range(4):
        try:
            r=requests.get(u,timeout=120)
            if r.status_code==200 and r.text.startswith('YEAR'):
                open(fn,'w').write(r.text); return 1
        except Exception as e: pass
        time.sleep(5*(k+1))
    return 0
with ThreadPoolExecutor(6) as ex: ok=sum(ex.map(get,cells))
print('done',ok,'of',len(cells),flush=True)
