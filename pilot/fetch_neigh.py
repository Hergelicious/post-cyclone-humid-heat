import pandas as pd, numpy as np, os, requests, time
from concurrent.futures import ThreadPoolExecutor
from global_land_mask import globe
lf=pd.read_csv('pilot/landfalls.csv')
cells=set()
for la,lo in zip(lf.lat,lf.lon):
    dl=2.0/max(np.cos(np.radians(la)),0.3)
    for a in np.arange(np.round(la*2)/2-2,np.round(la*2)/2+2.01,0.5):
        for b in np.arange(np.round(lo*2)/2-np.ceil(dl*2)/2,np.round(lo*2)/2+np.ceil(dl*2)/2+0.01,0.5):
            b2=((b+180)%360)-180
            d=2*6371*np.arcsin(np.sqrt(np.sin(np.radians(a-la)/2)**2+np.cos(np.radians(la))*np.cos(np.radians(a))*np.sin(np.radians(b2-lo)/2)**2))
            if d<=200 and globe.is_land(a,b2): cells.add((round(a,1),round(b2,1)))
cells=sorted(cells); print(len(cells),'cells within 200 km',flush=True)
pd.DataFrame(cells,columns=['clat','clon']).to_csv('pilot/neigh_cells.csv',index=False)
def get(c):
    la,lo=c; fn=f'pilot/power/{la}_{lo}.csv'
    if os.path.exists(fn): return 1
    u=("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,QV2M,PS,PRECTOTCORR"
       f"&community=AG&longitude={lo}&latitude={la}&start=19810101&end=20251231&format=CSV&header=false")
    for k in range(4):
        try:
            r=requests.get(u,timeout=120)
            if r.status_code==200 and r.text.startswith('YEAR'): open(fn,'w').write(r.text); return 1
        except Exception: pass
        time.sleep(5*(k+1))
    return 0
with ThreadPoolExecutor(6) as ex: print('done',sum(ex.map(get,cells)),flush=True)
