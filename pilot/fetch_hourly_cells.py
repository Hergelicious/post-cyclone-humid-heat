import pandas as pd, numpy as np, requests, io, os, time
from concurrent.futures import ThreadPoolExecutor
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index()
from global_land_mask import globe
def cells_within(la,lo,R):
    out=[]; dl=R/111/max(np.cos(np.radians(la)),0.3)
    for a in np.arange(np.round(la*2)/2-np.ceil(R/111*2)/2,np.round(la*2)/2+np.ceil(R/111*2)/2+0.01,0.5):
        for b in np.arange(np.round(lo*2)/2-np.ceil(dl*2)/2,np.round(lo*2)/2+np.ceil(dl*2)/2+0.01,0.5):
            b2=((b+180)%360)-180
            d=2*6371*np.arcsin(np.sqrt(np.sin(np.radians(a-la)/2)**2+np.cos(np.radians(la))*np.cos(np.radians(a))*np.sin(np.radians(b2-lo)/2)**2))
            if d<=R and globe.is_land(a,b2): out.append((round(a,1),round(b2,1)))
    return out
os.makedirs('pilot/hourly_cells',exist_ok=True)
jobs=[]
for r in lf[lf.season>=2001].itertuples():
    t0=r.time.normalize(); a=(t0+pd.Timedelta(days=1)).strftime('%Y%m%d'); b=(t0+pd.Timedelta(days=30)).strftime('%Y%m%d')
    for (la,lo) in cells_within(r.lat,r.lon,100): jobs.append((la,lo,a,b))
jobs=sorted(set(jobs)); print(len(jobs),'cell-windows',flush=True)
def get(j):
    la,lo,a,b=j; fn=f'pilot/hourly_cells/{la}_{lo}_{a}.csv'
    if os.path.exists(fn): return 1
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=LST"
    for k in range(3):
        try:
            t=requests.get(u,timeout=90).text
            if t.startswith('YEAR'):
                h=pd.read_csv(io.StringIO(t)).replace(-999,np.nan); h['HI']=hi_c(h.T2M,np.clip(h.RH2M,1,100))
                h.groupby(['YEAR','MO','DY']).HI.max().reset_index().to_csv(fn,index=False); return 1
        except Exception: pass
        time.sleep(3)
    return 0
n=0
with ThreadPoolExecutor(12) as ex:
    for ok in ex.map(get,jobs):
        n+=ok
        if n%500==0: print(n,flush=True)
print('done',n,flush=True)
