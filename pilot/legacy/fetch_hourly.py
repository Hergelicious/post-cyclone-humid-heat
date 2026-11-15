import pandas as pd, numpy as np, requests, io, os, time
from concurrent.futures import ThreadPoolExecutor
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index()
ev=pd.read_csv('pilot/power_events_exposure.csv').set_index('idx'); lf['pri']=(ev.d406.reindex(lf.index).fillna(0)>0).astype(int); lf=lf.sort_values('pri',ascending=False)
os.makedirs('pilot/hourly',exist_ok=True)
def get(r):
    fn=f'pilot/hourly/{r.index}.csv'
    if os.path.exists(fn): return 1
    t0=r.time.normalize(); la=np.round(r.lat*2)/2; lo=np.round(r.lon*2)/2
    a=(t0+pd.Timedelta(days=1)).strftime('%Y%m%d'); b=(t0+pd.Timedelta(days=30)).strftime('%Y%m%d')
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=LST"
    for k in range(4):
        try:
            h=pd.read_csv(io.StringIO(requests.get(u,timeout=120).text)).replace(-999,np.nan)
            if 'T2M' in h: 
                h['HI']=hi_c(h.T2M,np.clip(h.RH2M,1,100)); d=h.groupby(['YEAR','MO','DY']).HI.max().reset_index(); d.to_csv(fn,index=False); return 1
        except Exception: pass
        time.sleep(5*(k+1))
    return 0
with ThreadPoolExecutor(14) as ex: print('done',sum(ex.map(get,[r for r in lf.itertuples()])),flush=True)
