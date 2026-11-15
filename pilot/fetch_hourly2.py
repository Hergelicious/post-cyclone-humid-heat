import pandas as pd, numpy as np, requests, io, os, time, sys
from concurrent.futures import ThreadPoolExecutor
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index()
todo=[r for r in lf.itertuples() if r.season>=2001 and not os.path.exists(f'pilot/hourly/{r.index}.csv')]
print(len(todo),'to do',flush=True)
def get(r):
    fn=f'pilot/hourly/{r.index}.csv'; t0=r.time.normalize(); la=np.round(r.lat*2)/2; lo=np.round(r.lon*2)/2
    a=(t0+pd.Timedelta(days=1)).strftime('%Y%m%d'); b=(t0+pd.Timedelta(days=30)).strftime('%Y%m%d')
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=LST"
    for k in range(3):
        try:
            t=requests.get(u,timeout=90).text
            if t.startswith('YEAR'):
                h=pd.read_csv(io.StringIO(t)).replace(-999,np.nan); h['HI']=hi_c(h.T2M,np.clip(h.RH2M,1,100))
                h.groupby(['YEAR','MO','DY']).HI.max().reset_index().to_csv(fn,index=False); return 1
            print('bad',r.index,t[:80],flush=True)
        except Exception as e: print('err',r.index,str(e)[:60],flush=True)
        time.sleep(3)
    return 0
n=0
with ThreadPoolExecutor(10) as ex:
    for ok in ex.map(get,todo):
        n+=ok
        if n%50==0: print(n,flush=True)
print('done',n,flush=True)
