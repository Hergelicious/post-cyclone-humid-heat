import pandas as pd, numpy as np, requests, io, os, time
from concurrent.futures import ThreadPoolExecutor
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
ends={'Bhubaneswar':14,'Calcutta':7,'Houston':10,'NewOrleans':12,'SanJuan':170,'Visakhapatnam':14,'Zhanjiang':5,'Maoming':5}
ev=M[M.city.isin(ends)&(M.min7<0.6)]
os.makedirs('pilot/outage_hourly',exist_ok=True); jobs=[]
for r in ev.itertuples():
    la0=np.round(r.clat*2)/2; lo0=np.round(r.clon*2)/2; t0=r.time.normalize()
    a=(t0+pd.Timedelta(days=1)).strftime('%Y%m%d'); b=(t0+pd.Timedelta(days=ends[r.city])).strftime('%Y%m%d')
    for dla in (-0.5,0,0.5):
        for dlo in (-0.5,0,0.5): jobs.append((r.city,round(la0+dla,1),round(lo0+dlo,1),a,b,dla==0 and dlo==0))
def get(j):
    city,la,lo,a,b,centre=j; fn=f'pilot/outage_hourly/{city}_{la}_{lo}.csv'
    if os.path.exists(fn): return 1
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=LST"
    for k in range(3):
        try:
            t=requests.get(u,timeout=120).text
            if t.startswith('YEAR'):
                h=pd.read_csv(io.StringIO(t)).replace(-999,np.nan); h['HI']=hi_c(h.T2M,np.clip(h.RH2M,1,100))
                h[['YEAR','MO','DY','HR','T2M','RH2M','HI']].to_csv(fn,index=False); return 1
            else: print('bad',city,la,lo,t[:200],flush=True)
        except Exception as e: print('err',e,flush=True)
        time.sleep(3)
    return 0
with ThreadPoolExecutor(8) as ex: print(sum(ex.map(get,jobs)),'of',len(jobs))
