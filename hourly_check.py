import pandas as pd, numpy as np, requests, io
from concurrent.futures import ThreadPoolExecutor
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
ev=pd.read_csv('pilot/power_events_exposure.csv'); lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'])
ev['time']=lf.time.values[ev.idx]; s=ev[ev.season>=2002].sample(150,random_state=1)
def get(r):
    t0=pd.Timestamp(r.time).normalize(); la=np.round(r.lat*2)/2; lo=np.round(r.lon*2)/2
    a=(t0+pd.Timedelta(days=1)).strftime('%Y%m%d'); b=(t0+pd.Timedelta(days=30)).strftime('%Y%m%d')
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=LST"
    try:
        h=pd.read_csv(io.StringIO(requests.get(u,timeout=120).text)).replace(-999,np.nan)
        h['HI']=hi_c(h.T2M,np.clip(h.RH2M,1,100)); d=h.groupby(['YEAR','MO','DY']).HI.max()
        return dict(idx=r.idx,h35=(d>=35).sum(),h406=(d>=40.6).sum(),a35=r.d35,a406=r.d406,lmic=r.lmic)
    except Exception as e: return None
with ThreadPoolExecutor(6) as ex: R=[x for x in ex.map(get,[r for r in s.itertuples()]) if x]
R=pd.DataFrame(R); R.to_csv('pilot/hourly_check.csv',index=False)
print(len(R),'events; corr days35',round(R[['h35','a35']].corr().iloc[0,1],2),' mean hourly',round(R.h35.mean(),2),'daily-approx',round(R.a35.mean(),2))
print('days406 corr',round(R[['h406','a406']].corr().iloc[0,1],2),' mean hourly',round(R.h406.mean(),2),'approx',round(R.a406.mean(),2),' any406 agreement',round(((R.h406>0)==(R.a406>0)).mean(),2))
