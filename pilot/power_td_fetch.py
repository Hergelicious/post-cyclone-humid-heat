# Fetch hourly NASA POWER 2 m temperature and dewpoint (UTC) for the most populous LMIC landfall cells within 100 km, 2001-2024,
# for days t0-11 .. t0+32 (same window as the ERA5 extraction), to compare ERA5 and NASA POWER hour by hour.
import pandas as pd, numpy as np, requests, io, os
from concurrent.futures import ThreadPoolExecutor
D0=pd.Timestamp('1981-01-01')
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[(R.foot=='100')&R.lmic&(R.season>=2001)].copy()
R['key']=[f'{a}_{b}_{int(t)}' for (a,b),t in zip(R.cell,R.t0)]
top=R.sort_values('pop',ascending=False).drop_duplicates('key')
N=int(os.environ.get('NTOP','600')); top=top.head(N)
os.makedirs('data/power_td',exist_ok=True)
def get(r):
    (la,lo),t0=r.cell,int(r.t0); fn=f'data/power_td/{la}_{lo}_{t0}.csv'
    if os.path.exists(fn): return 1
    a=(D0+pd.Timedelta(days=t0-11)).strftime('%Y%m%d'); b=(D0+pd.Timedelta(days=t0+32)).strftime('%Y%m%d')
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,T2MDEW,RH2M&community=AG&longitude={lo}&latitude={la}&start={a}&end={b}&format=CSV&header=false&time-standard=UTC"
    for k in range(3):
        try:
            t=requests.get(u,timeout=120).text
            if t.startswith('YEAR'): open(fn,'w').write(t); return 1
        except Exception: pass
    return 0
with ThreadPoolExecutor(6) as ex: ok=sum(ex.map(get,[r for r in top.itertuples()]))
print('fetched',ok,'of',len(top),'cell-windows; population share of LMIC 2001-2024 record population',round(top['pop'].sum()/R.drop_duplicates('key')['pop'].sum(),3))
