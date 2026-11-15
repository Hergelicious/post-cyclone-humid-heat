import pandas as pd, numpy as np, os, time, requests, io
from concurrent.futures import ThreadPoolExecutor
D=pd.read_pickle('pilot/decomp_met_records.pkl')
src=open('pilot/reconcile.py').read()
mon_of=lambda t:(pd.Timestamp('1981-01-01')+pd.Timedelta(days=int(t))).month
D['month']=D.t0.map(mon_of); D['w']=D.pE+D.pL
sel=[]
for iso in ['BD','VN','CN','IN','PH']:
    g=D[D.iso2==iso].groupby('cell').agg(w=('w','sum'),month=('month',lambda x:x.mode()[0])).sort_values('w',ascending=False).head(15)
    for c,r in g.iterrows(): sel.append((iso,c[0],c[1],int(r.month)))
S=pd.DataFrame(sel,columns=['iso2','lat','lon','month']); S.to_csv('pilot/hourly_trend_cells.csv',index=False)
os.makedirs('data/hourly_trend',exist_ok=True)
jobs=[(r.lat,r.lon,y) for r in S.itertuples() for y in range(2001,2025)]
def get(j):
    la,lo,y=j; fn=f'data/hourly_trend/{la}_{lo}_{y}.csv'
    if os.path.exists(fn): return 1
    u=f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,QV2M&community=AG&longitude={lo}&latitude={la}&start={y}0101&end={y}1231&format=CSV&header=false&time-standard=LST"
    for k in range(4):
        try:
            r=requests.get(u,timeout=180)
            if r.status_code==200 and r.text.startswith('YEAR'): open(fn,'w').write(r.text); return 1
        except Exception: pass
        time.sleep(5*(k+1))
    return 0
print(len(jobs),flush=True)
with ThreadPoolExecutor(6) as ex: n=sum(ex.map(get,jobs))
print('done',n,flush=True)
