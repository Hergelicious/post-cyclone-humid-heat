# Stream EAGLE-I county outage files (figshare 24237376, v4) and keep rows inside US/PR landfall windows
# (days -14..+60 after landfall; +180 for Maria), all counties. Output: data/eaglei/win_YYYY.parquet
import pandas as pd, numpy as np, os, time
FILES={2015:42547822,2016:42547825,2017:42547828,2018:42547879,2019:42547885,2020:42547894,2021:42547891,2022:42547897,2023:44574907,2024:53581661}
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
US=L[L.iso2.isin(['US','PR'])&(L.season>=2015)]
win=[]
for r in US.itertuples():
    d=r.time.normalize(); hi=180 if r.track_id=='2017260N12310' else 60
    win.append((d-pd.Timedelta(days=14),d+pd.Timedelta(days=hi+1)))
for y,f in FILES.items():
    out=f'data/eaglei/win_{y}.parquet'
    if os.path.exists(out): continue
    W=[(a,b) for a,b in win if a.year<=y<=b.year]
    if not W: continue
    t=time.time(); keep=[]
    for ch in pd.read_csv(f'https://ndownloader.figshare.com/files/{f}',chunksize=3_000_000,dtype={'fips_code':str}):
        ch=ch.rename(columns={'sum':'customers_out'}); ts=pd.to_datetime(ch.run_start_time)
        m=np.zeros(len(ch),bool)
        for a,b in W: m|=((ts>=a)&(ts<b)).values
        if m.any():
            k=ch.loc[m,['fips_code','state','customers_out']].copy(); k['t']=ts[m].values; keep.append(k)
    K=pd.concat(keep); K.to_parquet(out,index=False)
    print(y,len(K),'rows kept',round(time.time()-t),'s',flush=True)
print('ALL DONE',flush=True)
