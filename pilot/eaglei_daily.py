# Stream full EAGLE-I yearly county files and keep daily county summaries for every day (needed for non-storm control windows):
# daily maximum and daily mean customers out (missing quarter-hours count as zero), and the number of quarter-hours reported.
import pandas as pd, numpy as np, os, time
FILES={2015:42547822,2016:42547825,2017:42547828,2018:42547879,2019:42547885,2020:42547894,2021:42547891,2022:42547897,2023:44574907,2024:53581661}
for y,f in FILES.items():
    out=f'data/eaglei/daily_{y}.parquet'
    if os.path.exists(out): continue
    t=time.time(); parts=[]
    for ch in pd.read_csv(f'https://ndownloader.figshare.com/files/{f}',chunksize=4_000_000,dtype={'fips_code':str}):
        ch=ch.rename(columns={'sum':'customers_out'}); ts=pd.to_datetime(ch.run_start_time)
        k=pd.DataFrame({'fips':ch.fips_code.str.zfill(5).values,'d':ts.dt.normalize().values,'v':ch.customers_out.values})
        k=k.drop_duplicates(['fips','d','v']) if False else k
        parts.append(k.groupby(['fips','d']).v.agg(['max','sum','count']).reset_index())
    K=pd.concat(parts).groupby(['fips','d']).agg({'max':'max','sum':'sum','count':'sum'}).reset_index()
    K['mean']=K['sum']/96.0; K=K.drop(columns='sum'); K.to_parquet(out,index=False)
    print(y,len(K),round(time.time()-t),'s',flush=True)
print('ALL DONE',flush=True)
