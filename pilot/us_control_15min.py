# Non-storm controls with the main 15-minute algorithm: for control years 2019 and 2022, take the same calendar windows as every
# landfall of other years, keep candidate counties within 800 km with no tropical cyclone track point within 500 km during the window,
# and apply exactly the county series, baseline, 5% criterion and episode-end rule of the main analysis. The daily approximation used
# for all years is applied to the same windows to measure its error.
import pandas as pd, numpy as np, os, sys, time
src=open('pilot/us_outage_heat.py').read()
exec(src[src.index('NAMES='):src.index('# ---- EAGLE-I rows')])
GAP=4; exec(src[src.index('def county_series'):src.index('NEXT=')])
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
US=L[L.iso2.isin(['US','PR'])&(L.season>=2015)].sort_values('time').reset_index(drop=True)
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county'].copy(); cty['fips']=cty.key.str.zfill(5)
lat_of=dict(zip(cty.fips,cty.lat)); lon_of=dict(zip(cty.fips,cty.lon)); F=list(cty.fips); la=np.array([lat_of[f] for f in F]); lo=np.array([lon_of[f] for f in F])
MCC=pd.read_csv('data/eaglei/MCC.csv',encoding='utf-8-sig'); MCC['fips']=MCC.County_FIPS.astype(str).str.zfill(5); cust=dict(zip(MCC.fips,MCC.Customers))
TR=pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('huracanpy').__file__),'_data','_ibtracs_files','jtwc.csv'),low_memory=False,parse_dates=['time'])
TR=TR[(TR.season>=2014)&(TR.season<=2025)]; TR['lon']=((TR.lon+180)%360)-180; TR=TR[(TR.lat>10)&(TR.lat<50)&(TR.lon>-110)&(TR.lon<-55)]; TR['d']=TR.time.dt.normalize()
tcd={d_:np.min(np.hypot(g.lat.values[:,None]-la[None],(g.lon.values[:,None]-lo[None])*np.cos(np.radians(la))[None]),0)*111<500 for d_,g in TR.groupby('d')}
FILES={2019:42547885,2022:42547897}
DA=pd.concat([pd.read_parquet(f'data/eaglei/daily_{y}.parquet') for y in FILES]); DA['fips']=DA.fips.str.zfill(5)
out=[]
for y,fid in FILES.items():
    jobs=[]
    for r in US[US.season!=y].itertuples():
        day0=r.time.normalize()
        try: d1=day0.replace(year=y)
        except ValueError: d1=day0.replace(year=y,day=28)
        t0=d1-pd.Timedelta(hours=24)+(r.time-day0)
        near=np.hypot(la-r.lat,(lo-r.lon)*np.cos(np.radians(r.lat)))*111<800
        tc=np.zeros(len(F),bool)
        for d_ in pd.date_range(d1-pd.Timedelta(days=13),d1+pd.Timedelta(days=30)): tc|=tcd.get(d_,np.zeros(len(F),bool))
        jobs.append((NAMES.get(r.track_id,r.track_id),r.season,t0.floor('h'),set(np.array(F)[near&~tc])))
    lo_t=min(j[2] for j in jobs)-pd.Timedelta(days=14); hi_t=max(j[2] for j in jobs)+pd.Timedelta(days=32)
    cache=f'data/eaglei/ctrl15_{y}.parquet'
    if not os.path.exists(cache):
        t=time.time(); keep=[]
        for ch in pd.read_csv(f'https://ndownloader.figshare.com/files/{fid}',chunksize=4_000_000,dtype={'fips_code':str}):
            ch=ch.rename(columns={'sum':'customers_out'}); ts=pd.to_datetime(ch.run_start_time); m=((ts>=lo_t)&(ts<hi_t)).values
            if m.any(): keep.append(pd.DataFrame({'fips':ch.fips_code.str.zfill(5).values[m],'t':ts.values[m],'customers_out':ch.customers_out.values[m]}))
        K=pd.concat(keep).drop_duplicates(['fips','t']); K.to_parquet(cache,index=False); print(y,'rows',len(K),round(time.time()-t),'s',flush=True)
    K=pd.read_parquet(cache); K['fips']=K.fips.astype(str); G={f:x for f,x in K.groupby('fips')}
    Dd=DA[(DA.d.dt.year==y)]; Dm=Dd.pivot_table(index='d',columns='fips',values='mean',aggfunc='max'); Dx=Dd.pivot_table(index='d',columns='fips',values='max',aggfunc='max')
    days=pd.date_range(f'{y}-01-01',f'{y}-12-31'); Dm=Dm.reindex(days).fillna(0); Dx=Dx.reindex(index=days,columns=Dm.columns).fillna(0)
    nh=720+24
    for nm,season,t0,cands in jobs:
        n15=0; ch15=0.0; nd=0; chd=0.0
        i0=days.get_loc(t0.normalize()+pd.Timedelta(days=1)) if (t0.normalize()+pd.Timedelta(days=1)) in days else None
        for f in cands:
            c=cust.get(f,np.nan)
            if not np.isfinite(c) or c<=0: continue
            x=G.get(f)
            if x is not None:
                base_rows=x[(x.t>=t0-pd.Timedelta(days=13))&(x.t<t0)]; s_base=county_series(base_rows,t0-pd.Timedelta(days=13),12*24) if len(base_rows) else np.zeros(12*24)
                s=county_series(x[(x.t>=t0)&(x.t<t0+pd.Timedelta(hours=nh))],t0,nh); b=np.median(s_base); ex=np.clip(s-b,0,None)
                pk=ex[:24*5].max()
                if pk>=max(0.05*c,100):
                    ip=int(ex[:24*5].argmax()); low=ex<=max(0.005*c,20); run=np.convolve(low.astype(int),np.ones(24,int),'valid')>=24; k=np.where(run[ip:])[0]
                    if len(k): ex[ip+k[0]:]=0.0
                    n15+=1; ch15+=ex.sum()
            if i0 is not None and f in Dm.columns and i0-13>=0 and i0+31<=len(days):
                M_=Dm[f].values; X_=Dx[f].values; b=np.median(M_[i0-13:i0]); pkd=X_[i0-1:i0+5].max()-b
                if pkd>=max(0.05*c,100): nd+=1; chd+=np.clip(M_[i0-1:i0+31]-b,0,None).sum()*24
        out.append(dict(control_year=y,storm=nm,season=season,cands=len(cands),n15=n15,custh15=ch15,ndaily=nd,custh_daily=chd))
    print(y,'done',flush=True)
O=pd.DataFrame(out); O.to_csv('pilot/us_control_15min.csv',index=False)
S=pd.read_csv('pilot/us_v2_storms_main.csv'); S=S[S.counties.notna()]
per=O.groupby('storm')[['n15','custh15','ndaily','custh_daily']].mean()
print('per storm-equivalent (mean of the two control years): counties 15-min %.0f, daily %.0f; customer-hours 15-min %.1f M, daily %.1f M'%(per.n15.sum(),per.ndaily.sum(),per.custh15.sum()/1e6,per.custh_daily.sum()/1e6))
print('storm windows (15-min main analysis): county-storms %d, customer-hours %.0f M; control/storm with the 15-min algorithm: %.2f%%'%(S.counties.sum(),S.cust_h.sum()/1e6,100*per.custh15.sum()/S.cust_h.sum()))
