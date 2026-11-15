# Non-storm control windows for the EAGLE-I outage selection: apply the same county criterion (excess outage >= 5% of customers,
# min 100, between day -1 and day +4) to the same calendar windows in other years 2015-2024 with no tropical cyclone track point
# within 500 km of the county from day -13 to day +30, and compare the number of qualifying counties and their customer-hours.
# Uses daily county summaries (data/eaglei/daily_YYYY.parquet).
import pandas as pd, numpy as np, glob
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
US=L[L.iso2.isin(['US','PR'])&(L.season>=2015)].sort_values('time').reset_index(drop=True)
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('NAMES='):src.index('# ---- EAGLE-I rows')])
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county'].copy(); cty['fips']=cty.key.str.zfill(5)
lat_of=dict(zip(cty.fips,cty.lat)); lon_of=dict(zip(cty.fips,cty.lon))
MCC=pd.read_csv('data/eaglei/MCC.csv',encoding='utf-8-sig'); MCC['fips']=MCC.County_FIPS.astype(str).str.zfill(5); cust=dict(zip(MCC.fips,MCC.Customers))
D=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('data/eaglei/daily_*.parquet'))]); D['fips']=D.fips.str.zfill(5)
D=D[D.fips.isin(lat_of)]
piv_mean=D.pivot_table(index='d',columns='fips',values='mean',aggfunc='max').sort_index()
piv_max=D.pivot_table(index='d',columns='fips',values='max',aggfunc='max').sort_index()
days=pd.date_range('2015-01-01','2024-12-31'); piv_mean=piv_mean.reindex(days).fillna(0); piv_max=piv_max.reindex(days).fillna(0)
F=list(piv_mean.columns); la=np.array([lat_of[f] for f in F]); lo=np.array([lon_of[f] for f in F]); C=np.array([cust.get(f,np.nan) for f in F])
TR=pd.read_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('huracanpy').__file__),'_data','_ibtracs_files','jtwc.csv'),low_memory=False,parse_dates=['time'])
TR=TR[(TR.season>=2014)&(TR.season<=2025)]; TR['lon']=((TR.lon+180)%360)-180; TR=TR[(TR.lat>10)&(TR.lat<50)&(TR.lon>-110)&(TR.lon<-55)]
TR['d']=TR.time.dt.normalize()
# for each day, counties with a TC track point within 500 km that day
tc_near={}
for d_,g in TR.groupby('d'):
    dd=np.min(np.hypot(g.lat.values[:,None]-la[None],(g.lon.values[:,None]-lo[None])*np.cos(np.radians(la))[None]),0)*111
    tc_near[d_]=dd<500
def tc_in(a,b):
    m=np.zeros(len(F),bool)
    for d_ in pd.date_range(a,b):
        if d_ in tc_near: m|=tc_near[d_]
    return m
M=piv_mean.values; X=piv_max.values; di={d:i for i,d in enumerate(days)}
def evaluate(day0,cand):
    i=di.get(day0)
    if i is None or i-13<0 or i+31>len(days): return None
    b=np.median(M[i-13:i],0); pk=X[i-1:i+5].max(0)-b
    q=cand&np.isfinite(C)&(C>0)&(pk>=np.maximum(0.05*C,100))
    ch=(np.clip(M[i-1:i+31]-b,0,None).sum(0)*24)
    return q,ch
rows=[]
for r in US.itertuples():
    nm=NAMES.get(r.track_id,r.track_id); day0=r.time.normalize()
    cand=np.hypot(la-r.lat,(lo-r.lon)*np.cos(np.radians(r.lat)))*111<800
    ev=evaluate(day0,cand)
    if ev is None: continue
    q,ch=ev; ctrl_q=[]; ctrl_ch=[]
    for y in range(2015,2025):
        if y==day0.year: continue
        try: d1=day0.replace(year=y)
        except ValueError: d1=day0.replace(year=y,day=28)
        ok=cand&~tc_in(d1-pd.Timedelta(days=13),d1+pd.Timedelta(days=30))
        e=evaluate(d1,ok)
        if e is None: continue
        qq,cc=e; frac=ok.sum()/cand.sum()
        ctrl_q.append(qq.sum()/max(frac,1e-9)); ctrl_ch.append((cc*qq).sum()/max(frac,1e-9))
    rows.append(dict(storm=nm,season=r.season,month=day0.month,cand=int(cand.sum()),storm_counties=int(q.sum()),storm_custh=float((ch*q).sum()),
                     ctrl_counties=float(np.mean(ctrl_q)) if ctrl_q else np.nan,ctrl_custh=float(np.mean(ctrl_ch)) if ctrl_ch else np.nan,n_ctrl=len(ctrl_q)))
R=pd.DataFrame(rows); R.to_csv('pilot/us_outage_control.csv',index=False)
print(R.round(1).to_string())
print('TOTAL storm counties %d, control counties per storm-equivalent %.0f (ratio %.3f); storm cust-h %.0f M, control %.0f M (ratio %.3f)'%(
      R.storm_counties.sum(),R.ctrl_counties.sum(),R.ctrl_counties.sum()/R.storm_counties.sum(),R.storm_custh.sum()/1e6,R.ctrl_custh.sum()/1e6,R.ctrl_custh.sum()/R.storm_custh.sum()))
