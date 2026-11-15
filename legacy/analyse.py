import pandas as pd, numpy as np, json, os, glob
import statsmodels.formula.api as smf
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'])
lf['clat']=np.round(lf.lat*2)/2; lf['clon']=np.round(lf.lon*2)/2
def hi_c(T,RH):
    F=T*9/5+32
    hi=0.5*(F+61+(F-68)*1.2+RH*0.094)
    full=(-42.379+2.04901523*F+10.14333127*RH-.22475541*F*RH-.00683783*F*F-.05481717*RH*RH
          +.00122874*F*F*RH+.00085282*F*RH*RH-.00000199*F*F*RH*RH)
    hi=np.where((hi+F)/2>=80,full,hi)
    return (hi-32)*5/9
cache={}
def series(la,lo):
    k=(la,lo)
    if k in cache: return cache[k]
    fn=f'pilot/power/{la}_{lo}.csv'
    if not os.path.exists(fn): cache[k]=None; return None
    s=pd.read_csv(fn).replace(-999,np.nan)
    s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
    q=s.QV2M/1000; p=s.PS*10  # hPa
    e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5))
    rh=np.clip(100*e/es,0,100); s['HI']=hi_c(s.T2M_MAX,rh)
    s=s.set_index('date'); cache[k]=s; return s
rows=[]
for r in lf.itertuples():
    s=series(r.clat,r.clon)
    if s is None: continue
    t0=r.time.normalize()
    w=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30)]
    rain=s.loc[t0-pd.Timedelta(days=1):t0+pd.Timedelta(days=2),'PRECTOTCORR'].sum()
    # baseline: same calendar window, 1981-2000, excluding the event year
    doy=t0.dayofyear; b=s[(s.index.year<=2000)&(s.index.year!=t0.year)]
    bd=((b.index.dayofyear-doy)%366).isin(range(1,31))
    nb=b[bd].groupby(b[bd].index.year)
    rows.append(dict(track=r.track_id,season=r.season,iso2=r.iso2,wind=r.lf_wind,lat=r.lat,lon=r.lon,
        d406=(w.HI>=40.6).sum(),d35=(w.HI>=35).sum(),himax=w.HI.max(),rain4d=rain,
        base_d35=nb.apply(lambda x:(x.HI>=35).sum()).mean(),base_d406=nb.apply(lambda x:(x.HI>=40.6).sum()).mean()))
ev=pd.DataFrame(rows)
wb=json.load(open('pilot/wb_countries.json'))[1]
inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
ev['inc']=ev.iso2.map(inc).fillna('NA')
ev['lmic']=ev.inc.isin(['LIC','LMC','UMC'])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean
ev['gmst']=ev.season.map(g)
ev['period']=pd.cut(ev.season,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024'])
ev.to_csv('pilot/events.csv',index=False)
print('events with climate data:',len(ev),' LMIC share:',round(ev.lmic.mean(),2))
print(ev.inc.value_counts().to_dict())
t=ev.groupby('period',observed=True).agg(n=('track','size'),any406=('d406',lambda x:(x>0).mean()),
   any35=('d35',lambda x:(x>0).mean()),mean_d35=('d35','mean'),base_d35=('base_d35','mean'),himax=('himax','mean'))
print(t.round(3))
print(ev.groupby(['lmic','period'],observed=True).agg(n=('track','size'),any406=('d406',lambda x:(x>0).mean()),mean_d35=('d35','mean'),excess_d35=('d35',lambda x: x.mean())).round(3))
ev['excess35']=ev.d35-ev.base_d35
m=smf.ols('d35 ~ gmst + base_d35',data=ev).fit(cov_type='cluster',cov_kwds={'groups':ev.iso2.astype('category').cat.codes})
print(m.params.round(3).to_dict(), m.pvalues.round(4).to_dict())
m2=smf.logit('I(d406>0) ~ gmst + base_d406',data=ev).fit(disp=0)
print('logit 40.6:',m2.params.round(3).to_dict(), m2.pvalues.round(4).to_dict())
