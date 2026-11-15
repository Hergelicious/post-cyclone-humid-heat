import pandas as pd, numpy as np, os, statsmodels.formula.api as smf
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'])
lf['clat']=np.round(lf.lat*2)/2; lf['clon']=np.round(lf.lon*2)/2
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean
off=-g.loc[1880:1900].mean()   # proxy offset to pre-industrial
gpi=g+off
print('GISTEMP offset to 1880-1900:',round(off,3),' 2024 above PI:',round(gpi[2024],2),' 2005-2024 mean:',round(gpi.loc[2005:2024].mean(),2))
def hi_c(T,RH):
    F=T*9/5+32; hi=0.5*(F+61+(F-68)*1.2+RH*0.094)
    full=(-42.379+2.04901523*F+10.14333127*RH-.22475541*F*RH-.00683783*F*F-.05481717*RH*RH+.00122874*F*F*RH+.00085282*F*RH*RH-.00000199*F*F*RH*RH)
    return (np.where((hi+F)/2>=80,full,hi)-32)*5/9
def stull(T,RH):
    return T*np.arctan(0.151977*np.sqrt(RH+8.313659))+np.arctan(T+RH)-np.arctan(RH-1.676331)+0.00391838*RH**1.5*np.arctan(0.023101*RH)-4.686035
cache={}
def series(la,lo):
    k=(la,lo)
    if k in cache: return cache[k]
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan)
    s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
    q=s.QV2M/1000; p=s.PS*10; e=q*p/(0.622+0.378*q)
    es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5)); s['RH']=np.clip(100*e/es,1,100)
    s['HI']=hi_c(s.T2M_MAX,s.RH); s['TW']=stull(s.T2M_MAX,s.RH)
    s=s.set_index('date'); cache[k]=s; return s
WL={'1.5':1.5,'2.0':2.0,'2.8':2.8}
rows=[]; beta_cache={}
for r in lf.itertuples():
    s=series(r.clat,r.clon); t0=r.time.normalize()
    w=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30)]
    w21=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=21)]
    rain=s.loc[t0-pd.Timedelta(days=1):t0+pd.Timedelta(days=2),'PRECTOTCORR'].sum()
    d=dict(idx=r.Index,track=r.track_id,season=r.season,iso2=r.iso2,lat=r.lat,lon=r.lon,wind=r.lf_wind,month=t0.month,
           rain4=rain,d35=(w.HI>=35).sum(),d406=(w.HI>=40.6).sum(),tw26=(w.TW>=26).sum(),tw28=(w.TW>=28).sum(),
           hot21=(w21.HI>=35).sum())
    # local scaling of window Tmax on GMST (same calendar window, all years)
    if (r.clat,r.clon,t0.month) not in beta_cache:
        m=s[s.index.month==t0.month]; a=m.groupby(m.index.year).T2M_MAX.mean(); a=a[a.index<=2024]
        beta_cache[(r.clat,r.clon,t0.month)]=np.polyfit(gpi.loc[a.index].values,a.values,1)[0]
    b=beta_cache[(r.clat,r.clon,t0.month)]; d['beta']=b
    if r.season>=2005:
        for k,v in WL.items():
            dT=b*(v-gpi[r.season]); T=w.T2M_MAX+dT
            # constant RH (upper) and RH falling 1 %-point per K of local warming (lower)
            for tag,rh in [('c',w.RH),('d',np.clip(w.RH-1.0*dT,1,100))]:
                d[f'p{k}_{tag}_d406']=(hi_c(T,rh)>=40.6).sum(); d[f'p{k}_{tag}_d35']=(hi_c(T,rh)>=35).sum(); d[f'p{k}_{tag}_tw28']=(stull(T,rh)>=28).sum()
    rows.append(d)
ev=pd.DataFrame(rows); ev['gpi']=ev.season.map(gpi)
ev.to_csv('pilot/power_events_full.csv',index=False)
ev['wet']=ev.rain4>=100; ev['wethot']=ev.wet&(ev.hot21>=7)
ev['period']=pd.cut(ev.season,[1980,2000,2012,2024],labels=['1981-2000','2001-2012','2013-2024'])
print(ev.groupby('period',observed=True).agg(n=('track','size'),wet=('wet','mean'),wethot=('wethot','mean'),tw28=('tw28','mean')).round(3))
m=smf.logit('wethot ~ gpi',data=ev.assign(wethot=ev.wethot.astype(int))).fit(disp=0,cov_type='cluster',cov_kwds={'groups':ev.iso2.astype('category').cat.codes})
print('wet-then-hot OR per +0.5C:',round(np.exp(.5*m.params.gpi),2),np.exp(.5*m.conf_int().loc['gpi']).round(2).tolist(),'p',round(m.pvalues.gpi,4))
m=smf.logit('wet ~ gpi',data=ev.assign(wet=ev.wet.astype(int))).fit(disp=0,cov_type='cluster',cov_kwds={'groups':ev.iso2.astype('category').cat.codes})
print('heavy storm rain OR per +0.5C:',round(np.exp(.5*m.params.gpi),2),'p',round(m.pvalues.gpi,4))
print('median local beta (K per K GMST):',round(ev.beta.median(),2))
a=ev[ev.season>=2005]
print('analogues 2005-2024:',len(a),' observed mean d406',round(a.d406.mean(),2),' d35',round(a.d35.mean(),2),' tw28',round(a.tw28.mean(),2))
for k in WL: print(k, {m:(round(a[f'p{k}_c_{m}'].mean(),2),round(a[f'p{k}_d_{m}'].mean(),2)) for m in ['d406','d35','tw28']})
