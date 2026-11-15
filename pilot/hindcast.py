import pandas as pd, numpy as np
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'])
cache={}
def series(la,lo):
    k=(la,lo)
    if k not in cache:
        s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan)
        s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
        q=s.QV2M/1000; p=s.PS*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5)); s['RH']=np.clip(100*e/es,1,100)
        s['HI']=hi_c(s.T2M_MAX,s.RH); cache[k]=s.set_index('date')
    return cache[k]
# Hindcast: for each 2005-2024 landfall, take its analogue window from the early period? Instead: fit local beta on 1981-2004 only,
# then shift each 1981-2004 same-calendar window (year y) to the GMST of the test year and compare predicted mean d406/d35 for 2005-2024 with observed.
rows=[]
for r in lf[lf.season>=2005].itertuples():
    la=np.round(r.lat*2)/2; lo=np.round(r.lon*2)/2; s=series(la,lo); t0=r.time.normalize(); m=t0.month
    mm=s[(s.index.month==m)&(s.index.year<=2004)]; a=mm.groupby(mm.index.year).T2M_MAX.mean()
    b=np.polyfit(gpi.loc[a.index].values,a.values,1)[0]
    obs=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30)]
    doy=t0.dayofyear; early=s[(s.index.year<=2004)]; off=(early.index.dayofyear-doy)%366; w=early[(off>=1)&(off<=30)]
    pred35=[];pred406=[];base35=[];base406=[]
    for y,wy in w.groupby(w.index.year):
        dT=b*(gpi[r.season]-gpi[y]); T=wy.T2M_MAX+dT
        for tag,rh in [('c',wy.RH),('d',np.clip(wy.RH-dT,1,100))]:
            h=hi_c(T,rh); (pred35 if tag=='c' else base35).append((h>=35).sum()); (pred406 if tag=='c' else base406).append((h>=40.6).sum())
        base_raw=(wy.HI>=35).sum()
    rows.append(dict(idx=r.Index,season=r.season,obs35=(obs.HI>=35).sum(),obs406=(obs.HI>=40.6).sum(),
        pred35_c=np.mean(pred35),pred35_d=np.mean(base35),pred406_c=np.mean(pred406),pred406_d=np.mean(base406),
        clim35=(w.HI>=35).groupby(w.index.year).sum().mean(),clim406=(w.HI>=40.6).groupby(w.index.year).sum().mean()))
H=pd.DataFrame(rows); H.to_csv('pilot/hindcast.csv',index=False)
print('n',len(H))
for k in ['35','406']:
    print(f'HI>={k}: observed mean {H["obs"+k].mean():.2f} | no-warming climatology {H["clim"+k].mean():.2f} | delta const RH {H["pred"+k+"_c"].mean():.2f} | delta RH-1/K {H["pred"+k+"_d"].mean():.2f}')
    print('   bias const RH',round(H["pred"+k+"_c"].mean()-H["obs"+k].mean(),2),' bias RH-1/K',round(H["pred"+k+"_d"].mean()-H["obs"+k].mean(),2),' corr',round(np.corrcoef(H["pred"+k+"_c"],H["obs"+k])[0,1],2))
