import pandas as pd, numpy as np, json, glob, statsmodels.formula.api as smf
cells=pd.read_csv('pilot/wbt_cells.csv'); key={(a,b):i for i,(a,b) in enumerate(zip(cells.wlat,cells.wlon))}
months=[(y,m) for y in range(1981,2023) for m in range(1,13)]
arrs=[np.load(f'data/wbt/daily/{y}-{m:02d}.npy') for y,m in months]
TW=np.concatenate(arrs,0); dates=pd.date_range('1981-01-01','2022-12-31'); assert len(dates)==TW.shape[0]
np.save('data/wbt/tw_daily_cells.npy',TW)
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']); lf=lf[lf.season<=2022].reset_index()
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g+(-g.loc[1880:1900].mean())
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
doy=np.array(dates.dayofyear); yr=np.array(dates.year)
rows=[]; bg=[]
for r in lf.itertuples():
    c=key[(r.wlat,r.wlon)]; s=TW[:,c]; t0=np.searchsorted(dates.values,r.time.normalize().to_datetime64())
    w=s[t0+1:t0+31]
    # same calendar window in every year (background), keyed by start day-of-year
    d0=doy[t0]; off=(doy-d0)%366; inwin=(off>=1)&(off<=30)
    yy=pd.Series(s[inwin]).groupby(yr[inwin])
    a26=yy.apply(lambda x:(x>=26).mean()*30); a28=yy.apply(lambda x:(x>=28).mean()*30)
    other=a26.index!=r.season
    rows.append(dict(idx=r.index,season=r.season,iso2=r.iso2,lat=r.lat,lon=r.lon,cell=c,wind=r.lf_wind,
        tw26=(w>=26).sum(),tw28=(w>=28).sum(),twmax=np.nanmax(w),c26=a26[other].mean(),c28=a28[other].mean()))
    for y_,v26,v28 in zip(a26.index,a26.values,a28.values):
        bg.append(dict(idx=r.index,cell=c,season=y_,iso2=r.iso2,b26=v26,b28=v28))
ev=pd.DataFrame(rows); bgd=pd.DataFrame(bg)
for d in (ev,bgd):
    d['inc']=d.iso2.map(inc).fillna('NA'); d['lmic']=d.inc.isin(['LIC','LMC','UMC']); d['gpi']=d.season.map(gpi)
ev['reg']=(np.round(ev.lat/2)*2).astype(str)+'_'+(np.round(ev.lon/2)*2).astype(str)
ev['period']=pd.cut(ev.season,[1980,2000,2017,2022],labels=['1981-2000','2001-2017','2018-2022'])
ev.to_csv('pilot/era5_events.csv',index=False); bgd.to_csv('pilot/era5_background.csv',index=False)
print('ERA5 events',len(ev))
print(ev.groupby('period',observed=True).agg(n=('cell','size'),any28=('tw28',lambda x:(x>0).mean()),any26=('tw26',lambda x:(x>0).mean()),tw26=('tw26','mean'),tw28=('tw28','mean')).round(3))
print(ev.groupby(['lmic','period'],observed=True).agg(n=('cell','size'),any28=('tw28',lambda x:(x>0).mean()),tw26=('tw26','mean')).round(3))
cl=lambda d: dict(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes})
res={}
for sub,nm in [(ev,'all'),(ev[ev.lmic],'LMIC'),(ev[~ev.lmic],'HIC')]:
    for y in ['tw26','tw28']:
        m=smf.ols(f'{y} ~ gpi + C(reg)',data=sub).fit(**cl(sub)); res[(nm,y)]=(m.params.gpi,*m.conf_int().loc['gpi'],m.pvalues.gpi)
        print(nm,y,'days per +1C:',round(m.params.gpi,2),m.conf_int().loc['gpi'].round(2).tolist(),'p',round(m.pvalues.gpi,4))
ev['hit28']=(ev.tw28>0).astype(int)
m=smf.logit('hit28 ~ gpi + c28',data=ev).fit(disp=0,**cl(ev)); print('any Tw>=28 OR per +0.5C:',round(np.exp(.5*m.params.gpi),2),np.exp(.5*m.conf_int().loc['gpi']).round(2).tolist())
# background (placebo) sensitivity: same windows in all years, location FE
bgd['loc']=bgd.idx.astype(str)
for y in ['b26','b28']:
    mm=smf.ols(f'{y} ~ gpi',data=bgd.assign(**{y:bgd[y]-bgd.groupby('loc')[y].transform('mean'),'gpi':bgd.gpi-bgd.groupby('loc').gpi.transform('mean')})).fit(**cl(bgd))
    print('background',y,'days per +1C:',round(mm.params.gpi,2),mm.conf_int().loc['gpi'].round(2).tolist())
ev['diff26']=ev.tw26-ev.c26; ev['diff28']=ev.tw28-ev.c28
print('post-landfall minus climatology: Tw26',round(ev.diff26.mean(),2),' Tw28',round(ev.diff28.mean(),2))
pd.to_pickle(res,'pilot/era5_res.pkl')
