import pandas as pd, numpy as np, xarray as xr, json, statsmodels.formula.api as smf
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index(drop=True)
pairs=pd.read_csv('data/gsdm/pairs.csv',dtype={'code':str}).sort_values('km').drop_duplicates('lf_idx')
dates=pd.date_range('1981-01-01','2020-12-31'); doy=np.array(dates.dayofyear); yr=np.array(dates.year)
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
cache={}
def ser(code):
    if code not in cache:
        d=xr.open_dataset(f'data/gsdm/nc/{code}.nc'); s=d.WBT.values.astype(float); s[d.flag.values==1]=np.nan; cache[code]=s
    return cache[code]
rows=[]
for p in pairs.itertuples():
    e=lf.iloc[p.lf_idx]; s=ser(p.code); t0=e.time.normalize()
    off=(doy-t0.dayofyear)%366; m=(off>=1)&(off<=30)
    # station-specific 90th percentile of warm-season windows in 1981-2010
    base=s[m&(yr<=2010)]; p90=np.nanpercentile(base,90) if np.isfinite(base).sum()>200 else np.nan
    df=pd.DataFrame({'y':yr[m],'v':s[m]}).groupby('y').v
    for y,grp in df:
        if grp.notna().sum()<20: continue
        f=30/grp.notna().sum()
        rows.append(dict(win=p.lf_idx,code=p.code,iso2=e.iso2,season=y,event=int(y==e.season),
            d26=(grp>=26).sum()*f,d28=(grp>=28).sum()*f,dp90=(grp>=p90).sum()*f,mean=grp.mean()))
B=pd.DataFrame(rows); B['gpi']=B.season.map(gpi); B['lmic']=B.iso2.map(inc).isin(['LIC','LMC','UMC'])
print('station windows:',B.win.nunique(),'stations',B.code.nunique(),'window-years',len(B))
cl=lambda d: dict(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes})
out=[]
for y in ['d26','d28','dp90','mean']:
    m=smf.ols(f'{y} ~ gpi + event + C(win)',data=B).fit(**cl(B))
    out.append(('all',y,m.params.gpi,*m.conf_int().loc['gpi'],m.pvalues.gpi,m.params.event,*m.conf_int().loc['event']))
    s=B[B.lmic]; m=smf.ols(f'{y} ~ gpi + event + C(win)',data=s).fit(**cl(s))
    out.append(('LMIC',y,m.params.gpi,*m.conf_int().loc['gpi'],m.pvalues.gpi,m.params.event,*m.conf_int().loc['event']))
R=pd.DataFrame(out,columns=['grp','y','per_C','lo','hi','p','event_effect','e_lo','e_hi']).round(3); print(R.to_string())
R.to_csv('pilot/station_background.csv',index=False)
