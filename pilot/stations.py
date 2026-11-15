import pandas as pd, numpy as np, xarray as xr, json
import statsmodels.formula.api as smf
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index(drop=True)
pairs=pd.read_csv('data/gsdm/pairs.csv',dtype={'code':str})
pairs=pairs.sort_values('km').drop_duplicates('lf_idx')
dates=pd.date_range('1981-01-01','2020-12-31')
cache={}
def ser(code):
    if code not in cache:
        d=xr.open_dataset(f'data/gsdm/nc/{code}.nc'); s=pd.Series(d.WBT.values,index=dates); f=pd.Series(d.flag.values,index=dates)
        s[f==1]=np.nan  # original/corrected observations only
        cache[code]=s
    return cache[code]
rows=[]
for p in pairs.itertuples():
    e=lf.iloc[p.lf_idx]
    if e.season>2020: continue
    s=ser(p.code); t0=e.time.normalize()
    w=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30)]
    if w.notna().sum()<20: continue
    frac=30/w.notna().sum()
    doy=t0.dayofyear
    clim=s[(s.index.year!=t0.year)&(((s.index.dayofyear-doy)%366)>=1)&(((s.index.dayofyear-doy)%366)<=30)]
    cy=clim.groupby(clim.index.year)
    rows.append(dict(lf_idx=p.lf_idx,code=p.code,km=p.km,season=e.season,iso2=e.iso2,
        d26=(w>=26).sum()*frac,d28=(w>=28).sum()*frac,twmax=w.max(),
        c26=cy.apply(lambda x:(x>=26).mean()*30).mean(),c28=cy.apply(lambda x:(x>=28).mean()*30).mean()))
st=pd.DataFrame(rows)
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
st['inc']=st.iso2.map(inc); st['lmic']=st.inc.isin(['LIC','LMC','UMC'])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; st['gmst']=st.season.map(g)
st['period']=pd.cut(st.season,[1980,2000,2020],labels=['1981-2000','2001-2020'])
st.to_csv('pilot/station_events.csv',index=False)
print(len(st),'events with station obs;',st.code.nunique(),'stations; LMIC share',round(st.lmic.mean(),2))
print(st.groupby('period',observed=True).agg(n=('code','size'),any28=('d28',lambda x:(x>0).mean()),d26=('d26','mean'),d28=('d28','mean'),c26=('c26','mean')).round(3))
for y,nm in [('d26','Tw>=26'),('d28','Tw>=28')]:
    m=smf.ols(f'{y} ~ gmst + C(code)',data=st).fit(cov_type='cluster',cov_kwds={'groups':st.iso2.astype('category').cat.codes})
    print(nm,'station FE, days per +1C GMST:',round(m.params.gmst,2),m.conf_int().loc['gmst'].round(2).tolist(),'p',round(m.pvalues.gmst,4))
st['diff26']=st.d26-st.c26
print('post-landfall minus same-window climatology (days Tw>=26): mean',round(st.diff26.mean(),2),'median',round(st.diff26.median(),2))
