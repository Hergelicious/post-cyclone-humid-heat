import pandas as pd, numpy as np, json, statsmodels.formula.api as smf
P=np.load('data/wbt/era5_precip_daily_cells.npy'); TW=np.load('data/wbt/tw_daily_cells.npy')
dP=pd.date_range('1981-01-01',periods=P.shape[0]); dT=pd.date_range('1981-01-01','2022-12-31')
cells=pd.read_csv('pilot/wbt_cells.csv'); key={(a,b):i for i,(a,b) in enumerate(zip(cells.wlat,cells.wlon))}
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']); pw=pd.read_csv('pilot/power_events_exposure.csv')
lf=lf.reset_index().rename(columns={'index':'idx'}).merge(pw[['idx','hot21','rain4','lmic','gpi','iso2']].rename(columns={'iso2':'iso2p'}),on='idx')
lf=lf[lf.season<=2021]
# local 4-day totals distribution per cell
R4=pd.DataFrame(P).rolling(4).sum().values
p99=np.nanpercentile(R4,99,axis=0)
rows=[]
for r in lf.itertuples():
    c=key[(r.wlat,r.wlon)]; t=np.searchsorted(dP.values,r.time.normalize().to_datetime64())
    rain=P[t-1:t+3,c].sum(); tt=np.searchsorted(dT.values,r.time.normalize().to_datetime64())
    tw26=(TW[tt+1:tt+22,c]>=26).sum()
    rows.append(dict(idx=r.idx,season=r.season,iso2=r.iso2,lmic=r.lmic,gpi=r.gpi,rain_e5=rain,wet_e5=rain>=100,wetp_e5=rain>=p99[c],
        hot21=r.hot21,tw26_21=tw26,rain_pw=r.rain4))
E=pd.DataFrame(rows)
E['wethot_e5']=(E.wet_e5&(E.hot21>=7)).astype(int); E['wethot_e5p']=(E.wetp_e5&(E.hot21>=7)).astype(int)
E['wethot_e5tw']=(E.wet_e5&(E.tw26_21>=7)).astype(int); E['wet_pw']=E.rain_pw>=100
E['period']=pd.cut(E.season,[1980,2000,2012,2021],labels=['1981-2000','2001-2012','2013-2021'])
print('ERA5 vs POWER 4-day rain: Spearman',round(E[['rain_e5','rain_pw']].corr('spearman').iloc[0,1],2),' agreement on wet',round((E.wet_e5==E.wet_pw).mean(),2))
print(E.groupby(['lmic','period'],observed=True).agg(n=('idx','size'),wet_e5=('wet_e5','mean'),wet_pw=('wet_pw','mean'),wetp=('wetp_e5','mean'),wh_e5=('wethot_e5','mean'),wh_e5p=('wethot_e5p','mean'),wh_tw=('wethot_e5tw','mean')).round(3))
cl=lambda d: dict(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes})
out=[]
for y in ['wet_e5','wetp_e5','wethot_e5','wethot_e5p','wethot_e5tw']:
    for lm,nm in [(True,'LMIC'),(False,'HIC')]:
        s=E[E.lmic==lm].assign(yy=lambda d: d[y].astype(int))
        m=smf.logit('yy ~ gpi',data=s).fit(disp=0,**cl(s)); o=np.exp(.5*np.r_[m.params.gpi,m.conf_int().loc['gpi']])
        out.append((y,nm,len(s),*o.round(2),round(m.pvalues.gpi,4)))
R=pd.DataFrame(out,columns=['outcome','group','n','OR_per_0.5C','lo','hi','p']); print(R.to_string()); R.to_csv('pilot/rain_check.csv',index=False)
E.to_csv('pilot/rain_events.csv',index=False)
