# Exploratory tests for alternative framings (13 Nov 2026): timing of heat after landfall, restoration-equivalent time, seasonal timing of landfalls
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
P=pd.read_csv('pilot/recon_daily_profile.csv').pivot(index='day',columns='period')
print('== A. share of landfalls with HI>=35 / >=40.6 by day (landfall cell)')
print(P.loc[[-10,-5,-2,-1,0,1,2,3,5,7,14,21,30]].round(3).to_string())
per=list(P['h35'].columns); e,l=per[0],per[-1]
print('== C. restoration-equivalent time: outage of D days in',e,'vs days in',l,'for same cumulative hazardous days')
for v in ['h35','h406']:
    ce=P[v][e].loc[1:30].cumsum(); cl=P[v][l].loc[1:30].cumsum()
    out=[]
    for D in [3,7,14,30]:
        tgt=ce.loc[D]; k=np.interp(tgt,cl.values,cl.index.values) if tgt<=cl.iloc[-1] else np.nan
        out.append((D,round(tgt,2),round(float(k),1)))
    print(v,'(D early, hazardous days, equivalent D late):',out)
# ---- B. seasonal timing
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False); L['basin']=L.basin.replace('','NA')
def clim(la,lo):
    f=f'pilot/power/{la}_{lo}.csv'
    if not os.path.exists(f): return None
    s=pd.read_csv(f).replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; ee=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    idx=pd.date_range('1981-01-01',periods=len(s)); hi=pd.Series(hi_c(s.T2M_MAX.values,np.clip(100*ee/es,1,100)),idx)
    h=hi['1981':'2000']; out={}
    for thr in (35,40.6):
        c=(h>=thr).groupby(h.index.dayofyear).mean().reindex(range(1,367)).interpolate().bfill().ffill().values
        c=np.convolve(np.r_[c[-7:],c,c[:7]],np.ones(15)/15,'valid')          # 15-day smoothing, circular
        cc=np.r_[c,c[:31]]; out[thr]=np.array([cc[d+1:d+31].sum() for d in range(366)])  # expected hazardous days in days 1-30 for landfall on doy d+1
    return out
pool={b:g.time.dt.dayofyear.values for b,g in L.groupby('basin')}
rows=[]
for r in L.itertuples():
    la=np.round(r.lat*2)/2; lo=np.round(r.lon*2)/2; c=clim(la,lo)
    if c is None:
        la=np.round(r.wlat*2)/2; lo=np.round(r.wlon*2)/2; c=clim(la,lo)
    if c is None: continue
    d=r.time.dayofyear-1
    rows.append(dict(season=r.season,basin=r.basin,iso2=r.iso2,act35=c[35][d],exp35=c[35][pool[r.basin]-1].mean(),act406=c[40.6][d],exp406=c[40.6][pool[r.basin]-1].mean(),doy=d+1))
B=pd.DataFrame(rows); B['tim35']=B.act35-B.exp35; B['tim406']=B.act406-B.exp406; B.to_csv('pilot/explore_seasonality.csv',index=False)
print('== B. fixed 1981-2000 climate: expected hazardous days at the actual landfall date (act), at basin-typical dates (exp = place effect), difference (timing effect)'); print('n',len(B))
B['per']=pd.cut(B.season,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024'])
print(B.groupby('per')[['act35','exp35','tim35','act406','exp406','tim406']].mean().round(2).to_string())
import statsmodels.formula.api as smf
gp=pd.read_csv('pilot/ml_train.csv')[['season','gmst']].drop_duplicates().set_index('season').gmst
B['yr']=B.season-1981
for v in ['act35','exp35','tim35','tim406']:
    m=smf.ols(f'{v}~yr',B).fit(cov_type='cluster',cov_kwds={'groups':B.season}); print(v,'trend per decade %.2f (%.2f to %.2f) p=%.3f'%(m.params.yr*10,*(m.conf_int().loc['yr']*10),m.pvalues.yr))
print('by basin, timing effect HI>=35, early vs late:'); print(B[B.per!='2001-2017'].groupby(['basin','per']).tim35.agg(['mean','count']).unstack().round(2).to_string())
print('mean landfall day of year by basin and period:'); print(B.groupby(['basin','per']).doy.median().unstack().round(0).to_string())
