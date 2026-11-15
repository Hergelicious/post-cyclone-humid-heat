import pandas as pd, numpy as np, statsmodels.formula.api as smf, itertools, pickle
src=open('pilot/reconcile.py').read()
exec(src[:src.index('# ---------- 1.')])
rng=np.random.default_rng(5)
out={}
# ---------- A. hazard trend robustness
ev=pd.read_csv('pilot/power_events_exposure.csv'); ev['reg']=(np.round(ev.lat/2)*2).astype(str)+'_'+(np.round(ev.lon/2)*2).astype(str)
lfa=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']); ev['month']=lfa.time.dt.month.values[ev.idx]
rows=[]
def fit(d,f='d35 ~ gpi + C(reg)',cov='country'):
    if cov=='country': m=smf.ols(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes.values})
    else: m=smf.ols(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':np.column_stack([d.iso2.astype('category').cat.codes.values,d.season.values])})
    ci=m.conf_int().loc['gpi']; return m.params.gpi,ci[0],ci[1],len(d)
for y in ['d35','d406']:
    rows.append((y,'baseline (country cluster)',*fit(ev,f'{y} ~ gpi + C(reg)')))
    rows.append((y,'two-way cluster (country x year)',*fit(ev,f'{y} ~ gpi + C(reg)','two')))
    rows.append((y,'+ calendar month FE',*fit(ev,f'{y} ~ gpi + C(reg) + C(month)')))
    yrs=ev.season.unique(); bs=[]
    for _ in range(400):
        pick=rng.choice(yrs,len(yrs)); s=pd.concat([ev[ev.season==k] for k in pick]); bs.append(smf.ols(f'{y} ~ gpi + C(reg)',data=s).fit().params.gpi)
    b=fit(ev,f'{y} ~ gpi + C(reg)')[0]; rows.append((y,'year-block bootstrap',b,np.percentile(bs,2.5),np.percentile(bs,97.5),len(ev)))
    for c in ['CN','IN','VN','PH','BD']:
        rows.append((y,f'excluding {c}',*fit(ev[ev.iso2!=c],f'{y} ~ gpi + C(reg)')))
    rows.append((y,'excluding all five',*fit(ev[~ev.iso2.isin(['CN','IN','VN','PH','BD'])],f'{y} ~ gpi + C(reg)')))
TR=pd.DataFrame(rows,columns=['outcome','specification','beta','lo','hi','n']); print(TR.round(2).to_string()); TR.to_csv('pilot/robust_trend.csv',index=False)
# ---------- B. exposure: unique calibrated person-days by year and country (100 km, HI>=40.6)
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[R.foot=='100']
rp406=0.953
U={}
for row in R.itertuples():
    s=series(*row.cell); m=s['HI'][row.t0+1:row.t0+31]>=40.6; scale=rp406
    h=hourly(*row.cell,row.t0) if row.season>=2001 else None
    if h is not None: m=h>=40.6; scale=1.0
    dd=U.setdefault(row.cell,{})
    for di in np.nonzero(m)[0]:
        k=row.t0+1+di
        if row['pop'] if False else True: pass
        v=row.pop*scale
        if v>dd.get(k,(0,None,None))[0]: dd[k]=(v,row.iso2,row.lmic)
uq=pd.DataFrame([(yr[k],v,iso,lm) for cell,dd in U.items() for k,(v,iso,lm) in dd.items()],columns=['year','pd','iso2','lmic'])
uq.to_csv('pilot/robust_unique_celldays.csv',index=False)
L=uq[uq.lmic]; ann=L.groupby('year').pd.sum().reindex(range(1981,2025),fill_value=0)/1e6
ann.to_csv('pilot/robust_annual_lmic.csv')
def ratio(a,late=(2018,2024),early=(1981,2000)): return a.loc[late[0]:late[1]].mean()/a.loc[early[0]:early[1]].mean()
def diff(a,late=(2018,2024),early=(1981,2000)): return a.loc[late[0]:late[1]].mean()-a.loc[early[0]:early[1]].mean()
eq=[(1981,1991),(1992,2002),(2003,2013),(2014,2024)]
print('equal periods (M person-days/yr):',[round(ann.loc[a:b].mean(),0) for a,b in eq])
print('10-yr moving mean 1990, 2000, 2010, 2020:',ann.rolling(10).mean().loc[[1990,2000,2010,2020,2024]].round(0).tolist())
bsr=[];bse=[];bsd=[]
for _ in range(2000):
    y1=rng.choice(range(1981,2001),20); y2=rng.choice(range(2018,2025),7); bsr.append(ann.loc[y2].mean()/ann.loc[y1].mean()); bsd.append(ann.loc[y2].mean()-ann.loc[y1].mean())
    a=rng.choice(range(1981,1992),11); b=rng.choice(range(2014,2025),11); bse.append(ann.loc[b].mean()/ann.loc[a].mean())
print('year-block ratio late/early',round(ratio(ann),2),np.percentile(bsr,[2.5,97.5]).round(2),' diff',round(diff(ann),0),np.percentile(bsd,[2.5,97.5]).round(0))
print('equal-length ratio 2014-24 / 1981-91',round(ann.loc[2014:].mean()/ann.loc[:1991].mean(),2),np.percentile(bse,[2.5,97.5]).round(2))
lo=[]
for c in ['CN','IN','VN','PH','BD','ALL5']:
    s=L[~L.iso2.isin(['CN','IN','VN','PH','BD'] if c=='ALL5' else [c])].groupby('year').pd.sum().reindex(range(1981,2025),fill_value=0)/1e6
    lo.append((c,s.loc[1981:2000].mean(),s.loc[2018:2024].mean(),ratio(s),s.loc[2014:].mean()/s.loc[:1991].mean()))
LO=pd.DataFrame(lo,columns=['excluded','early','late','ratio','ratio_equal']); print(LO.round(2).to_string()); LO.to_csv('pilot/robust_loo_exposure.csv',index=False)
# per-landfall and per-footprint-person rates
lfl=R[R.lmic].groupby('idx').agg(season=('season','first'),pd=('pd406','sum'),pop=('pop','sum'))
lfl['per']=pd.cut(lfl.season,[1980,2000,2017,2024],labels=['E','M','L'])
rt=lfl.groupby('per',observed=True).agg(n=('pd','size'),pd_per_lf=('pd','mean'),pop_per_lf=('pop','mean'),pd=('pd','sum'),pop=('pop','sum')); rt['days_per_person']=rt.pd/rt['pop']
print(rt.round(2)); rt.to_csv('pilot/robust_rates.csv')
# ---------- C. calibration ratios by decade / country / humidity
cal=[]
for row in R[R.season>=2001].itertuples():
    h=hourly(*row.cell,row.t0)
    if h is None: continue
    s=series(*row.cell); w=s['HI'][row.t0+1:row.t0+31]; rh=s['RH'][row.t0+1:row.t0+31].mean()
    cal.append((row.season,row.iso2,rh,row.pop,(w>=40.6).sum(),(h>=40.6).sum(),(w>=35).sum(),(h>=35).sum()))
C=pd.DataFrame(cal,columns=['season','iso2','rh','pop','a406','h406','a35','h35']); C['dec']=np.where(C.season<=2012,'2001-2012','2013-2024'); C['rhq']=pd.qcut(C.rh,3,labels=['dry','mid','humid'])
def rr(g): return pd.Series(dict(n=len(g),pd406=(g.h406*g['pop']).sum()/(g.a406*g['pop']).sum(),d406=g.h406.sum()/g.a406.sum(),d35=g.h35.sum()/g.a35.sum()))
top=C.iso2.value_counts().index[:6]
cr=pd.concat([C.groupby('dec').apply(rr),C.groupby('rhq',observed=True).apply(rr),C[C.iso2.isin(top)].groupby('iso2').apply(rr)])
print(cr.round(3)); cr.to_csv('pilot/robust_calibration.csv')
