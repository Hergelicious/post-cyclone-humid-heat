# Exploratory test: is hazardous heat after landfall associated with slower electricity restoration?
# (1) within county-storm daily panel: share of remaining outage restored each day vs that day's maximum HI,
#     with county-storm and day-since-peak fixed effects; (2) across counties of the same storm: hours from peak to 50% / 90% restored
#     vs mean daily maximum HI in the first five days after the peak, with storm fixed effects and damage proxies.
import pandas as pd, numpy as np, statsmodels.formula.api as smf, sys
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
recs=pd.read_pickle(f'pilot/us_v2_recs_{TAG}.pkl')
P=[]; X=[]
for r in recs:
    Ex=r['Ex']; HI=r['HI']; T=r['T']; tt=r['tt']
    for j,f in enumerate(r['fips']):
        e=Ex[:,j]; ip=int(e[:24*5].argmax()); pk=e[ip]; c=r['cust'][j]
        if pk<=0: continue
        # daily steps after the peak
        prev=pk; d=0
        hs=[]
        while ip+24*(d+1)<len(e) and d<14:
            seg=slice(ip+24*d,ip+24*(d+1)); now=e[ip+24*(d+1)-1]
            if prev<0.05*pk: break
            hmax=np.nanmax(HI[seg,j]); P.append(dict(cs=f"{r['storm']}|{f}",storm=r['storm'],d=d,hmax=hmax,hot=int(hmax>=35),tmin=np.nanmin(T[seg,j]),
                                                    rest=float(np.clip(1-now/prev,0,1)),rem=prev/pk))
            hs.append(hmax); prev=now; d+=1
        after=np.where(e[ip:]<=0.5*pk)[0]; a90=np.where(e[ip:]<=0.1*pk)[0]
        h5=np.nanmean([np.nanmax(HI[ip+24*k:ip+24*(k+1),j]) for k in range(5) if ip+24*(k+1)<=len(e)])
        X.append(dict(storm=r['storm'],fips=f,h50=float(after[0]) if len(after) else np.nan,h90=float(a90[0]) if len(a90) else np.nan,hmax5=h5,
                      frac=pk/c,lcust=np.log(c),dtrack=r['dtrack'][j] if np.isfinite(r['dtrack'][j]) else r['dland'][j],tpeak=tt[ip]))
P=pd.DataFrame(P); X=pd.DataFrame(X)
P.to_csv('pilot/us_restoration_panel.csv',index=False); X.to_csv('pilot/us_restoration_cross.csv',index=False)
def fitp(f,d):
    m=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.storm)[0]}); return m
print('panel rows',len(P),'county-storms',P.cs.nunique(),'storms',P.storm.nunique())
import statsmodels.api as sm
def within(d,xs):
    D=pd.get_dummies(d.d,prefix='d',drop_first=True).astype(float)
    Z=pd.concat([d[['rest']+xs].astype(float),D],axis=1); Z=Z-Z.groupby(d.cs.values).transform('mean')
    m=sm.OLS(Z.rest,Z.drop(columns='rest')).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.storm)[0]}); return m
Pd=P[P.storm.map(P.groupby('storm').hot.max())==1]
for xs,lab in [(['hmax'],'daily max HI (per degC)'),(['hot'],'HI>=35 day'),(['hmax','tmin'],'HI, with night minimum T')]:
    for d_,nm in [(P,'all'),(Pd,'storms with hot days')]:
        m=within(d_,xs); k=xs[0]
        print(f'{nm:22s} {lab:28s} coef {m.params[k]:+.4f} (95% {m.conf_int().loc[k,0]:+.4f} to {m.conf_int().loc[k,1]:+.4f}), p={m.pvalues[k]:.3f}; mean restored per day {d_.rest.mean():.3f}')
Xc=X.dropna(subset=['h50','hmax5']).copy(); Xc['l50']=np.log(Xc.h50.clip(lower=1)); Xc['lfrac']=np.log(Xc.frac); Xc['ldt']=np.log(Xc.dtrack.clip(lower=10))
for y in ('l50',):
    m=fitp(f'{y}~hmax5+lfrac+lcust+ldt+C(storm)',Xc)
    print(f'cross-county {y}: per degC of mean daily max HI {m.params["hmax5"]:+.4f} (95% {m.conf_int().loc["hmax5",0]:+.4f} to {m.conf_int().loc["hmax5",1]:+.4f}), p={m.pvalues["hmax5"]:.3f}, n={int(m.nobs)}  (log hours to 50% restored)')
X9=X.dropna(subset=['h90','hmax5']).copy(); X9['l90']=np.log(X9.h90.clip(lower=1)); X9['lfrac']=np.log(X9.frac); X9['ldt']=np.log(X9.dtrack.clip(lower=10))
m=fitp('l90~hmax5+lfrac+lcust+ldt+C(storm)',X9)
print(f'cross-county l90: per degC {m.params["hmax5"]:+.4f} (95% {m.conf_int().loc["hmax5",0]:+.4f} to {m.conf_int().loc["hmax5",1]:+.4f}), p={m.pvalues["hmax5"]:.3f}, n={int(m.nobs)}')
