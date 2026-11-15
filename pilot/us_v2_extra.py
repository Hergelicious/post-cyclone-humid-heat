# Extra outage analyses: (1) coverage-stratified simultaneous exposure and human share (documented state coverage, record
# completeness while out); (2) per-storm median shares; (3) restoration requirement for 2018-2024 landfalls; (4) severity of
# repeated exposure (customers through >=k hazardous peaks, consecutive hazardous days, peak plus warm night) under
# natural-only and warming-level climates.
import pandas as pd, numpy as np
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl'); R=pd.read_csv('pilot/us_v2_storms_main.csv'); R=R[R.counties.notna()]
C=pd.read_pickle('pilot/us_v2_curves_main.pkl'); FS=C['FS']; cur=C['curves']; i1=int(np.argmin(np.abs(FS-1)))
clims=list(recs[0]['cty'].keys()); nat=[k for k in clims if k.startswith('nat|')]; wl={w:[k for k in clims if k.startswith(f'wl{w}|')] for w in (1.5,2.0,2.8)}
M=1e6
# ---- (1) strata
def stratum(r):
    cov=r['cov']; comp=r['nrep']
    return dict(cov=np.where(np.isnan(cov),'undocumented (2015-2017, 2023-2024)',np.where(cov>=0.9,'documented >= 90%',np.where(cov>=0.5,'documented 50-90%','documented < 50%'))),
                comp=np.where(np.isnan(comp),'none',np.where(comp>=0.9,'records >= 90% of quarter-hours while out',np.where(comp>=0.5,'records 50-90%','records < 50%'))))
rows=[]
for r in recs:
    st=stratum(r); win=((r['tt']>=-24)&(r['tt']<720))[:,None]; E=(r['Ex']*win).sum(0)
    for j in range(len(r['fips'])):
        d=dict(storm=r['storm'],season=r['season'],cov=st['cov'][j],comp=st['comp'][j],cust_h=E[j],obs=r['cty']['obs']['hour35'][j],pk=r['cty']['obs']['peak35'][j])
        for k in nat: d[k]=r['cty'][k]['hour35'][j]
        rows.append(d)
D=pd.DataFrame(rows)
print('== coverage strata (hourly HI>=35)')
for col in ('cov','comp'):
    for g,X in D.groupby(col):
        o=X.obs.sum(); n=np.array([X[k].sum() for k in nat]); hs=1-n/o if o>0 else np.full(len(n),np.nan)
        print(f'  {g:45s} share of outage hours {100*X.cust_h.sum()/D.cust_h.sum():5.1f}% | HI>=35 share {100*o/X.cust_h.sum():4.1f}% | human {100*np.nanmean(hs):3.0f}% ({100*np.nanpercentile(hs,5):.0f} to {100*np.nanpercentile(hs,95):.0f})')
print('  completeness quartiles (outage-hour weighted):',np.round(np.percentile(np.concatenate([r['nrep'] for r in recs]),[25,50,75]),2))
# ---- (2) per storm
R['sh35']=R['hour35|obs']/R.cust_h; R['shday']=R['day35|obs']/R.cust_h
big=R[R.cust_h>=1e6]
print('== per-storm share of outage hours with HI>=35: median %.1f%% (IQR %.1f-%.1f), n=%d; storms >=1 M customer-hours median %.1f%% (n=%d); hot-day measure median %.0f%%'%(
    100*R.sh35.median(),100*R.sh35.quantile(.25),100*R.sh35.quantile(.75),len(R),100*big.sh35.median(),len(big),100*big.shday.median()))
# ---- (3) restoration requirement for 2018-2024
def fstar(curve,target):
    if target<=curve[0]: return FS[0]
    if target>=curve[-1]: return np.nan
    return float(np.interp(target,curve,FS))
for lab,sel in (('2018-2024',R.season>=2018),('without Irma',R.storm!='Irma')):
    S=list(R[sel].storm)
    for m in ('hour35','peak35'):
        agg=lambda k: sum(cur[(s,k,m)] for s in S if (s,k,m) in cur); obs=agg('obs')
        out=[]
        for w in (1.5,2.0,2.8):
            fs=np.array([fstar(agg(f'wl{w}|{k.split("|")[1]}'),obs[i1]) for k in wl[w]]); out.append(f'{w}: x{np.nanmedian(fs):.2f} ({np.nanpercentile(fs,5):.2f}-{np.nanpercentile(fs,95):.2f})')
        print(f'== restoration requirement {lab} {m}:','; '.join(out))
# ---- (4) severity distribution
names=['>=1 peak','>=2 peaks','>=3 peaks','>=4 peaks','>=5 peaks','2 consecutive hazardous days','3 consecutive hazardous days','peak followed by warm night']
def tot(k): return np.sum([r['cty'][k]['sev'].sum(0) for r in recs],0)
O=tot('obs'); Nn=np.array([tot(k) for k in nat]); Wv={w:np.array([tot(k) for k in wl[w]]) for w in wl}
S=[]
for i,nm in enumerate(names):
    d=dict(measure=nm,observed_M=O[i]/M,nat_mean_M=Nn[:,i].mean()/M,human_share=1-Nn[:,i].mean()/O[i] if O[i]>0 else np.nan,
           hs_lo=np.percentile(1-Nn[:,i]/O[i],5) if O[i]>0 else np.nan,hs_hi=np.percentile(1-Nn[:,i]/O[i],95) if O[i]>0 else np.nan)
    for w in wl: d[f'x{w}']=np.median(Wv[w][:,i])/O[i] if O[i]>0 else np.nan
    S.append(d)
S=pd.DataFrame(S); S.to_csv('pilot/us_v2_severity.csv',index=False); print('== severity (customers, million)'); print(S.round(3).to_string())
D.to_csv('pilot/us_v2_county_strata.csv',index=False)
