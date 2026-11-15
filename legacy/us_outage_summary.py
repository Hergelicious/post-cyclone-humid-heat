# Summary of the measured outage-heat analysis (US/PR landfalls 2015-2024, EAGLE-I + hourly ERA5)
import pandas as pd, numpy as np, sys
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
R=pd.read_csv(f'pilot/us_outage_storms_{TAG}.csv'); C=pd.read_pickle(f'pilot/us_outage_curves_{TAG}.pkl'); FS=C['FS']; cur=C['curves']
R=R[R.counties.notna()].copy(); pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
M=1e6
print('storms with qualifying outages',len(R),'of',len(R),'; counties-storms',int(R.counties.sum()))
tot=R[['cust_h','cust_h35','cust_h406','cust_h_hour35','cust_h_warmnight','cust_h_after72','cust_h35_after72']].sum()/M
print('TOTAL (million customer-hours, days 0-30):'); print(tot.round(1).to_string())
print('share of outage customer-hours on HI>=35 days %.0f%%, on HI>=40.6 days %.1f%%, in hours HI>=35 %.0f%%, in warm night hours %.0f%%'%(100*tot.cust_h35/tot.cust_h,100*tot.cust_h406/tot.cust_h,100*tot.cust_h_hour35/tot.cust_h,100*tot.cust_h_warmnight/tot.cust_h))
print('share of all outage customer-hours after 72 h %.0f%%; share of HI>=35-day customer-hours after 72 h %.0f%%'%(100*tot.cust_h_after72/tot.cust_h,100*tot.cust_h35_after72/tot.cust_h35))
cols=['storm','season','wind','counties','peak_out','cust_h','cust_h35','cust_h406','cust_h_hour35','cust_h_warmnight','h_to_50','h_to_90','h_to_99','share_cov_lt50','max_km_to_cmip']
T=R[cols].copy();
for c in ['cust_h','cust_h35','cust_h406','cust_h_hour35','cust_h_warmnight']: T[c]=T[c]/M
T['peak_out']=T.peak_out/1e3; print(T.sort_values('cust_h35',ascending=False).round(2).to_string(index=False))
# ---- attribution: natural-only (15 realisations); share of hot-day customer-hours due to human-induced warming
nat=[c for c in R if c.startswith('E35|nat|')]; mods=sorted(set(c.split('|')[2] for c in nat))
for thr in ('35','406'):
    obs=R[f'E{thr}|obs'].sum(); shares=np.array([1-R[f'E{thr}|nat|{c.split("|",2)[2]}'].sum()/obs for c in nat]) if obs>0 else np.array([np.nan])
    per_model={m:np.mean([1-R[c.replace('E35',f'E{thr}')].sum()/obs for c in nat if c.split('|')[2]==m]) for m in mods} if obs>0 else {}
    print(f'HI>={thr}: observed {obs/M:.1f} M; human share mean {100*shares.mean():.0f}% (5-95% over realisations {100*np.percentile(shares,5):.0f} to {100*np.percentile(shares,95):.0f}%); model means',{k:round(100*v) for k,v in per_model.items()})
# by storm (largest contributors)
top=R.sort_values('E35|obs',ascending=False).head(8)
print('per-storm human share HI>=35 (mean of 15):',{r.storm:round(100*(1-np.mean([r[c] for c in nat])/r['E35|obs'])) if r['E35|obs']>0 else None for _,r in top.iterrows()})
# ---- warming levels (26 models): ratio of hot-day customer-hours to observed, same outages
for thr in ('35','406'):
    obs=R[f'E{thr}|obs'].sum()
    for wl in (1.5,2.0,2.8):
        v=np.array([R[c].sum() for c in R if c.startswith(f'E{thr}|wl{wl}|')])
        print(f'HI>={thr} at {wl} C: median {np.median(v)/M:.1f} M ({np.percentile(v,5)/M:.1f}-{np.percentile(v,95)/M:.1f}), ratio to observed {np.median(v)/obs:.2f}')
# ---- restoration requirement: compression factor f* (duration multiplier) so that warmer-climate hot-day customer-hours equal the reference
storms=list(R.storm)
def agg(clim,thr): return sum(cur[(s,clim,thr)] for s in storms if (s,clim,thr) in cur)
def fstar(curve,target):
    # curve increases with f; find f where curve==target
    if target<=curve[0]: return FS[0]
    if target>=curve[-1]: return np.nan
    return float(np.interp(target,curve,FS))
i1=int(np.argmin(np.abs(FS-1.0)))
for thr in (35,406):
    obs=agg('obs',thr);
    if obs[i1]<=0: continue
    natc=[c.split('|',1)[1] for c in R if c.startswith('E35|nat|')]
    natv=np.array([agg(k,thr)[i1] for k in natc])
    f_nat=fstar(obs,natv.mean()); print(f'HI>={thr}: to bring today\'s hot-day customer-hours down to the natural-only level, outage durations would need to be x{f_nat:.2f} (i.e. {100*(1-f_nat):.0f}% shorter)')
    for wl in (1.5,2.0,2.8):
        mods26=sorted(set(c.split('|')[2] for c in R if c.startswith(f'E35|wl{wl}|')))
        fs=np.array([fstar(agg(f'wl{wl}|{m}',thr),obs[i1]) for m in mods26])
        print(f'  at {wl} C: durations x{np.nanmedian(fs):.2f} ({np.nanpercentile(fs,5):.2f}-{np.nanpercentile(fs,95):.2f}; n={np.isfinite(fs).sum()}) to keep observed hot-day customer-hours')
    # elasticity: hot-day customer-hours vs restoration speed in observed climate
    print('  observed climate, hot-day customer-hours relative to actual at duration x0.5, x0.75, x1.25:',[round(float(np.interp(f,FS,obs)/obs[i1]),2) for f in (0.5,0.75,1.25)])
