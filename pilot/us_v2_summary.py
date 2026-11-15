# Summary of version-2 outage metrics: totals, human share, warming levels, restoration requirement, prolonged outages,
# subsets (2018-2024, without Irma, data quality, distance to track, distance to CMIP6 landfall cell).
import pandas as pd, numpy as np, sys
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
R=pd.read_csv(f'pilot/us_v2_storms_{TAG}.csv'); R=R[R.counties.notna()].copy()
C=pd.read_pickle(f'pilot/us_v2_curves_{TAG}.pkl'); FS=C['FS']; cur=C['curves']; i1=int(np.argmin(np.abs(FS-1)))
nat=sorted({c.split('|',2)[2] for c in R if c.startswith('hour35|nat|')}); wl={w:sorted({c.split('|')[2] for c in R if c.startswith(f'hour35|wl{w}|')}) for w in (1.5,2.0,2.8)}
M=1e6; out={}
print(f'[{TAG}] landfalls with affected counties {len(R)}; county-storms {int(R.counties.sum())}; customer-hours {R.cust_h.sum()/M:.0f} M')
for m in ['day35','day406','hour35','hour406','peak35','peak406','night26']:
    o=R[f'{m}|obs'].sum(); n=np.array([R[f'{m}|nat|{k}'].sum() for k in nat]); sh=1-n/o if o>0 else np.full(len(n),np.nan)
    unit='M customer-days' if m.startswith('peak') else 'M customer-hours'
    share=o/R.cust_h.sum() if not m.startswith('peak') else np.nan
    line=f'{m}: {o/M:.2f} {unit}'+(f' ({100*share:.1f}% of outage hours)' if np.isfinite(share) else '')+f'; human share {100*sh.mean():.0f}% (5-95% {100*np.percentile(sh,5):.0f} to {100*np.percentile(sh,95):.0f})'
    mm=pd.Series(sh,index=[k.split('|')[0] for k in nat]).groupby(level=0).mean().round(2).to_dict()
    for w in (1.5,2.0,2.8):
        v=np.array([R[f'{m}|wl{w}|{k}'].sum() for k in wl[w]]); line+=f'; x{np.median(v)/o:.2f} at {w}' if o>0 else ''
    print(line); print('   model means',mm)
    out[m]=dict(obs=o,share=share,hs=sh.mean(),hs_lo=np.percentile(sh,5),hs_hi=np.percentile(sh,95))
pd.DataFrame(out).T.to_csv(f'pilot/us_v2_totals_{TAG}.csv')
# restoration requirement
def agg(clim,m): return sum(cur[(s,clim,m)] for s in R.storm if (s,clim,m) in cur)
def fstar(curve,target):
    if target<=curve[0]: return FS[0]
    if target>=curve[-1]: return np.nan
    return float(np.interp(target,curve,FS))
for m in ['hour35','peak35','day35','hour406','peak406']:
    obs=agg('obs',m)
    if obs[i1]<=0: continue
    natv=np.array([agg(f'nat|{k}',m)[i1] for k in nat]); fn=fstar(obs,natv.mean())
    s=f'restoration {m}: natural-only level needs x{fn:.2f};'
    for w in (1.5,2.0,2.8):
        fs=np.array([fstar(agg(f'wl{w}|{k}',m),obs[i1]) for k in wl[w]]); s+=f' {w}: x{np.nanmedian(fs):.2f} ({np.nanpercentile(fs,5):.2f}-{np.nanpercentile(fs,95):.2f}, n={np.isfinite(fs).sum()})'
    s+='; elasticity '+str([round(float(np.interp(f,FS,obs)/obs[i1]),2) for f in (0.5,0.75,1.25)]); print(s)
# prolonged outages
for m in ['hour35','peak35','night26','day35']:
    tot=R[f'{m}|obs'].sum(); print(f'{m}: share after 24/48/72 h of continuous outage',[round(100*R[f"{m}_after{D}"].sum()/tot) for D in (24,48,72)])
print('all outage hours after 24/48/72 h',[round(100*R[f"cust_h_after{D}"].sum()/R.cust_h.sum()) for D in (24,48,72)])
for t in ('peak35','peak406'): print(f'customers without power at >=k hazardous peaks ({t}), million:',[round(R[f"cust_{t}_ge{k}"].sum()/M,2) for k in range(1,6)])
# storm-level subsets for hour35 and peak35
def sub(mask,label):
    X=R[mask]; s=f'{label}: n={len(X)}, cust-h {X.cust_h.sum()/M:.0f} M'
    for m in ('day35','hour35','peak35'):
        o=X[f'{m}|obs'].sum(); n=np.array([X[f'{m}|nat|{k}'].sum() for k in nat]); s+=f'; {m} {o/M:.2f} M, human {100*(1-n.mean()/o):.0f}% ({100*np.percentile(1-n/o,5):.0f}-{100*np.percentile(1-n/o,95):.0f})'
        if not m.startswith('peak'): s+=f' [{100*o/X.cust_h.sum():.1f}%]'
    print(s)
sub(R.season>=2018,'2018-2024'); sub(R.storm!='Irma','without Irma'); sub(~R.storm.isin(['Irma','Beryl','Ida']),'without Irma, Beryl, Ida')
lo=[]
for s_ in R.storm:
    X=R[R.storm!=s_]; o=X['hour35|obs'].sum(); n=np.mean([X[f'hour35|nat|{k}'].sum() for k in nat]); lo.append(1-n/o)
print('leave-one-storm-out human share hour35: %.0f-%.0f%%'%(100*min(lo),100*max(lo)))
top=R.sort_values('hour35|obs',ascending=False).head(15)
print(top[['storm','season','counties','cust_h','day35|obs','hour35|obs','hour406|obs','peak35|obs','night26|obs']].assign(**{c:lambda d,c=c:d[c]/M for c in ['cust_h','day35|obs','hour35|obs','hour406|obs','peak35|obs','night26|obs']}).round(2).to_string())
print('top3 share of hour35',round(100*top.head(3)['hour35|obs'].sum()/R['hour35|obs'].sum()),'%  storms with any hour35:',int((R['hour35|obs']>0).sum()))
print('by month: share of outage hours in hours HI>=35',(R.groupby('month')['hour35|obs'].sum()/R.groupby('month').cust_h.sum()*100).round(1).to_dict())
# county-level subsets from records (observed metrics)
recs=pd.read_pickle(f'pilot/us_v2_recs_{TAG}.pkl')
def county_sub(fn,label):
    tot={m:0.0 for m in ('cust','day35','hour35','peak35')}; tw={m:0.0 for m in tot}
    for r in recs:
        k=fn(r); win=((r['tt']>=-24)&(r['tt']<720))[:,None]; E=r['Ex']*win
        tot['cust']+=E.sum(); tw['cust']+=E[:,k].sum()
        for m in ('day35','hour35','peak35'): tot[m]+=(E*r['flags'][m]).sum(); tw[m]+=(E*r['flags'][m])[:,k].sum()
    print(f'{label}: kept {100*tw["cust"]/tot["cust"]:.0f}% of outage hours; hour35 share {100*tw["hour35"]/tw["cust"]:.1f}% (all {100*tot["hour35"]/tot["cust"]:.1f}%); day35 {100*tw["day35"]/tw["cust"]:.0f}%; peak35 per 1,000 customer-hours {1000*tw["peak35"]/tw["cust"]:.1f} (all {1000*tot["peak35"]/tot["cust"]:.1f})')
county_sub(lambda r: np.nan_to_num(r['dtrack'],nan=1e9)<=200,'counties within 200 km of track')
county_sub(lambda r: np.nan_to_num(r['dtrack'],nan=1e9)<=400,'counties within 400 km of track')
county_sub(lambda r: r['dland']<=300,'counties within 300 km of landfall')
county_sub(lambda r: (r['cov']>=0.9)&(r['dqi']>=80),'state coverage >=90% and FEMA-region DQI >=80 (2018-2022 only)')
county_sub(lambda r: r['dcell']<=150,'counties within 150 km of a landfall cell')
