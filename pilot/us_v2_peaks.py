# Customers without power through k or more hazardous daily peaks, counting peaks from the county's outage onset onwards
# (customers still out at the k-th peak are bounded by the running minimum of customers out over peaks 1..k).
import pandas as pd, numpy as np, sys
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
recs=pd.read_pickle(f'pilot/us_v2_recs_{TAG}.pkl'); rows=[]
for r in recs:
    win=(r['tt']>=-24)&(r['tt']<720); ck={}
    for t in ('peak35','peak406'):
        c=np.zeros(5)
        for j in range(len(r['fips'])):
            hs=np.where(r['flags'][t][:,j]&win&(np.arange(len(win))>=r['onset'][j]))[0]
            if not len(hs): continue
            v=np.minimum.accumulate(r['Ex'][hs,j])
            for k in range(min(5,len(v))): c[k]+=v[k]
        ck[t]=c
    rows.append(dict(storm=r['storm'],**{f'{t}_ge{k+1}':ck[t][k] for t in ck for k in range(5)}))
D=pd.DataFrame(rows); D.to_csv(f'pilot/us_v2_peaks_{TAG}.csv',index=False)
print(TAG,'customers (million) at >=1..5 peaks HI>=35:',(D[[f'peak35_ge{k}' for k in range(1,6)]].sum()/1e6).round(2).tolist(),' HI>=40.6:',(D[[f'peak406_ge{k}' for k in range(1,4)]].sum()/1e6).round(3).tolist())
print(D.set_index('storm')[['peak35_ge1','peak35_ge3']].sort_values('peak35_ge1',ascending=False).head(14).div(1e3).round(0).to_string())
