# Robustness and diagnostics for the measured outage-heat analysis
import pandas as pd, numpy as np, sys
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
R=pd.read_csv(f'pilot/us_outage_storms_{TAG}.csv'); R=R[R.counties.notna()]
nat=[c for c in R if c.startswith('E35|nat|')]
def share(d,thr='35'):
    o=d[f'E{thr}|obs'].sum(); s=np.array([1-d[c.replace('E35',f'E{thr}')].sum()/o for c in nat]); return o/1e6,100*s.mean(),100*np.percentile(s,5),100*np.percentile(s,95)
print('human share HI>=35: all %.0f M, %.0f%% (%.0f-%.0f)'%share(R))
print('  2018-2024 only (state coverage known, Gulf/Atlantic states 63-99%%): %.0f M, %.0f%% (%.0f-%.0f)'%share(R[R.season>=2018]))
print('  without Irma: %.0f M, %.0f%% (%.0f-%.0f)'%share(R[R.storm!='Irma']))
print('  without Beryl: %.0f M, %.0f%% (%.0f-%.0f)'%share(R[R.storm!='Beryl']))
lo=[share(R[R.storm!=s])[1] for s in R.storm]; print('  leave-one-storm-out range of mean share: %.0f-%.0f%%'%(min(lo),max(lo)))
print('share of customer-hours on HI>=35 days: all %.0f%%, 2018-2024 %.0f%%, hurricane-strength %.0f%%'%(100*R.cust_h35.sum()/R.cust_h.sum(),100*R[R.season>=2018].cust_h35.sum()/R[R.season>=2018].cust_h.sum(),100*R[R.wind>=64].cust_h35.sum()/R[R.wind>=64].cust_h.sum()))
print('storms with >=1 M hot-day customer-hours:',int((R.cust_h35>=1e6).sum()),'; storms with any:',int((R.cust_h35>0).sum()),'of',len(R))
print('June-September landfalls: share hot %.0f%%; October-November %.0f%%'%(100*R[pd.to_datetime(R.landfall).dt.month<=9].cust_h35.sum()/R[pd.to_datetime(R.landfall).dt.month<=9].cust_h.sum(),100*R[pd.to_datetime(R.landfall).dt.month>=10].cust_h35.sum()/R[pd.to_datetime(R.landfall).dt.month>=10].cust_h.sum()))
# hours per customer affected (peak) and hot hours per customer
print('customer-hours per peak customer out: %.1f h; hot-day hours per peak customer: %.1f h'%(R.cust_h.sum()/R.peak_out.sum(),R.cust_h35.sum()/R.peak_out.sum()))
# profile by day since landfall
recs=pd.read_pickle(f'pilot/us_outage_series_{TAG}.pkl'); prof=np.zeros(31); hot=np.zeros(31)
for nm,fl,Ex,HId,tt in recs:
    d=np.floor(tt/24).astype(int); m=(d>=-1)&(d<30)
    for k in range(-1,30):
        s=(d==k); prof[k+1]+=Ex[s].sum(); hot[k+1]+=(Ex[s]*(HId[s]>=35)).sum()
P=pd.DataFrame(dict(day=np.arange(-1,30),cust_h=prof/1e6,cust_h35=hot/1e6)); P['share_hot']=P.cust_h35/P.cust_h
P.to_csv(f'pilot/us_outage_profile_{TAG}.csv',index=False); print(P.round(3).head(12).to_string(index=False))
# county checks against the eight-city analysis
for nm,f in [('Beryl','48201'),('Ida','22071')]:
    for n2,fl,Ex,HId,tt in recs:
        if n2==nm and f in fl:
            j=fl.index(f); print(nm,f,'peak out %.0f'%Ex[:,j].max(),'customer-hours %.1f M'%(Ex[:,j].sum()/1e6),'on HI>=35 days %.1f M'%((Ex[:,j]*(HId[:,j]>=35)).sum()/1e6),'= %.2f M customer-days'%((Ex[:,j]*(HId[:,j]>=35)).sum()/24/1e6))
