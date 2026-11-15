# Medicare beneficiaries who use electricity-dependent medical equipment (HHS emPOWER, county level, latest release) potentially
# without power in hazardous heat: county outage share (customers out / customers) applied to the county's emPOWER count, assuming
# outages fall uniformly across households. Also compares county rankings by customers out and by emPOWER beneficiaries out.
import pandas as pd, numpy as np
E=pd.read_csv('data/empower/county_current.csv',dtype={'FIPS_Code':str}); E['fips']=E.FIPS_Code.str.zfill(5); dme=dict(zip(E.fips,E.Power_Dependent_Devices_DME)); ben=dict(zip(E.fips,E.Medicare_Benes))
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl'); PK=pd.read_csv('pilot/us_v2_peaks_main.csv')
tot=dict(cust_h35=0,dme_h35=0,dme_peak1=0,dme_days=0,cust_peak1=0); rank=[]
for r in recs:
    win=((r['tt']>=-24)&(r['tt']<720))[:,None]; E_=r['Ex']*win; c=r['cust']; d=np.array([dme.get(f,np.nan) for f in r['fips']])
    share=E_/c[None]; F=r['HI']>=35
    tot['cust_h35']+=(E_*F).sum(); tot['dme_h35']+=np.nansum(share*F*d[None])
    tot['dme_days']+=np.nansum(share*r['flags']['peak35']*d[None])
    p1=r['cty']['obs']['sev'][:,0]; tot['cust_peak1']+=p1.sum(); tot['dme_peak1']+=np.nansum(p1/c*d)
    rank.append(pd.DataFrame(dict(storm=r['storm'],fips=r['fips'],cust_h35=(E_*F).sum(0),dme_h35=np.nan_to_num((share*F).sum(0)*d),dme_rate=d/c)))
K=pd.concat(rank); K=K[K.cust_h35>0]
print('emPOWER beneficiaries without power at >=1 hazardous daily peak (summed over storms): %.0f; beneficiary-days at hazardous peaks %.0f; beneficiary-hours with HI>=35 %.0f'%(tot['dme_peak1'],tot['dme_days'],tot['dme_h35']))
print('per 1,000 customer accounts out at >=1 peak: %.1f'%(1000*tot['dme_peak1']/tot['cust_peak1']))
# overlap of top-20% county lists within storms
ov=[];cap=[]
for s,X in K.groupby('storm'):
    n=max(1,int(np.ceil(0.2*len(X)))); a=set(X.nlargest(n,'cust_h35').fips); b=set(X.nlargest(n,'dme_h35').fips); ov.append(len(a&b)/n)
    cap.append((X[X.fips.isin(a)].dme_h35.sum(),X[X.fips.isin(b)].dme_h35.sum(),X.dme_h35.sum()))
cap=np.array(cap).sum(0)
print('top-20%% county lists by customer exposure and by emPOWER exposure overlap on average %.0f%%; emPOWER exposure captured by the customer ranking %.0f%% vs %.0f%% by the emPOWER ranking'%(100*np.mean(ov),100*cap[0]/cap[2],100*cap[1]/cap[2]))
print('county emPOWER users per customer account, exposed counties: median %.3f, 90th pct %.3f'%(K.dme_rate.median(),K.dme_rate.quantile(.9)))
