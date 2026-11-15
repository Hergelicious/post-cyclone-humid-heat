# Is the outage-heat overlap larger or smaller than expected from the local seasonal background?
# Keep each county's observed outage curve and replace its post-landfall heat with ERA5 heat from the 14 days before the storm
# window (days -14..-1) or from days +31..+44 after landfall, tiled by hour of day; compare simultaneous exposure (HI>=35 hours)
# and customers out at hazardous daily peaks with the observed values.
import pandas as pd, numpy as np, warnings, sys, os
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('GAP=int'):src.index('# ---- CMIP6 fields')])
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl'); L2=US.copy(); L2['nm']=[NAMES.get(t,t) for t in L2.track_id]; lfmap=dict(zip(L2.nm,L2.time))
rows=[]
for r in recs:
    lf=lfmap[r['storm']].floor('h'); t0=lf-pd.Timedelta(hours=24); fl=r['fips']; Ex=r['Ex']; nh=Ex.shape[0]; lons=[lon_of[f] for f in fl]
    win=((r['tt']>=-24)&(r['tt']<720))[:,None]
    obs35=(Ex*(r['HI']>=35)*win).sum(); obsday=(Ex*r['flags']['day35']*win).sum()
    res=dict(storm=r['storm'],obs_hour35=obs35,obs_day35=obsday,cust_h=(Ex*win).sum())
    for lab,start in (('pre',lf-pd.Timedelta(days=14)),('post',lf+pd.Timedelta(days=31))):
        Tm,RH=heat_block(fl,start,14*24)
        if np.isnan(Tm).all(): res[f'{lab}_hour35']=np.nan; continue
        H14=hi_c(Tm,RH); reps=int(np.ceil(nh/(14*24))); Ht=np.tile(H14,(reps,1))[:nh]
        dm=local_daymax(Ht,t0,lons)
        res[f'{lab}_hour35']=(Ex*(Ht>=35)*win).sum(); res[f'{lab}_day35']=(Ex*(dm>=35)*win).sum()
    rows.append(res)
D=pd.DataFrame(rows); D.to_csv('pilot/us_overlap_null.csv',index=False)
t=D.sum(numeric_only=True)
print('simultaneous HI>=35 customer-hours (M): observed %.0f, with pre-storm heat %.0f, with heat from days 31-44 %.0f'%(t.obs_hour35/1e6,t.pre_hour35/1e6,t.post_hour35/1e6))
print('hot-day customer-hours (M): observed %.0f, pre-storm %.0f, days 31-44 %.0f'%(t.obs_day35/1e6,t.pre_day35/1e6,t.post_day35/1e6))
print('shares of outage hours: observed %.1f%%, pre %.1f%%, post %.1f%%'%(100*t.obs_hour35/t.cust_h,100*t.pre_hour35/t.cust_h,100*t.post_hour35/t.cust_h))
top=D.sort_values('obs_hour35',ascending=False).head(8); print((top.set_index('storm')[['obs_hour35','pre_hour35','post_hour35']]/1e6).round(1))
