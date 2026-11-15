import pandas as pd, numpy as np, statsmodels.formula.api as smf
ev=pd.read_csv('pilot/power_events_exposure.csv'); ev['wet']=ev.rain4>=100; ev['wethot']=(ev.wet&(ev.hot21>=7)).astype(int)
cl=lambda d: dict(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes})
print('N',len(ev),'LMIC',ev.lmic.sum(),'countries',ev.iso2.nunique())
print(ev.groupby('period',observed=True).agg(n=('track','size'),any406=('d406',lambda x:(x>0).mean()),d35=('d35','mean'),d406=('d406','mean')).round(3))
print(ev.groupby(['lmic','period'],observed=True).agg(n=('track','size'),any406=('d406',lambda x:(x>0).mean()),d35=('d35','mean'),wet=('wet','mean'),wethot=('wethot','mean')).round(3))
ev['hit']=(ev.d406>0).astype(int); m=smf.logit('hit ~ gpi',data=ev).fit(disp=0,**cl(ev)); print('any HI40.6 OR/0.5C',np.round(np.exp(.5*np.r_[m.params.gpi,m.conf_int().loc['gpi']]),2))
for lm in [True,False]:
    s=ev[ev.lmic==lm]; m=smf.logit('wethot ~ gpi',data=s).fit(disp=0,**cl(s)); print('LMIC' if lm else 'HIC','wethot OR/0.5C',np.round(np.exp(.5*np.r_[m.params.gpi,m.conf_int().loc['gpi']]),2))
for p,n in [('1981-2000',20),('2001-2017',17),('2018-2024',7)]:
    s=ev[(ev.period==p)&(ev.wethot==1)]; print(p,'wethot people LMIC M/yr',round(s[s.lmic].pop100.sum()/n/1e6,1),'HIC',round(s[~s.lmic].pop100.sum()/n/1e6,1))
c=pd.read_csv('pilot/country_capacity.csv'); low=c[c.deficit>=0.4].iso2.tolist(); s=ev[ev.iso2.isin(low)]
print('lowcap',low); print(s.groupby('period',observed=True).agg(n=('track','size'),d35=('d35','mean'),any406=('d406',lambda x:(x>0).mean())).round(2))
print('lowcap 2013-2024 people',round(s[s.season>=2013].pop100.sum()/1e6,1),'landfalls',(s.season>=2013).sum())
r=ev[(ev.season>=2018)&ev.lmic].groupby('iso2').pd_d406.sum().sort_values(ascending=False); print((r/r.sum()).head(5).round(3), 'top5',round((r/r.sum()).head(5).sum(),3))
a=ev[ev.season>=2005]
for lm in [True,False]:
    s=a[a.lmic==lm]; out=[f"obs pd406 {(s.pop100*s.d406).sum()/20/1e6:.0f}M any {(s.d406>0).mean()*100:.0f}% days {s.d406.mean():.2f}"]
    for k in ['1.5','2.0','2.8']:
        out.append(f"{k}: pd {(s.pop100*s[f'p{k}_d_d406']).sum()/20/1e6:.0f}-{(s.pop100*s[f'p{k}_c_d406']).sum()/20/1e6:.0f}M any {(s[f'p{k}_d_d406']>0).mean()*100:.0f}-{(s[f'p{k}_c_d406']>0).mean()*100:.0f}% days {s[f'p{k}_d_d406'].mean():.1f}-{s[f'p{k}_c_d406'].mean():.1f} tw28 {s[f'p{k}_d_tw28'].mean():.1f}-{s[f'p{k}_c_tw28'].mean():.1f}")
    print('LMIC' if lm else 'HIC', ' | '.join(out), '| obs tw28',round(s.tw28.mean(),2))
