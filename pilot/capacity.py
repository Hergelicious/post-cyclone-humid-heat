import pandas as pd, numpy as np, json
ev=pd.read_csv('pilot/power_events_exposure.csv')
def wdi(ind):
    d=json.load(open(f'pilot/wdi/{ind}.json'))[1]
    df=pd.DataFrame([(x['country']['id'],x['countryiso3code'],x['country']['value'],int(x['date']),x['value']) for x in d if x['value'] is not None],columns=['iso2','iso3','name','yr','v'])
    return df.sort_values('yr').groupby('iso2').agg(v=('v','last'),yr=('yr','last'),name=('name','last'))
I=dict(elec='EG.ELC.ACCS.ZS',water='SH.H2O.BASW.ZS',san='SH.STA.BASS.ZS',undernour='SN.ITK.DEFC.ZS',washmort='SH.STA.WASH.P5')
F=pd.concat({k:wdi(v).v for k,v in I.items()},axis=1)
names=wdi('SP.POP.TOTL').name
c=ev[ev.season>=2013].groupby('iso2').agg(landfalls=('track','size'),pd35=('pd_d35','sum'),pd406=('pd_d406','sum'),pop=('pop100','sum'),inc=('inc','first'),lmic=('lmic','first'))
c['pd35_yr']=c.pd35/12; c['pd406_yr']=c.pd406/12
c=c.join(F,how='left').join(names.rename('name'))
D=pd.DataFrame({'elec':100-c.elec,'water':100-c.water,'san':100-c.san,'undernour':c.undernour,'washmort':np.log10(c.washmort.clip(lower=0.1))})
Dn=(D-D.min())/(D.max()-D.min()); c['deficit']=Dn.mean(axis=1,skipna=True); c['n_ind']=D.notna().sum(1)
c.to_csv('pilot/country_capacity.csv')
print(c.sort_values('pd35_yr',ascending=False)[['name','inc','landfalls','pd35_yr','pd406_yr','elec','water','san','undernour','washmort','deficit','n_ind']].head(25).round(2).to_string())
low=c[(c.deficit>=0.4)]
print('countries with deficit>=0.4:',list(low.name)); 
tot=c.pd35_yr.sum(); print('share of HI>=35 person-days 2013-2024 in countries with elec<90%:',round(c[c.elec<90].pd35_yr.sum()/tot,3),' sanitation<70%:',round(c[c.san<70].pd35_yr.sum()/tot,3))
print('people (person-days/yr, millions) HI>=35 in countries with elec<90%:',round(c[c.elec<90].pd35_yr.sum()/1e6,1))
