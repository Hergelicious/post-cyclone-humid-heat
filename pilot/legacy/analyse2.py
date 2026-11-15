import pandas as pd, numpy as np, json
import statsmodels.formula.api as smf
ev=pd.read_csv('pilot/events.csv')
ev['hit406']=(ev.d406>0).astype(int)
m2=smf.logit('hit406 ~ gmst + base_d406',data=ev).fit(disp=0,cov_type='cluster',cov_kwds={'groups':ev.iso2.astype('category').cat.codes})
print('logit HI>=40.6 any day, OR per +0.5C GMST:',round(np.exp(0.5*m2.params.gmst),2),'p=',round(m2.pvalues.gmst,4))
for sub,name in [(ev[ev.lmic],'LMIC'),(ev[~ev.lmic],'HIC/NA')]:
    m=smf.ols('d35 ~ gmst + base_d35',data=sub).fit(cov_type='cluster',cov_kwds={'groups':sub.iso2.astype('category').cat.codes})
    print(name,'extra days HI>=35 per +1C GMST:',round(m.params.gmst,2),'95%CI',m.conf_int().loc['gmst'].round(2).tolist())
# fragility layer
def wdi(ind):
    d=json.load(open(f'pilot/wdi/{ind}.json'))[1]
    df=pd.DataFrame([(x['country']['id'],x['countryiso3code'],int(x['date']),x['value']) for x in d if x['value'] is not None],columns=['iso2','iso3','yr','v'])
    return df.sort_values('yr').groupby('iso2').v.last()
F=pd.DataFrame({k:wdi(i) for k,i in dict(elec='EG.ELC.ACCS.ZS',water_basic='SH.H2O.BASW.ZS',san_basic='SH.STA.BASS.ZS',undernour='SN.ITK.DEFC.ZS',wash_mort='SH.STA.WASH.P5').items()})
r=ev[ev.season>=2018].groupby('iso2').agg(landfalls=('track','size'),hit406=('hit406','sum'),d35=('d35','sum'),inc=('inc','first'))
r=r.join(F,how='left'); r=r[r.inc.isin(['LIC','LMC','UMC'])].sort_values('d35',ascending=False)
print(r.head(18).round(1).to_string())
