# Selection check for the night-light outage series (Supplementary Table 10 note):
# pairs without usable images vs usable pairs vs the eight detected outages (first-week ratio < 0.6).
# Heat = days with HI >= 35 C (daily approximation) in days 1-30 at the 0.5 deg cell nearest the city.
import pandas as pd, numpy as np, os
from scipy.stats import mannwhitneyu as mw
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def tq(la,lo):
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    idx=pd.date_range('1981-01-01',periods=len(s)); return pd.Series(s.T2M_MAX.values,idx),pd.Series(np.clip(100*e/es,1,100),idx)
H=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
d35=[]
for r in H.itertuples():
    la=np.round(r.clat*2)/2; lo=np.round(r.clon*2)/2
    if not os.path.exists(f'pilot/power/{la}_{lo}.csv'): d35.append(np.nan); continue
    T,RH=tq(la,lo); t0=r.time.normalize(); w=pd.date_range(t0+pd.Timedelta(days=1),periods=30)
    d35.append(float(np.nansum(hi_c(T.reindex(w).values,RH.reindex(w).values)>=35)))
H['d35']=d35
H['grp']=np.where(H.status!='ok','excluded',np.where(H.min7<0.6,'detected','usable, not detected'))
ex=H[H.grp=='excluded']; us=H[H.status=='ok']; de=H[H.grp=='detected']; un=H[H.grp=='usable, not detected']
print('pairs',len(H),'missing d35',int(H.d35.isna().sum()))
R=pd.DataFrame([dict(grp=lab,n=len(d),wind_med=d.lf_wind.median(),hurr=(d.lf_wind>=64).mean(),d35_mean=d.d35.mean(),dist_med=d.dist.median()) for lab,d in [('excluded',ex),('usable (all)',us),('usable, not detected',un),('detected',de)]])
print(R.round(2).to_string())
t=lambda a,b,c: mw(a[c].dropna(),b[c].dropna()).pvalue
print('excluded vs usable (all):          P d35 %.2f, dist %.2f, wind %.2f'%(t(ex,us,'d35'),t(ex,us,'dist'),t(ex,us,'lf_wind')))
print('excluded vs usable, not detected:  P d35 %.2f, dist %.2f, wind %.2f'%(t(ex,un,'d35'),t(ex,un,'dist'),t(ex,un,'lf_wind')))
print('detected vs usable, not detected:  P d35 %.2f, wind %.4f'%(t(de,un,'d35'),t(de,un,'lf_wind')))
R.to_csv('pilot/outage_selection_check.csv',index=False)
