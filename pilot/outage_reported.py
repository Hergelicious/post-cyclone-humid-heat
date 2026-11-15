import pandas as pd, numpy as np
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def hi_series(la,lo):
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    return pd.Series(hi_c(s.T2M_MAX.values,np.clip(100*e/es,1,100)),index=pd.date_range('1981-01-01',periods=len(s)))
H=pd.read_csv('pilot/outage_heat.csv',parse_dates=['time']); M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
rep={'Visakhapatnam':(3,14,'about 20% of the city supplied on day 3; full restoration estimated at about two weeks'),
     'Zhanjiang':(2,5,'78% of households restored by day 2; parts still without power on day 5'),
     'Maoming':(2,5,'regional figures as for Zhanjiang'),
     'SanJuan':(100,170,'55% of customers restored after 100 days; 90% after about 170 days'),
     'Bhubaneswar':(8,13,'half of 450,000 consumers restored on day 7–8; pockets without power on day 13'),
     'Calcutta':(5,7,'utility reported 95% restored on day 5; pockets until day 7'),
     'NewOrleans':(10,11,'vast majority restored on day 10; 98% on day 11'),
     'Houston':(7,10,'1.3 million customers still out on day 2; most restored by day 10')}
rows=[]
for r in M[M.city.isin(rep)&(M.min7<0.6)].itertuples():
    lo,hi_,txt=rep[r.city]; h=hi_series(np.round(r.clat*2)/2,np.round(r.clon*2)/2); t=r.time.normalize()
    w1=h[t+pd.Timedelta(days=1):t+pd.Timedelta(days=lo)]; w2=h[t+pd.Timedelta(days=1):t+pd.Timedelta(days=hi_)]
    rows.append(dict(city=r.city,date=t.date(),rep_lo=lo,rep_hi=hi_,hi35=f"{(w1>=35).sum()}–{(w2>=35).sum()}",hi406=f"{(w1>=40.6).sum()}–{(w2>=40.6).sum()}",himax=round(w2.max(),1),ntl_last_dark=r.last_dark,ntl_rec=r.rec_day,source=txt))
O=pd.DataFrame(rows); pd.set_option('display.width',250); print(O.to_string()); O.to_csv('pilot/outage_reported.csv',index=False)
