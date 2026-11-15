import pandas as pd, numpy as np, os, requests, warnings; warnings.filterwarnings('ignore')
src=open('pilot/reconcile.py').read(); exec(src[src.index('def hi_c') if 'def hi_c' in src else 0:0])
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time']); ok=M[M.status=='ok'].copy(); ok['outage']=ok.outage.astype(bool)
S=pd.read_csv('pilot/hdntl_series2.csv')
# empirical false-positive rates from controls are approximated by thr design; stricter threshold
for th in [0.75,0.6,0.5]: print('threshold',th,'detections',(ok.min7<th).sum(),'of',len(ok))
def get(la,lo):
    fn=f'pilot/power/{la}_{lo}.csv'
    if not os.path.exists(fn):
        u=("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,QV2M,PS,PRECTOTCORR"f"&community=AG&longitude={lo}&latitude={la}&start=19810101&end=20251231&format=CSV&header=false")
        r=requests.get(u,timeout=180); open(fn,'w').write(r.text)
    s=pd.read_csv(fn).replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5)); rh=np.clip(100*e/es,1,100)
    return pd.Series(hi_c(s.T2M_MAX.values,rh),index=pd.date_range('1981-01-01',periods=len(s)))
rows=[]
for r in ok.itertuples():
    hi=get(np.round(r.clat*2)/2,np.round(r.clon*2)/2); t=r.time.normalize()
    w14=hi[t+pd.Timedelta(days=1):t+pd.Timedelta(days=14)]
    d=dict(i=r.Index,city=r.city,iso2=r.iso2,time=r.time,lf_wind=r.lf_wind,lmic=r.lmic,pop_k=r.pop,min7=r.min7,outage=r.outage and r.min7<0.6,last_dark=r.last_dark,rec_day=r.rec_day,hi35_14=(w14>=35).sum(),hi406_14=(w14>=40.6).sum(),himax_14=w14.max())
    if d['outage']:
        lo=int(r.last_dark) if np.isfinite(r.last_dark) else 1; hiend=int(r.rec_day) if np.isfinite(r.rec_day) else 60
        wl=hi[t+pd.Timedelta(days=1):t+pd.Timedelta(days=max(lo,1))]; wh=hi[t+pd.Timedelta(days=1):t+pd.Timedelta(days=hiend)]
        d.update(out_hi35_lo=(wl>=35).sum(),out_hi35_hi=(wh>=35).sum(),out_hi406_lo=(wl>=40.6).sum(),out_hi406_hi=(wh>=40.6).sum(),out_himax=wh.max())
    rows.append(d)
H=pd.DataFrame(rows); H['class']=np.select([H.outage&(H.hi35_14>0),H.outage,H.hi35_14>0],['both','outage only','heat only'],'neither')
H.to_csv('pilot/outage_heat.csv',index=False)
print(H['class'].value_counts()); print(pd.crosstab(H['class'],H.lmic))
pd.set_option('display.width',250); print(H[H.outage][['city','iso2','time','lf_wind','pop_k','min7','last_dark','rec_day','out_hi35_lo','out_hi35_hi','out_hi406_lo','out_hi406_hi','out_himax']].round(1).to_string())
print('share of 14-day windows with HI>=35 day:',round((H.hi35_14>0).mean(),2),' >=40.6:',round((H.hi406_14>0).mean(),2))
