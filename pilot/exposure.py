import pandas as pd, numpy as np, json
ev=pd.read_csv('pilot/power_events_full.csv')
T=np.load('data/ghsl/transform.npy')  # 30ss transform
P={ep:np.load(f'data/ghsl/pop_{ep}_0p1.npy') for ep in (1980,2000,2020)}
res=0.1; lon0=T[2]; lat0=T[5]
nlat,nlon=P[2020].shape
lats=lat0-(np.arange(nlat)+0.5)*res; lons=lon0+(np.arange(nlon)+0.5)*res
def popnear(la,lo,yr,R=100):
    i=np.where(np.abs(lats-la)<=1.2)[0]; dl=1.2/np.cos(np.radians(la))
    j=np.where(np.abs(((lons-lo+180)%360)-180)<=dl)[0]
    LA,LO=np.meshgrid(lats[i],lons[j],indexing='ij')
    a=np.sin(np.radians(LA-la)/2)**2+np.cos(np.radians(la))*np.cos(np.radians(LA))*np.sin(np.radians(LO-lo)/2)**2
    m=2*6371*np.arcsin(np.sqrt(a))<=R
    vals={ep:P[ep][np.ix_(i,j)][m].sum() for ep in P}
    if yr<=2000: w=(yr-1980)/20; return (1-w)*vals[1980]+w*vals[2000]
    if yr<=2020: w=(yr-2000)/20; return (1-w)*vals[2000]+w*vals[2020]
    return vals[2020]
ev['pop100']=[popnear(a,b,y) for a,b,y in zip(ev.lat,ev.lon,ev.season)]
for k in ['d35','d406','tw28']: ev[f'pd_{k}']=ev.pop100*ev[k]
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
ev['inc']=ev.iso2.map(inc).fillna('NA'); ev['lmic']=ev.inc.isin(['LIC','LMC','UMC'])
ev['period']=pd.cut(ev.season,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024'])
ev.to_csv('pilot/power_events_exposure.csv',index=False)
ny={'1981-2000':20,'2001-2017':17,'2018-2024':7}
t=ev.groupby(['period','lmic'],observed=True).agg(n=('track','size'),pop=('pop100','sum'),pd35=('pd_d35','sum'),pd406=('pd_d406','sum')).reset_index()
t['pd35_per_yr_bn']=t.pd35/t.period.map(ny).astype(float)/1e9; t['pd406_per_yr_m']=t.pd406/t.period.map(ny).astype(float)/1e6
print(t.round(3).to_string())
print('LMIC share of person-days HI>=40.6, 2018-2024:', round(ev[(ev.season>=2018)&ev.lmic].pd_d406.sum()/ev[ev.season>=2018].pd_d406.sum(),3))
print('LMIC share of person-days HI>=35, 2018-2024:', round(ev[(ev.season>=2018)&ev.lmic].pd_d35.sum()/ev[ev.season>=2018].pd_d35.sum(),3))
