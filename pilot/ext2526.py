import pandas as pd, numpy as np, json, sys
sys.path.insert(0,'pilot')
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
x=pd.read_csv('pilot/landfalls_ext_2013_26.csv',parse_dates=['time'])
wb=json.load(open('pilot/wb_countries.json'))[1]; inc2={c['iso2Code']:c['incomeLevel']['id'] for c in wb}; i3={c['id']:c['iso2Code'] for c in wb}
x['iso2']=x.iso.map(i3); x['lmic']=x.iso2.map(inc2).isin(['LIC','LMC','UMC'])
T=np.load('data/ghsl/transform.npy'); P20=np.load('data/ghsl/pop_2020_0p1.npy'); res=0.1
lats=T[5]-(np.arange(P20.shape[0])+0.5)*res; lons=T[2]+(np.arange(P20.shape[1])+0.5)*res
def pop(la,lo,R=100):
    i=np.where(np.abs(lats-la)<=1.2)[0]; j=np.where(np.abs(((lons-lo+180)%360)-180)<=1.2/np.cos(np.radians(la)))[0]
    LA,LO=np.meshgrid(lats[i],lons[j],indexing='ij'); a=np.sin(np.radians(LA-la)/2)**2+np.cos(np.radians(la))*np.cos(np.radians(LA))*np.sin(np.radians(LO-lo)/2)**2
    return P20[np.ix_(i,j)][2*6371*np.arcsin(np.sqrt(a))<=R].sum()
rows=[]
for r in x.itertuples():
    fn=f'pilot/power26/{r.clat}_{r.clon}.csv'
    s=pd.read_csv(fn).replace(-999,np.nan); s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
    q=s.QV2M/1000; p=s.PS*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5)); s['HI']=hi_c(s.T2M_MAX,np.clip(100*e/es,1,100))
    s=s.set_index('date'); t0=r.time.normalize(); w=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30),'HI'].dropna()
    if len(w)<20: continue
    f=30/len(w); rows.append(dict(sid=r.sid,name=r.name,country=r.country,time=r.time,lmic=r.lmic,year=r.time.year,d35=(w>=35).sum()*f,d406=(w>=40.6).sum()*f,pop=pop(r.lat,r.lon)))
X=pd.DataFrame(rows); X['grp']=pd.cut(X.year,[2012,2017,2022,2024,2026],labels=['2013-2017','2018-2022','2023-2024','2025-2026'])
print(X.groupby('grp',observed=True).agg(n=('sid','size'),any406=('d406',lambda v:(v>0).mean()),d35=('d35','mean'),d406=('d406','mean')).round(3))
print(X.groupby(['lmic','grp'],observed=True).agg(n=('sid','size'),any406=('d406',lambda v:(v>0).mean()),d35=('d35','mean')).round(3))
y=X[X.year>=2025]; print('2025-26 LMIC person-days HI>=40.6 (M):',round((y[y.lmic]['pop']*y[y.lmic].d406).sum()/1e6,0),' n LMIC',y.lmic.sum())
print(y.sort_values('d406',ascending=False).head(8)[['name','country','time','d406','d35']].round(1).to_string())
X.to_csv('pilot/ext_events.csv',index=False)
