import pandas as pd, numpy as np, json, os, sys
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
rng=np.random.default_rng(3)
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index()
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
lf['lmic']=lf.iso2.map(inc).isin(['LIC','LMC','UMC'])
# population on 0.5 deg POWER grid
T=np.load('data/ghsl/transform.npy'); P={ep:np.load(f'data/ghsl/pop_{ep}_0p1.npy') for ep in (1980,2000,2020)}
nlat,nlon=P[2020].shape; lats=T[5]-(np.arange(nlat)+0.5)*0.1; lons=T[2]+(np.arange(nlon)+0.5)*0.1
ri=np.clip(((90-lats)/0.5).astype(int),0,359); ci=np.clip(((lons+180)/0.5).astype(int),0,719)
P05={}
for ep,a in P.items():
    g=np.zeros((360,720)); np.add.at(g,(ri[:,None].repeat(nlon,1).ravel(),ci[None,:].repeat(nlat,0).ravel()),a.ravel()); P05[ep]=g
def pop_cell(la,lo,yr):
    i=int((90-la)/0.5); j=int((lo+180)/0.5)%720
    v={ep:P05[ep][i,j] for ep in P05}
    if yr<=2000: w=(yr-1980)/20; return (1-w)*v[1980]+w*v[2000]
    if yr<=2020: w=(yr-2000)/20; return (1-w)*v[2000]+w*v[2020]
    return v[2020]
cache={}
def series(la,lo):
    k=(la,lo)
    if k not in cache:
        fn=f'pilot/power/{la}_{lo}.csv'
        if not os.path.exists(fn): cache[k]=None; return None
        s=pd.read_csv(fn).replace(-999,np.nan); s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
        q=s.QV2M/1000; p=s.PS*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5)); s['HI']=hi_c(s.T2M_MAX,np.clip(100*e/es,1,100))
        s=s.set_index('date'); cache[k]=(s.HI.values>=35, s.HI.values>=40.6, s.index); 
    return cache[k]
dates=pd.date_range('1981-01-01','2025-12-31'); assert len(dates)==len(series(np.round(lf.lat[0]*2)/2,np.round(lf.lon[0]*2)/2)[2])
from global_land_mask import globe
def cells_within(la,lo,R):
    out=[]; dl=R/111/max(np.cos(np.radians(la)),0.3)
    for a in np.arange(np.round(la*2)/2-np.ceil(R/111*2)/2,np.round(la*2)/2+np.ceil(R/111*2)/2+0.01,0.5):
        for b in np.arange(np.round(lo*2)/2-np.ceil(dl*2)/2,np.round(lo*2)/2+np.ceil(dl*2)/2+0.01,0.5):
            b2=((b+180)%360)-180
            d=2*6371*np.arcsin(np.sqrt(np.sin(np.radians(a-la)/2)**2+np.cos(np.radians(la))*np.cos(np.radians(a))*np.sin(np.radians(b2-lo)/2)**2))
            if d<=R and globe.is_land(a,b2): out.append((round(a,1),round(b2,1)))
    return out
def wind_R(w): return float(np.clip(50+(w-34)*150/66,50,200))
foot={'50':lambda r:50,'100':lambda r:100,'200':lambda r:200,'wind':lambda r:wind_R(r.lf_wind)}
# per landfall, per cell: hot-day masks in window
recs=[]  # (idx, footprint, cell, pop, d35, d406, dayset35 start index)
hot35={k:{} for k in foot}; hot406={k:{} for k in foot}  # footprint -> cell -> set of day indices (for de-dup)
tot=[]
for r in lf.itertuples():
    t0=np.searchsorted(dates.values,r.time.normalize().to_datetime64()); yr=r.season
    for k,f in foot.items():
        R=f(r); cs=cells_within(r.lat,r.lon,R); e35=e406=0.0; pp=0.0
        for (a,b) in cs:
            s=series(a,b)
            if s is None: continue
            m35=s[0][t0+1:t0+31]; m406=s[1][t0+1:t0+31]; pc=pop_cell(a,b,yr); pp+=pc
            e35+=pc*m35.sum(); e406+=pc*m406.sum()
            d35=hot35[k].setdefault((a,b),{}); d406=hot406[k].setdefault((a,b),{})
            for di in np.nonzero(m35)[0]: d35[t0+1+di]=pc
            for di in np.nonzero(m406)[0]: d406[t0+1+di]=pc
        tot.append(dict(idx=r.index,season=yr,iso2=r.iso2,lmic=r.lmic,foot=k,R=R,ncell=len(cs),pop=pp,pd35=e35,pd406=e406))
E=pd.DataFrame(tot); E['period']=pd.cut(E.season,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024'])
E.to_csv('pilot/spatial_exposure_events.csv',index=False)
ny={'1981-2000':20,'2001-2017':17,'2018-2024':7}
print('Summed person-days HI>=40.6 (M/yr) by footprint, LMIC:')
S=E[E.lmic].groupby(['foot','period'],observed=True).pd406.sum().unstack(); print((S.div(pd.Series(ny))/1e6).round(0))
print('LMIC share 2018-2024 by footprint:', (E[(E.season>=2018)&E.lmic].groupby('foot').pd406.sum()/E[E.season>=2018].groupby('foot').pd406.sum()).round(3).to_dict())
# de-duplicated unique person-days by period and income (cell-day counted once)
cell_lmic={}
for r in lf.itertuples():
    for (a,b) in cells_within(r.lat,r.lon,200): cell_lmic.setdefault((a,b),r.lmic)
rows=[]
for k in foot:
    for thr,H in [('35',hot35[k]),('406',hot406[k])]:
        for (cell,dd) in H.items():
            for di,pc in dd.items():
                y=dates[di].year; rows.append((k,thr,cell_lmic.get(cell,False),y,pc))
U=pd.DataFrame(rows,columns=['foot','thr','lmic','year','pd']); U['period']=pd.cut(U.year,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024'])
UU=U.groupby(['foot','thr','lmic','period'],observed=True).pd.sum().unstack('period'); UU=UU.div(pd.Series(ny))/1e6; UU.to_csv('pilot/spatial_unique.csv'); print('Unique person-days (M/yr):'); print(UU.round(1))
# bootstrap by country for 100 km, LMIC, 2018-2024 and 1981-2000 sums
e=E[(E.foot=='100')&E.lmic]; cs=e.iso2.unique(); bs=[]
for _ in range(500):
    pick=rng.choice(cs,len(cs)); s=pd.concat([e[e.iso2==c] for c in pick])
    bs.append([s[s.period==p].pd406.sum()/ny[p]/1e6 for p in ny])
bs=np.array(bs); print('bootstrap 95% (M/yr) LMIC HI>=40.6 100 km:',{p:np.percentile(bs[:,i],[2.5,97.5]).round(0).tolist() for i,p in enumerate(ny)})
np.save('pilot/boot_exposure.npy',bs)
