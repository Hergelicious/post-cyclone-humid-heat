import pandas as pd, numpy as np, json, os, glob, itertools, sys
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
from global_land_mask import globe
rng=np.random.default_rng(21)
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time']).reset_index()
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
lf['lmic']=lf.iso2.map(inc).isin(['LIC','LMC','UMC'])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
dates=pd.date_range('1981-01-01','2025-12-31'); yr=np.array(dates.year); doy=np.array(dates.dayofyear); mon=np.array(dates.month)
# population 0.5 deg
T=np.load('data/ghsl/transform.npy'); P={ep:np.load(f'data/ghsl/pop_{ep}_0p1.npy') for ep in (1980,2000,2020)}
nlat,nlon=P[2020].shape; lats=T[5]-(np.arange(nlat)+0.5)*0.1; lons=T[2]+(np.arange(nlon)+0.5)*0.1
ri=np.clip(((90-lats)/0.5).astype(int),0,359); ci=np.clip(((lons+180)/0.5).astype(int),0,719); P05={}
for ep,a in P.items():
    gg=np.zeros((360,720)); np.add.at(gg,(ri[:,None].repeat(nlon,1).ravel(),ci[None,:].repeat(nlat,0).ravel()),a.ravel()); P05[ep]=gg
def pop_cell(la,lo,yv):
    i=int((90-la)/0.5); j=int((lo+180)/0.5)%720; v={ep:P05[ep][i,j] for ep in P05}
    if yv<=2000: w=(yv-1980)/20; return (1-w)*v[1980]+w*v[2000]
    if yv<=2020: w=(yv-2000)/20; return (1-w)*v[2000]+w*v[2020]
    return v[2020]
cache={}
def series(la,lo):
    k=(la,lo)
    if k not in cache:
        fn=f'pilot/power/{la}_{lo}.csv'
        if not os.path.exists(fn): cache[k]=None; return None
        s=pd.read_csv(fn).replace(-999,np.nan)
        q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5)); rh=np.clip(100*e/es,1,100)
        cache[k]=dict(T=s.T2M_MAX.values.astype('float32'),RH=rh.astype('float32'),HI=hi_c(s.T2M_MAX.values,rh).astype('float32'))
    return cache[k]
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
# hourly cell-window data
def hourly(la,lo,t0):
    fn=f'pilot/hourly_cells/{la}_{lo}_{(dates[t0]+pd.Timedelta(days=1)).strftime("%Y%m%d")}.csv'
    if not os.path.exists(fn): return None
    h=pd.read_csv(fn); 
    if len(h)<30: return None
    return h.HI.values[:30]
# ---------- 1. cell-level records (100 km and other footprints), with hourly where available
rec=[]; pairs=[]
for r in lf.itertuples():
    t0=np.searchsorted(dates.values,r.time.normalize().to_datetime64())
    for k,f in foot.items():
        for (a,b) in cells_within(r.lat,r.lon,f(r)):
            s=series(a,b)
            if s is None: continue
            w=s['HI'][t0+1:t0+31]; d35=(w>=35).sum(); d406=(w>=40.6).sum(); m406=(w>=40.6)
            h=hourly(a,b,t0) if (k=='100' and r.season>=2001) else None
            if h is not None:
                h406=(h>=40.6).sum(); h35=(h>=35).sum(); hm=(h>=40.6); pairs.append((d35,d406,h35,h406,pop_cell(a,b,r.season)))
            else: h406=h35=np.nan; hm=None
            rec.append(dict(idx=r.index,season=r.season,iso2=r.iso2,lmic=r.lmic,foot=k,cell=(a,b),t0=t0,pop=pop_cell(a,b,r.season),d35=d35,d406=d406,h35=h35,h406=h406,m406=m406,hm=hm))
R=pd.DataFrame(rec); pr=pd.DataFrame(pairs,columns=['d35','d406','h35','h406','pop'])
r35=pr.h35.sum()/pr.d35.sum(); r406=pr.h406.sum()/pr.d406.sum(); rp406=(pr.h406*pr['pop']).sum()/(pr.d406*pr['pop']).sum(); rp35=(pr.h35*pr['pop']).sum()/(pr.d35*pr['pop']).sum()
print('cell-level calibration: days35',round(r35,3),'days406',round(r406,3),'person-days406',round(rp406,3),'person-days35',round(rp35,3),'n cell-windows',len(pr))
# calibrated days: hourly where available, else approx*ratio (person-day ratio for exposure)
R['c406']=np.where(R.h406.notna(),R.h406,R.d406*rp406); R['c35']=np.where(R.h35.notna(),R.h35,R.d35*rp35)
R['pd406']=R['pop']*R.c406; R['pd35']=R['pop']*R.c35
R['period']=pd.cut(R.season,[1980,2000,2017,2024],labels=['1981-2000','2001-2017','2018-2024']); ny={'1981-2000':20,'2001-2017':17,'2018-2024':7}
S=R.groupby(['foot','lmic','period'],observed=True).agg(pd406=('pd406','sum'),pd35=('pd35','sum')).reset_index()
for c in ['pd406','pd35']: S[c]=S[c]/S.period.map(ny).astype(float)/1e6
print(S.pivot_table(index=['foot','lmic'],columns='period',values='pd406').round(0))
# unique person-days (100 km): day sets per cell; hourly mask where available else approx mask scaled
U={}
for row in R[R.foot=='100'].itertuples():
    m=row.hm if row.hm is not None else row.m406; scale=1.0 if row.hm is not None else rp406
    dd=U.setdefault((row.cell,row.lmic),{})
    for di in np.nonzero(m)[0]: dd[row.t0+1+di]=max(dd.get(row.t0+1+di,0),row.pop*scale)
uq=[]
for (cell,lm),dd in U.items():
    for di,v in dd.items(): uq.append((lm,yr[di],v))
UQ=pd.DataFrame(uq,columns=['lmic','year','pd']); UQ['period']=pd.cut(UQ.year,[1980,2000,2017,2024],labels=list(ny))
uqt=UQ.groupby(['lmic','period'],observed=True).pd.sum().unstack()/pd.Series(ny)/1e6; print('UNIQUE calibrated person-days 406, 100 km (M/yr):'); print(uqt.round(0))
S.to_csv('pilot/recon_exposure_by_footprint.csv',index=False); uqt.to_csv('pilot/recon_unique_100km.csv')
# landfall-level bootstrap of LMIC 100 km totals and ratio
e=R[(R.foot=='100')&R.lmic].groupby(['idx','period'],observed=True).pd406.sum().reset_index(); bs=[]
for _ in range(500):
    s=pd.concat([e[e.period==p].sample((e.period==p).sum(),replace=True) for p in ny]); t=s.groupby('period',observed=True).pd406.sum()/pd.Series(ny)/1e6; bs.append([t['1981-2000'],t['2018-2024'],t['2018-2024']/t['1981-2000']])
bs=np.array(bs); print('bootstrap (landfalls): early',np.percentile(bs[:,0],[2.5,97.5]).round(0),' late',np.percentile(bs[:,1],[2.5,97.5]).round(0),' ratio',np.percentile(bs[:,2],[2.5,97.5]).round(2))
R.drop(columns=['m406','hm']).to_pickle('pilot/recon_records.pkl')
# ---------- 2. decomposition with exact endpoints (100 km, LMIC, HI>=40.6)
def clim(cell,t0,years):
    s=series(*cell)['HI']; d=dates[t0]; out=[]
    for y in years:
        tt=np.searchsorted(dates.values,pd.Timestamp(year=y,month=d.month,day=min(d.day,28)).to_datetime64()); out.append((s[tt+1:tt+31]>=40.6).sum())
    return np.mean(out)*rp406
yrs={'E':range(1981,2001),'L':range(2018,2025)}
D=R[(R.foot=='100')&R.lmic&R.period.isin(['1981-2000','2018-2024'])].copy(); D['S']=np.where(D.season<=2000,'E','L')
D['cE']=[clim(c,t,yrs['E']) for c,t in zip(D.cell,D.t0)]; D['cL']=[clim(c,t,yrs['L']) for c,t in zip(D.cell,D.t0)]
D['pE']=[pop_cell(*c,1990) for c in D.cell]; D['pL']=[pop_cell(*c,2021) for c in D.cell]
def days(d,H): return d.c406 if (H==d.S).all() else np.clip(d.c406+(d['c'+H]-d['c'+d.S.iloc[0]]),0,None)
def E(S,Pp,H,d=None):
    d=D if d is None else d; s=d[d.S==S]; dy=s.c406 if H==S else np.clip(s.c406+(s['c'+H]-s['c'+S]),0,None)
    return (s['p'+Pp]*dy).sum()/ny['1981-2000' if S=='E' else '2018-2024']/1e6
def shap(d=None):
    fac=['S','P','H']; sh={f:0.0 for f in fac}
    for perm in itertools.permutations(fac):
        st={'S':'E','P':'E','H':'E'}; prev=E(st['S'],st['P'],st['H'],d)
        for f in perm:
            st[f]='L'; cur=E(st['S'],st['P'],st['H'],d); sh[f]+=(cur-prev)/6; prev=cur
    return sh,E('E','E','E',d),E('L','L','L',d)
sh,b0,b1=shap(); tot=b1-b0
print('DECOMP: modelled early',round(b0,0),'late',round(b1,0),'(observed, actual-year pop:',round(D[D.S=="E"].pd406.sum()/20/1e6,0),round(D[D.S=="L"].pd406.sum()/7/1e6,0),')')
print({k:(round(v,0),round(v/tot,3)) for k,v in sh.items()}, 'climate only',round(E('E','E','L'),0))
# bootstrap shares by landfall
ids={S:D[D.S==S].idx.unique() for S in 'EL'}; B=[]
for _ in range(200):
    # resample with replacement: duplicate rows for repeated idx
    pick=np.concatenate([rng.choice(ids[S],len(ids[S])) for S in 'EL']); sub=pd.concat([D[D.idx==i] for i in pick])
    s_,a_,b_=shap(sub); t_=b_-a_; B.append([s_['H']/t_,s_['P']/t_,s_['S']/t_,s_['H'],s_['P'],s_['S']])
B=np.array(B); print('share CIs H,P,S:',[np.percentile(B[:,i],[2.5,97.5]).round(2).tolist() for i in range(3)],' abs CIs:',[np.percentile(B[:,i],[2.5,97.5]).round(0).tolist() for i in range(3,6)])
pd.DataFrame(dict(factor=['climate','population','storms'],abs=[sh['H'],sh['P'],sh['S']],share=[sh['H']/tot,sh['P']/tot,sh['S']/tot],abs_lo=np.percentile(B[:,3:],2.5,axis=0),abs_hi=np.percentile(B[:,3:],97.5,axis=0),share_lo=np.percentile(B[:,:3],2.5,axis=0),share_hi=np.percentile(B[:,:3],97.5,axis=0),early=b0,late=b1)).to_csv('pilot/recon_decomposition.csv',index=False)
# ---------- 3. spatial projections (100 km, 2005-2024 landfalls), unique person-days, delta method per cell
beta={}
def get_beta(cell,m):
    k=(cell,m)
    if k not in beta:
        s=series(*cell); sel=(mon==m)&(yr<=2024); a=pd.Series(s['T'][sel]).groupby(yr[sel]).mean(); beta[k]=np.polyfit(gpi.loc[a.index].values,a.values,1)[0]
    return beta[k]
WL=[1.5,2.0,2.8]; proj={}
A=R[(R.foot=='100')&(R.season>=2005)]
for lm in [True,False]:
    sub=A[A.lmic==lm]; out={}
    for wl in WL+['obs']:
        for tag in (['c','d'] if wl!='obs' else ['o']):
            sets={}; tot=0.0
            for row in sub.itertuples():
                s=series(*row.cell); t0=row.t0; Tw=s['T'][t0+1:t0+31]; RHw=s['RH'][t0+1:t0+31]
                if wl=='obs': hi=s['HI'][t0+1:t0+31]
                else:
                    dT=get_beta(row.cell,mon[t0])*(wl-gpi[row.season]); hi=hi_c(Tw+dT, RHw if tag=='c' else np.clip(RHw-dT,1,100))
                m=(hi>=40.6); dd=sets.setdefault(row.cell,{})
                for di in np.nonzero(m)[0]: dd[t0+1+di]=max(dd.get(t0+1+di,0),row.pop*rp406)
            u=sum(v for dd in sets.values() for v in dd.values())/20/1e6
            out[f'{wl}_{tag}']=u
    proj['LMIC' if lm else 'HIC']=out; print('PROJ',('LMIC' if lm else 'HIC'),{k:round(v,0) for k,v in out.items()})
pd.DataFrame(proj).to_csv('pilot/recon_projections.csv')
# ---------- 4. day-by-day profile (landfall cell, HI>=35 and 40.6 calibrated)
prof=[]
for r in lf.itertuples():
    s=series(np.round(r.lat*2)/2,np.round(r.lon*2)/2); t0=np.searchsorted(dates.values,r.time.normalize().to_datetime64())
    w=s['HI'][t0-10:t0+31]
    for k,v in enumerate(w): prof.append((r.season,r.lmic,k-10,v>=35,v>=40.6))
PR=pd.DataFrame(prof,columns=['season','lmic','day','h35','h406']); PR['period']=pd.cut(PR.season,[1980,2000,2017,2024],labels=list(ny))
PR.groupby(['period','day'],observed=True)[['h35','h406']].mean().to_csv('pilot/recon_daily_profile.csv'); print('profile done')
