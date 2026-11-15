# ERA5-led rebuild of the exposure analysis (hourly 0.25 deg ERA5 at the 0.5 deg cell centres; heat index from 2 m T and dewpoint;
# daily maxima over local calendar days; days 1-30 after landfall). Same records, population and de-duplication as the NASA POWER analysis.
import pandas as pd, numpy as np, os, json, sys, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
D0=pd.Timestamp('1981-01-01'); NB=44                      # UTC days t0-11 .. t0+32
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[R.foot=='100'].reset_index(drop=True)
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); pid={k:p for k,p in zip(pts.key,pts.pid)}
R['pid']=[pid[f'{a}_{b}'] for a,b in R.cell]; R['lon']=[b for a,b in R.cell]; R['lat']=[a for a,b in R.cell]
YMAX=int(os.environ.get('YMAX','2024'))
R=R[R.season<=YMAX].reset_index(drop=True)
n=len(R); T=np.full((n,NB*24),np.nan,'float32'); Td=T.copy()
act={}
for i,t0 in enumerate(R.t0.values):
    for k in range(NB): act.setdefault(int(t0)-11+k,[]).append((i,k))
miss=0
for d in sorted(act):
    f=f'data/era5q/{(D0+pd.Timedelta(days=d)):%Y%m%d}.npz'
    if not os.path.exists(f): miss+=1; continue
    z=np.load(f); ix=dict(zip(z['pid'],range(len(z['pid'])))); TT=z['T']; DD=z['Td']
    for i,k in act[d]:
        j=ix.get(R.pid[i])
        if j is not None: T[i,k*24:(k+1)*24]=TT[:,j]/100; Td[i,k*24:(k+1)*24]=DD[:,j]/100
print('records',n,'missing day files',miss,flush=True)
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
RH=np.clip(100*es(Td)/es(T),1,100)
# local calendar days 1..30 after the landfall date: local hour index offset
off=np.array([int(round(lo/15)) for lo in R.lon])          # local = UTC + off
def local_days(X,how='max',hours=None):
    """X: (n, NB*24) UTC hourly from day t0-11 00 UTC. Returns (n,30) per local day t0+1..t0+30."""
    out=np.full((n,30),np.nan,'float32')
    for o in np.unique(off):
        r=np.where(off==o)[0]; st=(11+1)*24-o                  # UTC hour index where local day t0+1 starts
        blk=X[r][:,st:st+30*24].reshape(len(r),30,24)
        if hours is not None: blk=np.where(hours[None,None,:],blk,np.nan)
        out[r]=np.nanmax(blk,2) if how=='max' else np.nanmin(blk,2)
    return out
HI=hi_c(T,RH); HImax=local_days(HI)
# night after each local day: 18:00 .. 06:00 next day local
def night_min(X):
    out=np.full((n,30),np.nan,'float32')
    for o in np.unique(off):
        r=np.where(off==o)[0]; st=(11+1)*24-o+18
        blk=X[r][:,st:st+30*24].reshape(len(r),30,24)[:,:,:12]; out[r]=np.nanmin(blk,2)
    return out
Tnight=night_min(T)
mon=np.array([[(D0+pd.Timedelta(days=int(t0)+1+k)).month for k in range(30)] for t0 in R.t0.values])
dates=np.array([[int(t0)+1+k for k in range(30)] for t0 in R.t0.values])
R['e35']=(HImax>=35).sum(1); R['e406']=(HImax>=40.6).sum(1); R['valid']=np.isfinite(HImax).all(1)
print('records with complete windows',int(R.valid.sum()),'of',n,flush=True)
# ---- unique cell-day person-days by period
def unique_pd(flag,mask=None,pop=None):
    m=R.valid.values if mask is None else (R.valid.values&mask)
    k=np.repeat(R.pid.values[m],30)*100000+dates[m].ravel(); v=(flag[m]*(R['pop'].values[m][:,None] if pop is None else pop[m][:,None])).ravel()
    yr=np.repeat(R.season.values[m],30); lm=np.repeat(R.lmic.values[m],30)
    df=pd.DataFrame(dict(k=k,v=v,yr=yr,lm=lm)).sort_values('v',ascending=False).drop_duplicates('k')
    return df
per={'1981-2000':(1981,2000),'2001-2017':(2001,2017),'2018-2024':(2018,2024),'2018-2021':(2018,2021),'1981-1991':(1981,1991),'1992-2002':(1992,2002),'2003-2013':(2003,2013),'2014-2024':(2014,2024)}
U=unique_pd((HImax>=40.6).astype('float32'))
res={}
for p,(a,b) in per.items():
    for lm in (True,False):
        res[(p,lm)]=U[(U.yr>=a)&(U.yr<=b)&(U.lm==lm)].v.sum()/(b-a+1)/1e6
print('ERA5 unique person-days HI>=40.6 (million/yr), LMIC:',{p:round(res[(p,True)],1) for p in per},' HIC:',{p:round(res[(p,False)],1) for p in per},flush=True)
U35=unique_pd((HImax>=35).astype('float32'))
print('ERA5 unique person-days HI>=35 (million/yr), LMIC:',{p:round(U35[(U35.yr>=a)&(U35.yr<=b)&(U35.lm)].v.sum()/(b-a+1)/1e6,0) for p,(a,b) in per.items()},flush=True)
# NASA POWER for the same records (calibrated/hourly mix as published) for comparison
print('NASA POWER same records (pd406 column, summed not unique), LMIC:',{p:round(R[(R.season>=a)&(R.season<=b)&R.lmic].pd406.sum()/(b-a+1)/1e6,1) for p,(a,b) in per.items()})
# by country and resampled-year interval for the 2018-2024 / 1981-2000 ratio
kk=np.repeat(R.pid.values[R.valid.values],30)*100000+dates[R.valid.values].ravel(); iso=dict(zip(kk,np.repeat(R.iso2.values[R.valid.values],30)))
U['iso']=U.k.map(iso)
for c in ['CN','IN','VN','PH','BD']:
    print(c,{p:round(U[(U.yr>=a)&(U.yr<=b)&(U.iso==c)].v.sum()/(b-a+1)/1e6,1) for p,(a,b) in list(per.items())[:3]})
ann=U[U.lm].groupby('yr').v.sum().reindex(range(1981,2025),fill_value=0)/1e6; ann.to_csv('pilot/era5_annual_lmic.csv')
rng=np.random.default_rng(5); rat=[ann.loc[rng.choice(range(2018,2025),7)].mean()/ann.loc[rng.choice(range(1981,2001),20)].mean() for _ in range(2000)]
print('ratio 2018-2024 to 1981-2000: %.2f (95%% %.1f-%.1f)'%(ann.loc[2018:2024].mean()/ann.loc[1981:2000].mean(),*np.percentile(rat,[2.5,97.5])),flush=True)
# ---- warm nights after dangerous days
dn=((HImax>=40.6)&(Tnight>=26)).astype('float32'); Un=unique_pd(dn)
for p,(a,b) in per.items():
    num=Un[(Un.yr>=a)&(Un.yr<=b)&Un.lm].v.sum(); den=U[(U.yr>=a)&(U.yr<=b)&U.lm].v.sum()
    print(p,'share of dangerous person-days followed by a night >=26 C: %.0f%%'%(100*num/den if den else np.nan))
R[['idx','season','iso2','lmic','cell','t0','pop','e35','e406','valid']].to_pickle(f'pilot/era5_records_{YMAX}.pkl')
np.savez_compressed(f'pilot/era5_dmax_{YMAX}.npz',HImax=HImax,Tnight=Tnight)
# ---- per-landfall days at the landfall cell (nearest record cell to the landfall point)
R['dist']=[np.hypot(a-L.lat[i],(b-L.lon[i])*np.cos(np.radians(a))) for i,(a,b) in zip(R.idx,R.cell)]
LC=R[R.valid].sort_values('dist').drop_duplicates('idx').set_index('idx')
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
LC['G']=gpi.loc[LC.season].values+0.22*0; LC['reg']=[f'{int(np.floor(L.lat[i]/2))}_{int(np.floor(L.lon[i]/2))}' for i in LC.index]; LC['iso']=L.iso2.reindex(LC.index).values
import statsmodels.formula.api as smf
for v in ('e35','e406'):
    m=smf.ols(f'{v}~G+C(reg)',LC).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(LC.iso)[0]})
    print(f'ERA5 landfall-cell days {v} per 1 C: %.2f (%.2f to %.2f), n=%d'%(m.params.G,*m.conf_int().loc['G'],len(LC)))
for p,(a,b) in per.items():
    x=LC[(LC.season>=a)&(LC.season<=b)]; print(p,'mean days HI>=35 %.1f, share with >=1 day HI>=40.6 %.1f%%'%(x.e35.mean(),100*(x.e406>0).mean()))
print('DONE',flush=True)

# ---- attribution (15 hist-nat realisations) and warming levels (26 models) on the 2005-2024 records, hourly shifts
if os.environ.get('SHIFTS','1')=='1':
    cells=np.load('data/cmip6/cells4276.npy'); ck={(round(a,1),round(b,1)):i for i,(a,b) in enumerate(cells)}
    sel=np.where(R.valid.values&(R.season.values>=2005))[0]; ci=np.array([ck.get((round(R.lat[i],1),round(R.lon[i],1)),-1) for i in sel])
    print('2005-2024 records',len(sel),'without CMIP6 cell',int((ci<0).sum()),flush=True)
    sel=sel[ci>=0]; ci=ci[ci>=0]
    hm=np.array([[ (D0+pd.Timedelta(days=int(R.t0[i])-11)+pd.Timedelta(hours=h)).month-1 for h in range(0,NB*24,24)] for i in sel]); hm=np.repeat(hm,24,1)
    Ts=T[sel]; RHs=RH[sel]; G=pd.read_csv('pilot/gmst.csv'); G=G[G.Source=='GISTEMP'].set_index('Year').Mean; gpi=G-G.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
    yrs=R.season.values[sel]; fac=(gsm.loc[yrs].values/gsm.loc[1995:2014].mean())[:,None]; gobs=gsm.loc[yrs].values[:,None]
    Rs=R.iloc[sel].reset_index(drop=True); lc=Rs.index.isin(Rs.reset_index().merge(LC.reset_index()[['idx','pid']],on=['idx','pid'])['index'])
    def run(dT,dH,mult):
        h=hi_c(Ts+dT[hm,ci[:,None]]*mult,np.clip(RHs+dH[hm,ci[:,None]]*mult,1,100))
        full=np.full((n,NB*24),np.nan,'float32'); full[sel]=h; m=local_days(full)[sel]
        f=np.zeros((n,30),'float32'); f[sel]=(m>=40.6); u=unique_pd(f,mask=np.isin(np.arange(n),sel))
        return u[u.lm].v.sum()/20/1e6, (m[lc]>=35).sum(1).mean()
    base=run(np.zeros((12,len(cells)),'float32'),np.zeros((12,len(cells)),'float32'),0.0); print('observed 2005-2024: LMIC person-days HI>=40.6 %.1f M/yr; landfall-cell days HI>=35 %.2f'%base,flush=True)
    A=np.load('data/cmip6/humandelta15.npz'); att=[]
    for k,key in enumerate(A['keys'] if os.environ.get('ATT','1')=='1' else []):
        v=run(A['dT'][k],A['dH'][k],-fac); att.append((key,1-v[0]/base[0],1-v[1]/base[1]))
    At=pd.DataFrame(att,columns=['key','pd406','d35'])
    if len(At): At['model']=At.key.str.split('|').str[0]; At.to_csv('pilot/era5_attribution.csv',index=False)
    if len(At): print('ERA5 attribution: person-days HI>=40.6 %.0f%% (5-95%% %.0f-%.0f); landfall-cell days HI>=35 %.0f%% (%.0f-%.0f); model means'%(100*At.pd406.mean(),100*At.pd406.quantile(.05),100*At.pd406.quantile(.95),100*At.d35.mean(),100*At.d35.quantile(.05),100*At.d35.quantile(.95)),At.groupby('model')[['pd406','d35']].mean().round(2).to_dict(),flush=True)
    S=np.load('data/cmip6/slopes26.npz'); pr=[]
    for k,m in enumerate(S['models']):
        for wl in [float(x) for x in os.environ.get('WLS','1.5,2.0,2.8').split(',')]:
            v=run(S['bT'][k],S['bH'][k],(wl-gobs)); pr.append((m,wl,v[0],v[1]))
    Pr=pd.DataFrame(pr,columns=['model','wl','lmic_pd406','d35']); Pr.to_csv(os.environ.get('PROJOUT','pilot/era5_projections.csv'),index=False)
    print('ERA5 projections, LMIC person-days HI>=40.6 (median, 5-95% over models):',{wl:(round(g.lmic_pd406.median()),round(g.lmic_pd406.quantile(.05)),round(g.lmic_pd406.quantile(.95))) for wl,g in Pr.groupby('wl')},flush=True)
print('DONE SHIFTS',flush=True)
