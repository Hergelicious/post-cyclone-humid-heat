# Measured power outages (EAGLE-I, county, 15 min) and hourly ERA5 humid heat after US/PR landfalls, 2015-2024.
# Builds storm-attributable county outage series, counts customer-hours without power on hazardous-heat days and hours,
# and evaluates the same outages under natural-only (15 CMIP6 realisations) and warming-level (26 models) climates,
# including the restoration speed needed to hold hot-day customer-hours at a reference level.
import pandas as pd, numpy as np, os, glob, sys, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
GAP=int(sys.argv[1]) if len(sys.argv)>1 else 4        # max gap (15-min intervals) filled by interpolation inside an outage
TAG=sys.argv[2] if len(sys.argv)>2 else 'main'
D0=pd.Timestamp('1981-01-01'); H=30*24                  # window: landfall -24 h to +30 days
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county'].copy(); cty['fips']=cty.key.str.zfill(5)
pid_of=dict(zip(cty.fips,cty.pid)); lat_of=dict(zip(cty.fips,cty.lat)); lon_of=dict(zip(cty.fips,cty.lon))
MCC=pd.read_csv('data/eaglei/MCC.csv',encoding='utf-8-sig'); MCC['fips']=MCC.County_FIPS.astype(str).str.zfill(5); cust=dict(zip(MCC.fips,MCC.Customers))
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False)
US=L[L.iso2.isin(['US','PR'])&(L.season>=2015)].sort_values('time').reset_index()
if os.environ.get('YMAX'): US=US[US.season<=int(os.environ['YMAX'])]
NAMES={'2017260N12310':'Maria','2022266N12294':'Ian','2018280N18273':'Michael','2020233N14313':'Laura','2021239N17281':'Ida','2024268N17278':'Helene','2017228N14314':'Harvey',
 '2023239N21274':'Idalia','2017242N16333':'Irma','2024279N21265':'Milton','2020256N25281':'Sally','2020279N16284':'Delta','2024253N21266':'Francine','2018242N13343':'Florence',
 '2020299N18277':'Zeta','2017277N11279':'Nate','2020211N13306':'Isaias','2020205N26272':'Hanna','2016242N24279':'Hermine','2024216N20284':'Debby','2024181N09320':'Beryl',
 '2022311N21293':'Nicole','2019192N29274':'Barry','2021256N21265':'Nicholas','2023265N29284':'Ophelia','2018246N22283':'Gordon','2021182N09317':'Elsa','2021222N14301':'Fred'}
# ---- EAGLE-I rows
_EC={}
def eaglei(a,b):
    out=[]
    for y in range(a.year,b.year+1):
        if y not in _EC:
            if len(_EC)>1: _EC.pop(min(_EC))
            f=f'data/eaglei/win_{y}.parquet'
            if not os.path.exists(f): _EC[y]=None; continue
            x=pd.read_parquet(f,columns=['fips_code','customers_out','t']); x['fips']=x.fips_code.str.zfill(5).astype('category'); x=x.drop(columns='fips_code')
            _EC[y]=x.drop_duplicates(['fips','t'])
        x=_EC[y]
        if x is not None: out.append(x[(x.t>=a)&(x.t<b)])
    return pd.concat(out) if out else pd.DataFrame(columns=['fips','customers_out','t'])
cov=pd.read_csv('data/eaglei/coverage_history.csv'); cov['yr']=pd.to_datetime(cov.year,format='%m/%d/%y').dt.year
ST={'01':'AL','12':'FL','13':'GA','22':'LA','28':'MS','37':'NC','45':'SC','48':'TX','51':'VA','47':'TN','72':'PR','15':'HI','05':'AR','24':'MD','10':'DE','11':'DC','34':'NJ','36':'NY','42':'PA','54':'WV','21':'KY','40':'OK','29':'MO','17':'IL','18':'IN','39':'OH','09':'CT','25':'MA','44':'RI','33':'NH','50':'VT','23':'ME'}
covd={(r.state,r.yr):r.max_pct_covered for r in cov.itertuples()}
# ---- ERA5 hourly at county points
dayfiles={}
def era5_day(d):
    if d not in dayfiles:
        f=f'data/era5q/{(D0+pd.Timedelta(days=d)):%Y%m%d}.npz'
        z=np.load(f); dayfiles[d]=(dict(zip(z['pid'],range(len(z['pid'])))),z['T'],z['Td'])
    return dayfiles[d]
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
def heat_block(fips_list,t0,nh):
    """hourly T and RH (UTC hours from t0) for counties"""
    d_first=(t0.normalize()-D0).days; nd=int(np.ceil((t0-t0.normalize()+pd.Timedelta(hours=nh))/pd.Timedelta(days=1)))+1
    Tm=np.full((nd*24,len(fips_list)),np.nan,'float32'); Dm=Tm.copy()
    pids=[pid_of[f] for f in fips_list]
    for k in range(nd):
        try: idx,T,Td=era5_day(d_first+k)
        except FileNotFoundError: continue
        col=[idx.get(p,-1) for p in pids]; ok=np.array(col)>=0
        Tm[k*24:(k+1)*24,ok]=T[:,np.array(col)[ok]]/100; Dm[k*24:(k+1)*24,ok]=Td[:,np.array(col)[ok]]/100
    off=int((t0-t0.normalize())/pd.Timedelta(hours=1)); Tm=Tm[off:off+nh]; Dm=Dm[off:off+nh]
    return Tm,np.clip(100*es(Dm)/es(Tm),1,100)
def local_daymax(X,t0,lons):
    """per-hour value of the maximum over that hour's local calendar day (local time = UTC + round(lon/15) h)"""
    nh,nc=X.shape; out=np.empty_like(X); h0=int((t0-t0.normalize())/pd.Timedelta(hours=1)); offs=np.array([round(lo/15) for lo in lons])
    for o in np.unique(offs):
        j=np.where(offs==o)[0]; sh=(h0+o)%24; n=int(np.ceil((nh+sh)/24))*24
        P=np.full((n,len(j)),-np.inf,X.dtype); P[sh:sh+nh]=X[:,j]; P=np.where(np.isnan(P),-np.inf,P)
        m=P.reshape(n//24,24,len(j)).max(1); out[:,j]=np.repeat(m,24,0)[sh:sh+nh]
    return out
# ---- CMIP6 fields at nearest extracted cell
cells=np.load('data/cmip6/cells4276.npy'); S=np.load('data/cmip6/slopes26.npz'); A=np.load('data/cmip6/humandelta15.npz')
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
def nearest_cell(la,lo):
    d=np.sqrt((cells[:,0]-la)**2+(((cells[:,1]-lo+180)%360-180)*np.cos(np.radians(la)))**2)*111; k=int(d.argmin()); return k,d[k]
# ---- outage series per storm and county
def county_series(rows,t0,nh):
    """hourly mean customers out from 15-min rows; missing quarter-hours filled inside outages up to GAP intervals, else 0"""
    q=pd.Series(rows.customers_out.values,index=rows.t.values).sort_index()
    grid=pd.date_range(t0,periods=nh*4,freq='15min'); s=q.reindex(grid)
    if GAP>0: s=s.interpolate(limit=GAP,limit_area='inside')
    return s.fillna(0).values.reshape(nh,4).mean(1)
NEXT={i:[(US.time[j].floor('h'),US.lat[j],US.lon[j]) for j in range(i+1,len(US)) if US.time[j]-US.time[i]<pd.Timedelta(days=31)] for i in range(len(US))}
recs=[]; curves={}; storm_rows=[]
WLs=[1.5,2.0,2.8]; FS=np.round(np.arange(0.3,1.51,0.02),2)
for si,r in US.iterrows():
    nm=NAMES.get(r.track_id,r.track_id); lf=r.time.floor('h'); t0=lf-pd.Timedelta(hours=24); nh=H+24
    hi=180*24 if nm=='Maria' else nh
    W=eaglei(t0-pd.Timedelta(days=13),t0+pd.Timedelta(hours=hi)); W=W[W.fips.astype(str).isin(pid_of)]; W['fips']=W.fips.astype(str)
    if W.empty: storm_rows.append(dict(storm=nm,note='no EAGLE-I rows')); continue
    # candidate counties: within 800 km of landfall point
    near={f for f in W.fips.unique() if np.hypot(lat_of[f]-r.lat,(lon_of[f]-r.lon)*np.cos(np.radians(r.lat)))*111<800}; W=W[W.fips.isin(near)]
    out={}
    for f,x in W.groupby('fips'):
        base_rows=x[x.t<t0]; s_base=county_series(base_rows,t0-pd.Timedelta(days=13),12*24) if len(base_rows) else np.zeros(12*24)
        s=county_series(x[x.t>=t0],t0,nh); b=np.median(s_base); ex=np.clip(s-b,0,None)
        c=cust.get(f,np.nan)
        if not np.isfinite(c) or c<=0: continue
        pk=ex[:24*5].max()                                  # peak in landfall -24 h .. +96 h
        if pk<max(float(os.environ.get('THR','0.05'))*c,100): continue
        # storm episode: ends at the first 24 consecutive hours after the peak with <=0.5% of customers (min 20) out, or at the next landfall within 800 km
        ip=int(ex[:24*5].argmax()); low=ex<=max(0.005*c,20); run=np.convolve(low.astype(int),np.ones(24,int),'valid')>=24
        k=np.where(run[ip:])[0]
        if len(k): ex[ip+k[0]:]=0.0
        for t_next,la_n,lo_n in NEXT[si]:
            if np.hypot(lat_of[f]-la_n,(lon_of[f]-lo_n)*np.cos(np.radians(la_n)))*111<800:
                cut=int((t_next-pd.Timedelta(hours=24)-t0)/pd.Timedelta(hours=1))
                if 0<cut<nh: ex[cut:]=0.0
                break
        out[f]=(ex,c,b)
    if not out: storm_rows.append(dict(storm=nm,note='no county reached 5%')); continue
    fl=sorted(out); Ex=np.stack([out[f][0] for f in fl],1)
    Tm,RH=heat_block(fl,t0,nh); lons=[lon_of[f] for f in fl]
    HI=hi_c(Tm,RH); HIdmax=local_daymax(HI,t0,lons)
    loc_hour=np.array([[(h+round(lo/15))%24 for lo in lons] for h in range(nh)])
    night=(loc_hour>=18)|(loc_hour<6)
    st=[ST.get(f[:2],'') for f in fl]; covv=np.array([covd.get((s_,r.season),np.nan) for s_ in st])
    # ---- climates: observed, natural-only (15), warming levels (26 x 3); hot-day flags HI>=35 and >=40.6
    mon=(t0+pd.to_timedelta(np.arange(nh),unit='h')).month.values-1
    nc=[nearest_cell(lat_of[f],lon_of[f]) for f in fl]; ci=np.array([k for k,_ in nc]); dkm=np.array([d for _,d in nc])
    def flags(dT,dH):  # dT,dH: (12, ncell) monthly shifts
        h=hi_c(Tm+dT[mon][:,ci],np.clip(RH+dH[mon][:,ci],1,100)); m=local_daymax(h,t0,lons); return m>=35,m>=40.6
    fac=gsm.loc[r.season]/gsm.loc[1995:2014].mean(); gobs=gsm.loc[r.season]
    clim={'obs':(HIdmax>=35,HIdmax>=40.6)}
    for k,key in enumerate(A['keys']): clim[f'nat|{key}']=flags(-A['dT'][k]*fac,-A['dH'][k]*fac)
    for k,m in enumerate(S['models']):
        for wl in WLs: clim[f'wl{wl}|{m}']=flags(S['bT'][k]*(wl-gobs),S['bH'][k]*(wl-gobs))
    # ---- metrics
    tt=np.arange(nh)-24.0                                    # hours after landfall
    win=(tt>=-24)&(tt<H)
    Ef=np.empty((len(FS),nh,len(fl)),'float32')
    for a,f in enumerate(FS):                                 # restoration compressed by factor f (f<1 faster)
        src=(tt+24)/f-24
        Ef[a]=np.stack([np.interp(src,tt,Ex[:,j],right=0.0) for j in range(len(fl))],1)*win[:,None]
    def E_of_all(flag): return np.einsum('fhc,hc->f',Ef,flag.astype('float32'))
    row=dict(storm=nm,track_id=r.track_id,season=r.season,landfall=str(lf),wind=r.lf_wind,iso2=r.iso2,counties=len(fl),
             customers_in_counties=sum(out[f][1] for f in fl),peak_out=Ex.sum(1).max(),
             cust_h=(Ex*win[:,None]).sum(),cust_h35=(Ex*(HIdmax>=35)*win[:,None]).sum(),cust_h406=(Ex*(HIdmax>=40.6)*win[:,None]).sum(),
             cust_h_hour35=(Ex*(HI>=35)*win[:,None]).sum(),cust_h_warmnight=(Ex*(night&(Tm>=26))*win[:,None]).sum(),
             cust_h_after72=(Ex*(tt[:,None]>=72)).sum(),cust_h35_after72=(Ex*(HIdmax>=35)*(tt[:,None]>=72)).sum(),
             share_cov_lt50=float(np.nanmean(covv<0.5)),max_km_to_cmip=float(dkm.max()),cust_h_cmip_le400=(Ex*win[:,None])[:,dkm<=400].sum())
    agg=Ex.sum(1); pk=agg.max(); ip=int(agg.argmax())
    for q in (0.5,0.9,0.99):
        after=np.where(agg[ip:]<=(1-q)*pk)[0]; row[f'h_to_{int(q*100)}']=float(tt[ip+after[0]]) if len(after) else np.nan
    for k_,(f35,f406) in clim.items():
        row[f'E35|{k_}']=(Ex*f35*win[:,None]).sum(); row[f'E406|{k_}']=(Ex*f406*win[:,None]).sum()
    # restoration-speed curves under observed, natural-only mean flags and warming levels (model-median flags not averaged: store per climate)
    for k_ in clim:
        curves[(nm,k_,35)]=E_of_all(clim[k_][0]); curves[(nm,k_,406)]=E_of_all(clim[k_][1])
    storm_rows.append(row); recs.append((nm,fl,Ex,HIdmax,tt))
    print(nm,r.season,'counties',len(fl),'peak %.0f'%pk,'Mcust-h %.1f'%(row['cust_h']/1e6),'HI35 %.1f'%(row['cust_h35']/1e6),'HI40.6 %.1f'%(row['cust_h406']/1e6),flush=True)
R=pd.DataFrame(storm_rows); R.to_csv(f'pilot/us_outage_storms_{TAG}.csv',index=False)
pd.to_pickle(dict(curves=curves,FS=FS),f'pilot/us_outage_curves_{TAG}.pkl'); pd.to_pickle(recs,f'pilot/us_outage_series_{TAG}.pkl')
print('done',len(R))
