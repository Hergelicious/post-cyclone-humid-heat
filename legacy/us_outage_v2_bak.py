# Measured power outages (EAGLE-I, county, 15 min) and hourly ERA5 humid heat after US/PR landfalls, 2015-2024, version 2.
# Adds simultaneous metrics (customer-hours in hours with HI above threshold), customers without power at the hottest hour of
# hazardous days (customer-days), warm-night hours, exposure after 24/48/72 h of continuous outage, customers facing several
# hazardous peaks, county-point CMIP6 changes, track distance and data-quality fields for robustness tests.
# usage: python3 us_outage_v2.py [GAP] [TAG] ; env THR (default 0.05), RADIUS (km, default 800), CM (cty|cell)
import pandas as pd, numpy as np, os, sys, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
src=open('pilot/us_outage_heat.py').read()
exec(src[src.index('GAP=int'):src.index('# ---- CMIP6 fields')])
RADIUS=float(os.environ.get('RADIUS','800')); CM=os.environ.get('CM','cty'); THR=float(os.environ.get('THR','0.05'))
exec(src[src.index('def county_series'):src.index('NEXT=')])
NEXT={i:[(US.time[j].floor('h'),US.lat[j],US.lon[j]) for j in range(i+1,len(US)) if US.time[j]-US.time[i]<pd.Timedelta(days=31)] for i in range(len(US))}
# ---- CMIP6: county points (default) or nearest landfall cell (previous version)
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
if CM=='cty':
    S=np.load('data/cmip6cty/slopes26.npz'); A=np.load('data/cmip6cty/humandelta15.npz'); cf=list(np.load('data/cmip6cty/fips.npy',allow_pickle=True)); cidx={f:k for k,f in enumerate(cf)}
    def cmip_index(f): return cidx[f],0.0
else:
    cells=np.load('data/cmip6/cells4276.npy'); S=np.load('data/cmip6/slopes26.npz'); A=np.load('data/cmip6/humandelta15.npz')
    def cmip_index(f):
        la,lo=lat_of[f],lon_of[f]; d=np.sqrt((cells[:,0]-la)**2+(((cells[:,1]-lo+180)%360-180)*np.cos(np.radians(la)))**2)*111; k=int(d.argmin()); return k,d[k]
cellsx=np.load('data/cmip6/cells4276.npy')
def dist_cell(f):
    la,lo=lat_of[f],lon_of[f]; d=np.sqrt((cellsx[:,0]-la)**2+(((cellsx[:,1]-lo+180)%360-180)*np.cos(np.radians(la)))**2)*111; return d.min()
# ---- tracks (JTWC via huracanpy) and quality
TR=pd.read_csv('/usr/local/lib/python3.13/site-packages/huracanpy/_data/_ibtracs_files/jtwc.csv',low_memory=False,parse_dates=['time'])
TR=TR[TR.season>=2014]; TR['lon']=((TR.lon+180)%360)-180
FEMA={'CT':1,'ME':1,'MA':1,'NH':1,'RI':1,'VT':1,'NJ':2,'NY':2,'PR':2,'DE':3,'DC':3,'MD':3,'PA':3,'VA':3,'WV':3,'AL':4,'FL':4,'GA':4,'KY':4,'MS':4,'NC':4,'SC':4,'TN':4,
      'IL':5,'IN':5,'MI':5,'MN':5,'OH':5,'WI':5,'AR':6,'LA':6,'NM':6,'OK':6,'TX':6,'IA':7,'KS':7,'MO':7,'NE':7,'HI':9}
DQ=pd.read_csv('data/eaglei/DQI.csv'); dqi={(r.fema,r.year):r.DQI for r in DQ.itertuples()}
FS=np.round(np.arange(0.3,1.51,0.02),2); WLs=[1.5,2.0,2.8]
recs=[]; curves={}; rows=[]
for si,r in US.iterrows():
    nm=NAMES.get(r.track_id,r.track_id); lf=r.time.floor('h'); t0=lf-pd.Timedelta(hours=24); nh=H+24
    hi=180*24 if nm=='Maria' else nh
    W=eaglei(t0-pd.Timedelta(days=13),t0+pd.Timedelta(hours=hi)); W=W[W.fips.astype(str).isin(pid_of)]; W['fips']=W.fips.astype(str)
    if W.empty: rows.append(dict(storm=nm,note='no EAGLE-I rows')); continue
    near={f for f in W.fips.unique() if np.hypot(lat_of[f]-r.lat,(lon_of[f]-r.lon)*np.cos(np.radians(r.lat)))*111<RADIUS}; W=W[W.fips.isin(near)]
    trk=TR[(TR.track_id==r.track_id)&(TR.time>=lf-pd.Timedelta(hours=24))&(TR.time<=lf+pd.Timedelta(hours=72))]
    out={}; nrep={}
    for f,x in W.groupby('fips'):
        base_rows=x[x.t<t0]; s_base=county_series(base_rows,t0-pd.Timedelta(days=13),12*24) if len(base_rows) else np.zeros(12*24)
        xs=x[x.t>=t0]; s=county_series(xs,t0,nh); b=np.median(s_base); ex=np.clip(s-b,0,None)
        c=cust.get(f,np.nan)
        if not np.isfinite(c) or c<=0: continue
        pk=ex[:24*5].max()
        if pk<max(THR*c,100): continue
        ip=int(ex[:24*5].argmax()); low=ex<=max(0.005*c,20); run=np.convolve(low.astype(int),np.ones(24,int),'valid')>=24
        k=np.where(run[ip:])[0]
        if len(k): ex[ip+k[0]:]=0.0
        for t_next,la_n,lo_n in NEXT[si]:
            if np.hypot(lat_of[f]-la_n,(lon_of[f]-lo_n)*np.cos(np.radians(la_n)))*111<800:
                cut=int((t_next-pd.Timedelta(hours=24)-t0)/pd.Timedelta(hours=1))
                if 0<cut<nh: ex[cut:]=0.0
                break
        out[f]=(ex,c,b)
        qi=np.unique(((xs.t-t0)/pd.Timedelta(minutes=15)).astype(int).values); qi=qi[(qi>=0)&(qi<nh*4)]
        hrs=np.where(ex>0)[0]; need=(hrs[:,None]*4+np.arange(4)[None]).ravel()
        nrep[f]=float(np.isin(need,qi).mean()) if len(need) else np.nan   # share of quarter-hours with a record while the county was out
    if not out: rows.append(dict(storm=nm,note='no county reached threshold')); continue
    fl=sorted(out); Ex=np.stack([out[f][0] for f in fl],1); nc_=len(fl)
    Tm,RH=heat_block(fl,t0,nh); lons=[lon_of[f] for f in fl]
    tt=np.arange(nh)-24.0; win=((tt>=-24)&(tt<H))[:,None]
    loc_hour=(np.arange(nh)[:,None]+int((t0-t0.normalize())/pd.Timedelta(hours=1))+np.array([round(lo/15) for lo in lons])[None])%24
    night=(loc_hour>=18)|(loc_hour<6)
    mon=(t0+pd.to_timedelta(np.arange(nh),unit='h')).month.values-1
    ci=np.array([cmip_index(f)[0] for f in fl]); dcell=np.array([dist_cell(f) for f in fl])
    dtrack=np.array([np.min(np.hypot(trk.lat.values-lat_of[f],(trk.lon.values-lon_of[f])*np.cos(np.radians(lat_of[f]))))*111 if len(trk) else np.nan for f in fl])
    dland=np.array([np.hypot(lat_of[f]-r.lat,(lon_of[f]-r.lon)*np.cos(np.radians(r.lat)))*111 for f in fl])
    st=[ST.get(f[:2],'') for f in fl]; covv=np.array([covd.get((s_,int(r.season)),np.nan) for s_ in st]); dq=np.array([dqi.get((FEMA.get(s_,0),int(r.season)),np.nan) for s_ in st])
    h0=int((t0-t0.normalize())/pd.Timedelta(hours=1)); offs=np.array([round(lo/15) for lo in lons])
    def peak_flag(X,thr):
        """True at the first hour of each local calendar day at which HI reaches that day's maximum, when the maximum >= thr"""
        F=np.zeros(X.shape,bool)
        for o in np.unique(offs):
            j=np.where(offs==o)[0]; sh=(h0+o)%24; n=int(np.ceil((nh+sh)/24))*24
            P=np.full((n,len(j)),-np.inf,'float32'); P[sh:sh+nh]=np.nan_to_num(X[:,j],nan=-np.inf)
            R3=P.reshape(n//24,24,len(j)); am=R3.argmax(1); mx=R3.max(1)
            G=np.zeros(R3.shape,bool); dd,cc=np.meshgrid(np.arange(n//24),np.arange(len(j)),indexing='ij'); G[dd,am,cc]=mx>=thr
            F[:,j]=G.reshape(n,len(j))[sh:sh+nh]
        return F
    def metrics(T_,RH_):
        h=hi_c(T_,RH_); dm=local_daymax(h,t0,lons)
        return {'day35':dm>=35,'day406':dm>=40.6,'hour35':h>=35,'hour406':h>=40.6,'peak35':peak_flag(h,35),'peak406':peak_flag(h,40.6),'night26':night&(T_>=26)}
    clim={'obs':metrics(Tm,RH)}
    fac=gsm.loc[r.season]/gsm.loc[1995:2014].mean(); gobs=gsm.loc[r.season]
    for k,key in enumerate(A['keys']):
        dT=-A['dT'][k]*fac; dH=-A['dH'][k]*fac; clim[f'nat|{key}']=metrics(Tm+dT[mon][:,ci],np.clip(RH+dH[mon][:,ci],1,100))
    for k,m in enumerate(S['models']):
        for wl in WLs:
            dT=S['bT'][k]*(wl-gobs); dH=S['bH'][k]*(wl-gobs); clim[f'wl{wl}|{m}']=metrics(Tm+dT[mon][:,ci],np.clip(RH+dH[mon][:,ci],1,100))
    onset=np.array([int(np.argmax(Ex[:24*5,j]>=max(0.005*out[f][1],20))) for j,f in enumerate(fl)])
    Exw=Ex*win
    # ---- per-county metrics and severity of repeated exposure, for every climate
    day_id=(np.arange(nh)[:,None]+h0+offs[None])//24; hidx=np.arange(nh)
    def severity(F):
        pk=F['peak35']&win&(hidx[:,None]>=onset[None]); n26=F['night26']; out_=np.zeros((nc_,8),'float32')
        for j in range(nc_):
            hs=np.where(pk[:,j])[0]
            if not len(hs): continue
            v=Ex[hs,j]; rm=np.minimum.accumulate(v); dd=day_id[hs,j]
            for kk in range(min(5,len(v))): out_[j,kk]=rm[kk]
            c2=[min(v[i],v[i+1]) for i in range(len(v)-1) if dd[i+1]==dd[i]+1]; out_[j,5]=max(c2) if c2 else 0
            c3=[min(v[i],v[i+1],v[i+2]) for i in range(len(v)-2) if dd[i+1]==dd[i]+1 and dd[i+2]==dd[i]+2]; out_[j,6]=max(c3) if c3 else 0
            dn=0.0
            for i,h in enumerate(hs):
                nt=[x for x in range(h+1,min(h+19,nh)) if ((x+h0+offs[j])%24>=18 or (x+h0+offs[j])%24<6)]
                if nt and n26[nt,j].all(): dn=max(dn,min(v[i],Ex[nt,j].min()))
            out_[j,7]=dn
        return out_
    row=dict(storm=nm,track_id=r.track_id,season=int(r.season),month=lf.month,landfall=str(lf),wind=r.lf_wind,counties=nc_,customers_in_counties=sum(out[f][1] for f in fl),
             peak_out=Ex.sum(1).max(),cust_h=Exw.sum())
    for k_,F in clim.items():
        for mname,flag in F.items(): row[f'{mname}|{k_}']=float((Exw*flag).sum())
    # exposure after 24/48/72 h of continuous outage (customers still out at hour t have been out since the county's onset)
    O=clim['obs']
    el=np.arange(nh)[:,None]-onset[None]
    for D in (24,48,72):
        for mname in ('hour35','peak35','night26','day35'): row[f'{mname}_after{D}'] = float((Exw*O[mname]*(el>=D)).sum())
        row[f'cust_h_after{D}']=float((Exw*(el>=D)).sum())
    # customers without power at k or more hazardous daily peaks (running minimum across successive peaks)
    for thr_ in ('peak35','peak406'):
        F=O[thr_]&win; ck=np.zeros(5)
        for j in range(nc_):
            hs=np.where(F[:,j])[0]
            if not len(hs): continue
            v=np.minimum.accumulate(Ex[hs,j])
            for kk in range(min(5,len(v))): ck[kk]+=v[kk]
        for kk in range(5): row[f'cust_{thr_}_ge{kk+1}']=ck[kk]
    # restoration curves for hourly, peak and hot-day metrics
    Ef=np.empty((len(FS),nh,nc_),'float32')
    for a,f in enumerate(FS):
        srcx=(tt+24)/f-24; Ef[a]=np.stack([np.interp(srcx,tt,Ex[:,j],right=0.0) for j in range(nc_)],1)*win
    for k_,F in clim.items():
        for mname in ('hour35','peak35','day35','hour406','peak406'):
            curves[(nm,k_,mname)]=np.einsum('fhc,hc->f',Ef,F[mname].astype('float32'))
    CTY={k_:dict(hour35=(Exw*F['hour35']).sum(0).astype('float32'),peak35=(Exw*F['peak35']).sum(0).astype('float32'),day35=(Exw*F['day35']).sum(0).astype('float32'),sev=severity(F)) for k_,F in clim.items()}
    rows.append(row)
    recs.append(dict(cty=CTY,storm=nm,season=int(r.season),fips=fl,Ex=Ex.astype('float32'),HI=hi_c(Tm,RH).astype('float32'),T=Tm.astype('float32'),tt=tt,cust=np.array([out[f][1] for f in fl]),
                     dtrack=dtrack,dland=dland,dcell=dcell,cov=covv,dqi=dq,nrep=np.array([nrep[f] for f in fl]),onset=onset,
                     flags={m_:clim['obs'][m_] for m_ in ('day35','hour35','peak35','night26','peak406','hour406','day406')}))
    print(nm,r.season,'counties',nc_,'Mcust-h %.1f'%(row['cust_h']/1e6),'day35 %.1f hour35 %.1f peak35(Mcust-day) %.2f'%(row['day35|obs']/1e6,row['hour35|obs']/1e6,row['peak35|obs']/1e6),flush=True)
pd.DataFrame(rows).to_csv(f'pilot/us_v2_storms_{TAG}.csv',index=False)
pd.to_pickle(dict(curves=curves,FS=FS),f'pilot/us_v2_curves_{TAG}.pkl'); pd.to_pickle(recs,f'pilot/us_v2_recs_{TAG}.pkl')
print('done',len(rows))
