import pandas as pd, numpy as np, glob, os
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def tq(la,lo):
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    idx=pd.date_range('1981-01-01',periods=len(s)); return pd.Series(s.T2M_MAX.values,idx),pd.Series(np.clip(100*e/es,1,100),idx)
M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
# restoration milestones: (day after landfall, fraction of affected customers still without power); from utility / government reports
MS={'Bhubaneswar':[(0,1),(6,0.63),(7.5,0.5),(12,0.14),(14,0.12)],
    'Calcutta':[(0,1),(5,0.05),(7,0.03)],
    'Houston':[(0,1),(1,0.75),(2,0.60),(3,0.49),(4,0.38),(6,0.20),(10,0)],
    'NewOrleans':[(0,1),(11,0.02),(12,0)],
    'SanJuan':[(0,1),(100,0.45),(170,0.10)],
    'Visakhapatnam':[(0,1),(3,0.8),(14,0)],
    'Zhanjiang':[(0,1),(2,0.22),(5,0)],
    'Maoming':[(0,1),(2,0.22),(5,0)]}
# initial share of customers without power reported by the utility or government (None = not reported)
REP={'SanJuan':1.0,'NewOrleans':1.0,'Houston':0.87,'Bhubaneswar':1.0,'Visakhapatnam':1.0,'Calcutta':None,'Zhanjiang':None,'Maoming':None}
GR={'NewOrleans':'A','SanJuan':'A','Houston':'B','Calcutta':'B','Bhubaneswar':'B','Zhanjiang':'B','Visakhapatnam':'C','Maoming':'C'}
MODE='linear'
def out(ms,t,mode=None):
    mode=mode or MODE; d,f=np.array(ms).T; t=np.asarray(t,float)
    if mode=='linear': v=np.interp(t,d,f)
    else:
        k=np.clip(np.searchsorted(d,t,side='right')-1,0,len(d)-1)   # last milestone at or before t
        v=f[k] if mode=='late' else f[np.clip(k+1,0,len(d)-1)]
    return np.where(t>d[-1],0.0 if f[-1]==0 else np.nan,v)
# CMIP6 deltas (1995-2014 base, scaled to observed warming), nearest model-cell
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean(); gsm=gpi.rolling(11,center=True,min_periods=5).mean()
cells=np.load('data/cmip6/cells4276.npy'); J=pd.read_csv('pilot/attr_jobs.csv')
def loadm(m,e,r):
    f=f'data/cmip6/{m}_{e}_{r}.npz'
    if not os.path.exists(f): return None
    d=dict(np.load(f)); n=min(len(d['gmst']),len(d['tasmax']),len(d['hurs']),len(d['yr'])); return {k:v[:n] for k,v in d.items()}
ev=M[M.city.isin(MS)&(M.min7<0.6)].copy()
ev['cell']=[int(np.argmin((cells[:,0]-a)**2+((cells[:,1]-b+180)%360-180)**2)) for a,b in zip(ev.clat,ev.clon)]
D={}
for (m,r),_ in J.groupby(['m','r']):
    h=loadm(m,'historical',r); n=loadm(m,'hist-nat',r)
    if h is None or n is None: continue
    sh=(h['yr']>=1995)&(h['yr']<=2014); sn=(n['yr']>=1995)&(n['yr']<=2014)
    dG=h['gmst'][sh].mean()-n['gmst'][sn].mean(); sc=gsm.loc[1995:2014].mean()/dG
    D[(m,r)]={c:(np.array([h['tasmax'][sh&(h['mo']==k),c].mean()-n['tasmax'][sn&(n['mo']==k),c].mean() for k in range(1,13)])*sc,
                 np.array([h['hurs'][sh&(h['mo']==k),c].mean()-n['hurs'][sn&(n['mo']==k),c].mean() for k in range(1,13)])*sc) for c in ev.cell.unique()}
print('members',len(D))
rows=[]; curves=[]
scen={'reported':lambda t,ms,m:out(ms,t,m),'25% faster':lambda t,ms,m:out(ms,t/0.75,m),'50% faster':lambda t,ms,m:out(ms,t/0.5,m),'72 h cap':lambda t,ms,m:np.where(t<=3,out(ms,t,m),0.0)}
keys=list(D)
def hourly_city(city,la,lo,dts):
    h=pd.read_csv(f'pilot/outage_hourly/{city}_{la}_{lo}.csv'); h['d']=pd.to_datetime(dict(year=h.YEAR,month=h.MO,day=h.DY))
    g={d:(x.T2M.values,np.clip(x.RH2M.values,1,100)) for d,x in h.groupby('d')}
    return [g.get(d,(np.array([np.nan]),np.array([np.nan]))) for d in dts]
def dmax(hh,dT=None,dH=None,fac=1.0,mo=None):
    out=[]
    for k,(t,rh) in enumerate(hh):
        if dT is None: out.append(np.nanmax(hi_c(t,rh)))
        else: out.append(np.nanmax(hi_c(t-dT[mo[k]-1]*fac,np.clip(rh-dH[mo[k]-1]*fac,1,100))))
    return np.array(out)
for r in ev.itertuples():
    ms=MS[r.city]; T,RH=tq(np.round(r.clat*2)/2,np.round(r.clon*2)/2); t0=r.time.normalize()
    end=int(np.ceil(ms[-1][0])); days=np.arange(1,end+1); dts=[t0+pd.Timedelta(days=int(k)) for k in days]
    Tw=T.loc[dts].values; Hw=RH.loc[dts].values; HId=hi_c(Tw,Hw); mo=np.array([d.month for d in dts]); yr=t0.year
    la0=round(float(np.round(r.clat*2)/2),1); lo0=round(float(np.round(r.clon*2)/2),1); HH=hourly_city(r.city,la0,lo0,dts); HI=dmax(HH)
    NB=[dmax(hourly_city(r.city,round(la0+a,1),round(lo0+b,1),dts)) for a in (-0.5,0,0.5) for b in (-0.5,0,0.5)]
    ntl=float(np.clip(1-r.min7,0,1)); rep=REP[r.city]; popk=r.pop*1e3
    for k,d in enumerate(days): curves.append(dict(city=r.city,day=d,out=float(out(ms,d-0.5,'linear')),hi=HI[k],hi_daily=HId[k]))
    row=dict(city=r.city,date=t0.date(),grade=GR[r.city],lmic=r.lmic,pop=popk,share_ntl=ntl,share_rep=rep if rep is not None else np.nan,
             aff_low=popk*ntl,aff_high=popk*(rep if rep is not None else 1.0),end_day=end)
    for mode in ['linear','early','late']:
        for sn,f in scen.items():
            o=np.nan_to_num(f(days-0.5,ms,mode)); tag=f'{mode}|{sn}'
            row[f'od|{tag}']=o.sum(); row[f'h35|{tag}']=(o*(HI>=35)).sum(); row[f'h406|{tag}']=(o*(HI>=40.6)).sum()
    o=np.nan_to_num(out(ms,days-0.5,'linear')); fac=gsm.loc[yr]/gsm.loc[1995:2014].mean()
    row['hi_out']=(o*HI).sum()/o.sum(); row['hi_out_max']=np.max(HI[o>0.05]) if (o>0.05).any() else np.nan
    row['h35_dailyapprox']=(o*(HId>=35)).sum(); row['h406_dailyapprox']=(o*(HId>=40.6)).sum()
    nb35=[(o*(x>=35)).sum() for x in NB]; nb406=[(o*(x>=40.6)).sum() for x in NB]
    row['h35_nb_min']=min(nb35); row['h35_nb_max']=max(nb35); row['h406_nb_min']=min(nb406); row['h406_nb_max']=max(nb406)
    for key in keys:
        dT,dH=D[key][r.cell]; hc=dmax(HH,dT,dH,fac,mo)
        row[f'cf35|{key[0]}|{key[1]}']=(o*(hc>=35)).sum(); row[f'cf406|{key[0]}|{key[1]}']=(o*(hc>=40.6)).sum(); row[f'dhi|{key[0]}|{key[1]}']=(o*(HI-hc)).sum()/o.sum()
    rows.append(row)
O=pd.DataFrame(rows); O.to_pickle('pilot/outage_package2.pkl'); pd.DataFrame(curves).to_csv('pilot/outage_curves.csv',index=False)
cf35c=[c for c in O if c.startswith('cf35|')]; cf406c=[c for c in O if c.startswith('cf406|')]
def agg(d,aff,mode='linear'):
    a=d[aff].values; g=lambda c:(d[c].values*a).sum()/1e6
    res=dict(n=len(d),aff=a.sum()/1e6,pd_out=g(f'od|{mode}|reported'),pd35=g(f'h35|{mode}|reported'),pd406=g(f'h406|{mode}|reported'))
    for sn in ['25% faster','50% faster','72 h cap']:
        res[f'red35 {sn}']=1-g(f'h35|{mode}|{sn}')/res['pd35'] if res['pd35'] else np.nan
        res[f'red406 {sn}']=1-g(f'h406|{mode}|{sn}')/res['pd406'] if res['pd406'] else np.nan
    if mode=='linear':
        C35=np.array([g(c) for c in cf35c]); C406=np.array([g(c) for c in cf406c])
        res.update(att35=res['pd35']-C35.mean(),att35_lo=res['pd35']-np.percentile(C35,95),att35_hi=res['pd35']-np.percentile(C35,5),sh35=1-C35.mean()/res['pd35'],sh35_lo=1-np.percentile(C35,95)/res['pd35'],sh35_hi=1-np.percentile(C35,5)/res['pd35'])
        if res['pd406']: res.update(att406=res['pd406']-C406.mean(),sh406=1-C406.mean()/res['pd406'],sh406_lo=1-np.percentile(C406,95)/res['pd406'],sh406_hi=1-np.percentile(C406,5)/res['pd406'])
    return res
sets=[('All eight',O),('Without San Juan',O[O.city!='SanJuan']),('Without grade C (A and B only)',O[O.grade!='C']),('Without Kolkata',O[O.city!='Calcutta']),('LMIC cities',O[O.lmic]),('High-income cities',O[~O.lmic])]
R=[]
for aff in ['aff_low','aff_high']:
    for lab,d in sets: R.append(dict(population=aff,set=lab,mode='linear',**agg(d,aff)))
    for mode in ['early','late']: R.append(dict(population=aff,set='All eight',mode=mode,**agg(O,aff,mode)))
R=pd.DataFrame(R); pd.set_option('display.width',300); print(R.round(2).to_string()); R.to_csv('pilot/outage_sensitivity2.csv',index=False)
# per model (mean over its realisations), all eight, both populations
PM=[]
for aff in ['aff_low','aff_high']:
    a=O[aff].values; F35=(O['h35|linear|reported']*a).sum(); F406=(O['h406|linear|reported']*a).sum()
    for m in sorted(set(k[0] for k in keys)):
        c35=np.mean([(O[f'cf35|{m}|{rr}']*a).sum() for (mm,rr) in keys if mm==m]); c406=np.mean([(O[f'cf406|{m}|{rr}']*a).sum() for (mm,rr) in keys if mm==m])
        PM.append(dict(population=aff,model=m,att35=(F35-c35)/1e6,sh35=1-c35/F35,att406=(F406-c406)/1e6,sh406=1-c406/F406))
PM=pd.DataFrame(PM); print(PM.round(2).to_string()); PM.to_csv('pilot/outage_attr_models.csv',index=False)
# per city table
T=[]
for r in O.itertuples():
    C=np.array([getattr(r,'_asdict')()[c] if False else O.loc[r.Index,c] for c in cf35c]); f=O.loc[r.Index,'h35|linear|reported']
    T.append(dict(city=r.city,grade=r.grade,share_ntl=r.share_ntl,share_rep=r.share_rep,aff_low=r.aff_low/1e6,aff_high=r.aff_high/1e6,days_out=O.loc[r.Index,'od|linear|reported'],
                  h35=f,h406=O.loc[r.Index,'h406|linear|reported'],att35_days=f-C.mean(),sh35=(1-C.mean()/f) if f else np.nan,
                  red25=1-O.loc[r.Index,'h35|linear|25% faster']/f if f else np.nan,red50=1-O.loc[r.Index,'h35|linear|50% faster']/f if f else np.nan,red72=1-O.loc[r.Index,'h35|linear|72 h cap']/f if f else np.nan,
                  hi_out=r.hi_out,hi_out_max=r.hi_out_max,dhi=np.mean([O.loc[r.Index,c] for c in O if c.startswith('dhi|')]),dhi_lo=np.percentile([O.loc[r.Index,c] for c in O if c.startswith('dhi|')],5),dhi_hi=np.percentile([O.loc[r.Index,c] for c in O if c.startswith('dhi|')],95),
                  h35_early=O.loc[r.Index,'h35|early|reported'],h35_late=O.loc[r.Index,'h35|late|reported'],
                  h35_dailyapprox=O.loc[r.Index,'h35_dailyapprox'],h406_dailyapprox=O.loc[r.Index,'h406_dailyapprox'],h35_nb_min=r.h35_nb_min,h35_nb_max=r.h35_nb_max,h406_nb_min=r.h406_nb_min,h406_nb_max=r.h406_nb_max))
T=pd.DataFrame(T); print(T.round(2).to_string()); T.to_csv('pilot/outage_city2.csv',index=False)
