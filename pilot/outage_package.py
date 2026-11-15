import pandas as pd, numpy as np, glob, os
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def tq(la,lo):
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    idx=pd.date_range('1981-01-01',periods=len(s)); return pd.Series(s.T2M_MAX.values,idx),pd.Series(np.clip(100*e/es,1,100),idx)
M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
# restoration milestones: (day after landfall, fraction of affected customers still without power); from utility / government reports
MS={'Bhubaneswar':[(0,1),(7.5,0.5),(13,0)],
    'Calcutta':[(0,1),(5,0.05),(7,0)],
    'Houston':[(0,1),(1,0.75),(2,0.60),(3,0.49),(4,0.38),(6,0.20),(10,0)],
    'NewOrleans':[(0,1),(11,0.02),(12,0)],
    'SanJuan':[(0,1),(100,0.45),(170,0.10)],
    'Visakhapatnam':[(0,1),(3,0.8),(14,0)],
    'Zhanjiang':[(0,1),(2,0.22),(5,0)],
    'Maoming':[(0,1),(2,0.22),(5,0)]}
GR={'NewOrleans':'A','SanJuan':'A','Houston':'B','Calcutta':'B','Bhubaneswar':'B','Zhanjiang':'B','Visakhapatnam':'C','Maoming':'C'}
def out(ms,t):
    d,f=np.array(ms).T; return np.interp(t,d,f,right=0 if f[-1]==0 else np.nan)
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
scen={'reported':lambda t,ms:out(ms,t),'25% faster':lambda t,ms:out(ms,t/0.75),'50% faster':lambda t,ms:out(ms,t/0.5),'72 h cap':lambda t,ms:np.where(t<=3,out(ms,t),0.0)}
for r in ev.itertuples():
    ms=MS[r.city]; T,RH=tq(np.round(r.clat*2)/2,np.round(r.clon*2)/2); t0=r.time.normalize()
    end=int(np.ceil(ms[-1][0])); days=np.arange(1,end+1); dts=[t0+pd.Timedelta(days=int(k)) for k in days]
    Tw=T.loc[dts].values; Hw=RH.loc[dts].values; HI=hi_c(Tw,Hw); mo=np.array([d.month for d in dts]); yr=t0.year
    dark=float(np.clip(1-r.min7,0,1)); popk=r.pop*1e3; aff=popk*dark
    for k,d in enumerate(days): curves.append(dict(city=r.city,day=d,out=float(out(ms,d-0.5)),hi=HI[k]))
    base={}
    for sn,f in scen.items():
        o=np.nan_to_num(f(days-0.5,ms))
        base[sn]=dict(pdays=aff*o.sum(),pd35=aff*(o*(HI>=35)).sum(),pd406=aff*(o*(HI>=40.6)).sum(),cd35=(o*(HI>=35)).sum(),cd406=(o*(HI>=40.6)).sum())
    # counterfactual (reported curve)
    o=np.nan_to_num(scen['reported'](days-0.5,ms)); fac=gsm.loc[yr]/gsm.loc[1995:2014].mean(); cf35=[];cf406=[]
    for key,dd in D.items():
        dT,dH=dd[r.cell]; hc=hi_c(Tw-dT[mo-1]*fac,np.clip(Hw-dH[mo-1]*fac,1,100))
        cf35.append((o*(hc>=35)).sum()); cf406.append((o*(hc>=40.6)).sum())
    cf35=np.array(cf35); cf406=np.array(cf406); f35=base['reported']['cd35']; f406=base['reported']['cd406']
    row=dict(city=r.city,date=t0.date(),grade=GR[r.city],lmic=r.lmic,pop_k=r.pop,dark=dark,affected=aff,end_day=end,
             outage_days_per_person=o.sum(),heat35_days_per_person=f35,heat406_days_per_person=f406,
             pd_out=base['reported']['pdays'],pd35=base['reported']['pd35'],pd406=base['reported']['pd406'],
             hn35_mean=cf35.mean(),hn35_lo=np.percentile(cf35,5),hn35_hi=np.percentile(cf35,95),hn406_mean=cf406.mean(),cf35_all=cf35,cf406_all=cf406)
    for sn in ['25% faster','50% faster','72 h cap']: row[f'pd35_{sn}']=base[sn]['pd35']; row[f'pd406_{sn}']=base[sn]['pd406']
    rows.append(row)
O=pd.DataFrame(rows)
O.to_pickle('pilot/outage_package.pkl'); O.drop(columns=['cf35_all','cf406_all']).to_csv('pilot/outage_package.csv',index=False); pd.DataFrame(curves).to_csv('pilot/outage_curves.csv',index=False)
# totals and human share
S=O[O.city!='SanJuan']
for lab,d in [('all eight',O),('excluding San Juan',S),('grade A or B',O[O.grade<'C'])]:
    tot35=d.pd35.sum(); tot406=d.pd406.sum()
    print(lab,'person-days out',round(d.pd_out.sum()/1e6,2),'M; HI>=35',round(tot35/1e6,2),'M; HI>=40.6',round(tot406/1e6,2),'M')
    for sn in ['25% faster','50% faster','72 h cap']: print('   ',sn,'HI>=35 reduction',round(1-d[f'pd35_{sn}'].sum()/tot35,2),' HI>=40.6 reduction',round(1-d[f'pd406_{sn}'].sum()/tot406,2) if tot406 else None)
# human share of heat-exposed outage person-days, member ensemble
w=O.affected.values; f=(O.heat35_days_per_person*w).sum()
# per-member recompute

for lab,d in [('all',O),('LMIC',O[O.lmic]),('HIC',O[~O.lmic])]:
    a=d.affected.values
    for th,fc,cc in [('35','heat35_days_per_person','cf35_all'),('40.6','heat406_days_per_person','cf406_all')]:
        F=(d[fc].values*a).sum(); C=np.array([(np.stack(d[cc].values)[:,j]*a).sum() for j in range(15)])
        if F>0: print(lab,'HI>=',th,'human share mean',round(1-C.mean()/F,2),'member 5-95',np.percentile(1-C/F,[5,95]).round(2),'model-member range',round((1-C/F).min(),2),round((1-C/F).max(),2))
