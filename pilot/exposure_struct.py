import pandas as pd, numpy as np
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[R.foot=='100'].reset_index(drop=True)
LF=lf.set_index('index')
rp406=0.953
# masks per record (calibrated scheme as in the headline exposure)
M=[];S=[]
for row in R.itertuples():
    s=series(*row.cell); m=s['HI'][row.t0+1:row.t0+31]>=40.6; sc=rp406
    h=hourly(*row.cell,row.t0) if row.season>=2001 else None
    if h is not None: m=h>=40.6; sc=1.0
    M.append(m); S.append(sc)
print('masks',len(M),flush=True)
# ---------- A. people exposed at least once x days per exposed person (unique cell-days, LMIC)
U={}
for k,row in enumerate(R.itertuples()):
    if not row.lmic: continue
    for di in np.nonzero(M[k])[0]:
        key=(row.cell,row.t0+1+di); v=row.pop*S[k]
        if v>U.get(key,(0,))[0]: U[key]=(v,row.pop,row.iso2)
cd=pd.DataFrame([(c,yr[d],v,p,iso) for (c,d),(v,p,iso) in U.items()],columns=['cell','year','pd','pop','iso2'])
cy=cd.groupby(['cell','year']).agg(pd=('pd','sum'),pop=('pop','max'),iso2=('iso2','first')).reset_index()
ann=cy.groupby('year').agg(PD=('pd','sum'),N=('pop','sum')).reindex(range(1981,2025),fill_value=0)
def per(a,b): x=ann.loc[a:b]; return x.PD.mean()/1e6, x.N.mean()/1e6, x.PD.sum()/x.N.sum()
rows=[]
for lab,(a,b) in [('1981–2000',(1981,2000)),('2001–2017',(2001,2017)),('2018–2024',(2018,2024))]:
    pdm,N,Dd=per(a,b); rows.append(dict(period=lab,person_days=pdm,people_exposed=N,days_per_exposed_person=Dd))
for c in ['CN','IN','VN','PH','BD']:
    for lab,(a,b) in [('1981–2000',(1981,2000)),('2018–2024',(2018,2024))]:
        x=cy[(cy.iso2==c)&(cy.year>=a)&(cy.year<=b)]; ny=b-a+1
        rows.append(dict(period=f'{c} {lab}',person_days=x.pd.sum()/ny/1e6,people_exposed=x['pop'].sum()/ny/1e6,days_per_exposed_person=x.pd.sum()/x['pop'].sum() if len(x) else np.nan))
A=pd.DataFrame(rows); print(A.round(2).to_string()); A.to_csv('pilot/exposure_ND.csv',index=False)
e=A.iloc[0]; l=A.iloc[2]; print('ratio PD',round(l.person_days/e.person_days,2),'N',round(l.people_exposed/e.people_exposed,2),'D',round(l.days_per_exposed_person/e.days_per_exposed_person,2))
# ---------- B. recurrence: cells with dangerous post-landfall days in >=2 (>=3) different years within a decade
rec=[]
for lab,(a,b) in [('1981–1990',(1981,1990)),('1991–2000',(1991,2000)),('2001–2010',(2001,2010)),('2011–2020',(2011,2020)),('2015–2024',(2015,2024))]:
    x=cy[(cy.year>=a)&(cy.year<=b)].groupby('cell').agg(ny=('year','nunique')).reset_index()
    mid=(a+b)//2+1; x['pop']=[pop_cell(*c,mid) for c in x.cell]
    rec.append(dict(decade=lab,cells_any=len(x),people_any=x['pop'].sum()/1e6,people_2plus=x[x.ny>=2]['pop'].sum()/1e6,people_3plus=x[x.ny>=3]['pop'].sum()/1e6))
B=pd.DataFrame(rec); print(B.round(1).to_string()); B.to_csv('pilot/exposure_recurrence.csv',index=False)
# ---------- C. finer population allocation: 0.1 deg pixels within 100 km of the landfall point, heat from their own 0.5 deg cell
lat01=lats; lon01=lons
cellset={}
for k,row in enumerate(R.itertuples()): cellset.setdefault(row.idx,{})[row.cell]=k
def pixpop(yv):
    if yv<=2000: w=(yv-1980)/20; return (1-w)*P[1980]+w*P[2000]
    if yv<=2020: w=(yv-2000)/20; return (1-w)*P[2000]+w*P[2020]
    return P[2020]
PPcache={}
U2={}; moved=0.0; total=0.0; dropped=0.0
for idx,cells in cellset.items():
    r=LF.loc[idx]; yv=int(r.season)
    if yv not in PPcache: PPcache[yv]=pixpop(yv)
    pp=PPcache[yv]
    i0=np.searchsorted(-lat01,-(r.lat+1.0)); i1=np.searchsorted(-lat01,-(r.lat-1.0)); dl=1.0/max(np.cos(np.radians(r.lat)),0.3)
    j0=np.searchsorted(lon01,r.lon-dl); j1=np.searchsorted(lon01,r.lon+dl)
    la=lat01[i0:i1][:,None]; lo=lon01[j0:j1][None,:]; sub=pp[i0:i1,j0:j1]
    d=2*6371*np.arcsin(np.sqrt(np.sin(np.radians(la-r.lat)/2)**2+np.cos(np.radians(r.lat))*np.cos(np.radians(la))*np.sin(np.radians(lo-r.lon)/2)**2))
    sel=(d<=100)&(sub>0)
    if not sel.any(): continue
    LA=np.broadcast_to(la,sel.shape)[sel]; LO=np.broadcast_to(lo,sel.shape)[sel]; PV=sub[sel]
    cc=[(round(np.round(a*2)/2,1),round(np.round(b*2)/2,1)) for a,b in zip(LA,LO)]
    keys=list(cells); karr=np.array(keys)
    agg={}
    for c,pv,a,b in zip(cc,PV,LA,LO):
        total+=pv
        if c not in cells:
            j=int(np.argmin((karr[:,0]-a)**2+(karr[:,1]-b)**2)); c2=keys[j]
            if np.hypot(c2[0]-a,c2[1]-b)>0.75: dropped+=pv; continue
            moved+=pv; c=c2
        agg[c]=agg.get(c,0)+pv
    lm=bool(r.lmic)
    if not lm: continue
    for c,w in agg.items():
        k=cells[c]
        for di in np.nonzero(M[k])[0]:
            key=(c,R.t0[k]+1+di); v=w*S[k]
            if v>U2.get(key,0): U2[key]=v
print('pixel population reassigned to neighbouring cell',round(moved/total,3),'dropped',round(dropped/total,3),flush=True)
f2=pd.DataFrame([(yr[d],v) for (c,d),v in U2.items()],columns=['year','pd'])
a2=f2.groupby('year').pd.sum().reindex(range(1981,2025),fill_value=0)/1e6
base=pd.read_csv('pilot/robust_annual_lmic.csv',index_col=0).iloc[:,0]
C=pd.DataFrame([dict(scheme='0.5° cells within 100 km (main)',early=base.loc[1981:2000].mean(),late=base.loc[2018:2024].mean()),
                dict(scheme='0.1° population pixels within 100 km',early=a2.loc[1981:2000].mean(),late=a2.loc[2018:2024].mean())])
C['ratio']=C.late/C.early; print(C.round(2).to_string()); C.to_csv('pilot/exposure_finepop.csv',index=False)
