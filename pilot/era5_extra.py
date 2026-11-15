# Remaining ERA5-led analyses for the rewrite: daily profile, maps, leave-one-country-out, days per resident, people exposed and
# recurrence, person-nights, regression robustness, Shapley decomposition with ERA5 local trends, 3.3 C projection.
import pandas as pd, numpy as np, os, sys, json, warnings
warnings.filterwarnings('ignore')
src=open('pilot/era5_rebuild.py').read(); exec(src[:src.index("HI=hi_c(T,RH)")])     # loads R, T, Td, RH, n, off, local_days, es
# surface pressure (12 UTC each day) for specific humidity
Pd=np.full((n,NB),np.nan,'float32')
for d in sorted(act):
    f=f'data/era5q/{(D0+pd.Timedelta(days=d)):%Y%m%d}.npz'; z=np.load(f); ix=dict(zip(z['pid'],range(len(z['pid'])))); PP=z['P']
    for i,k in act[d]:
        j=ix.get(R.pid[i]);
        if j is not None: Pd[i,k]=PP[j]/10
Ph=np.repeat(Pd,24,1)
HI=hi_c(T,RH)
def local_range(X,first,ndays,how='max'):
    """per local day from t0+first to t0+first+ndays-1"""
    out=np.full((n,ndays),np.nan,'float32')
    for o in np.unique(off):
        r=np.where(off==o)[0]; st=(11+first)*24-o
        blk=X[r][:,st:st+ndays*24].reshape(len(r),ndays,24); out[r]=np.nanmax(blk,2) if how=='max' else np.nanmin(blk,2)
    return out
HIall=local_range(HI,-10,41); HImax=HIall[:,11:]           # local days -10..30; days 1..30
dates=np.array([[int(t0)+1+k for k in range(30)] for t0 in R.t0.values])
G=pd.read_csv('pilot/gmst.csv'); G=G[G.Source=='GISTEMP'].set_index('Year').Mean; gpi=G-G.loc[1880:1900].mean()
per={'1981-2000':(1981,2000),'2001-2017':(2001,2017),'2018-2024':(2018,2024)}
out={}
# ---- 1. daily profile at the landfall cell
R['dist']=[np.hypot(a-L.lat[i],(b-L.lon[i])*np.cos(np.radians(a))) for i,(a,b) in zip(R.idx,R.cell)]
lc=R.sort_values('dist').drop_duplicates('idx').index.values
prof=[]
for p,(a,b) in per.items():
    m=lc[(R.season.values[lc]>=a)&(R.season.values[lc]<=b)]
    for k in range(41): prof.append(dict(period=p,day=k-10,h35=np.nanmean(HIall[m,k]>=35),h406=np.nanmean(HIall[m,k]>=40.6)))
prof=pd.DataFrame(prof); prof.to_csv('pilot/era5_daily_profile.csv',index=False)
print(prof.pivot(index='day',columns='period',values='h35').loc[[-10,-5,-1,0,1,3,7,14,30]].round(3).to_string(),flush=True)
# ---- 2. unique cell-days and maps
def uniq(flag,pop=None,mask=None):
    m=np.ones(n,bool) if mask is None else mask
    pp=(R['pop'].values if pop is None else pop)[m][:,None]
    k=np.repeat(R.pid.values[m],30)*100000+dates[m].ravel()
    df=pd.DataFrame(dict(k=k,v=(flag[m]*pp).ravel(),yr=np.repeat(R.season.values[m],30),lm=np.repeat(R.lmic.values[m],30),iso=np.repeat(R.iso2.values[m],30),pid=np.repeat(R.pid.values[m],30)))
    return df.sort_values('v',ascending=False).drop_duplicates('k')
U=uniq((HImax>=40.6).astype('float32'))
latc=dict(zip(R.pid,R.lat)); lonc=dict(zip(R.pid,R.lon)); U['g2']=[f'{int(np.floor(latc[p]/2)*2)}_{int(np.floor(lonc[p]/2)*2)}' for p in U.pid]
mp=[]
for p,(a,b) in [('1981-2000',(1981,2000)),('2018-2024',(2018,2024))]:
    x=U[(U.yr>=a)&(U.yr<=b)].groupby('g2').v.sum()/(b-a+1)/1e6
    for g_,v in x.items(): mp.append(dict(period=p,lat=int(g_.split('_')[0]),lon=int(g_.split('_')[1]),pd406=v))
pd.DataFrame(mp).to_csv('pilot/era5_map2deg.csv',index=False)
ann=U[U.lm].groupby(['yr','iso']).v.sum().unstack(fill_value=0)/1e6; ann=ann.reindex(range(1981,2025),fill_value=0); ann.to_csv('pilot/era5_annual_country.csv')
tot=ann.sum(1); rng=np.random.default_rng(11)
def ratio(series):
    r=[series.loc[rng.choice(range(2018,2025),7)].mean()/max(series.loc[rng.choice(range(1981,2001),20)].mean(),1e-9) for _ in range(2000)]
    return series.loc[2018:2024].mean()/series.loc[1981:2000].mean(),np.percentile(r,2.5),np.percentile(r,97.5)
res=[('main',)+ratio(tot)]
blk=[tot.loc[a:a+10].mean() for a in (1981,1992,2003,2014)]
r=[tot.loc[rng.choice(range(2014,2025),11)].mean()/tot.loc[rng.choice(range(1981,1992),11)].mean() for _ in range(2000)]
res.append(('blocks 2014-2024/1981-1991',blk[3]/blk[0],np.percentile(r,2.5),np.percentile(r,97.5)))
for c in ['CN','IN','VN','PH','BD']: res.append((f'without {c}',)+ratio(tot-ann.get(c,0)))
res.append(('without all five',)+ratio(tot-ann[[c for c in ['CN','IN','VN','PH','BD'] if c in ann]].sum(1)))
d_=[tot.loc[rng.choice(range(2018,2025),7)].mean()-tot.loc[rng.choice(range(1981,2001),20)].mean() for _ in range(2000)]
RR=pd.DataFrame(res,columns=['contrast','ratio','lo','hi']); RR.to_csv('pilot/era5_ratios.csv',index=False); print(RR.round(2).to_string(),flush=True)
print('blocks',np.round(blk,0),'difference %.0f (%.0f-%.0f)'%(tot.loc[2018:2024].mean()-tot.loc[1981:2000].mean(),*np.percentile(d_,[2.5,97.5])),'running mean 1990,2000,2010,2020,2024',tot.rolling(10).mean().loc[[1990,2000,2010,2020,2024]].round(0).tolist())
print('LMIC share 2018-2024: %.0f%%'%(100*U[(U.yr>=2018)].query('lm').v.sum()/U[(U.yr>=2018)].v.sum()))
cty=[]
for c in ['CN','IN','VN','PH','BD']:
    for p,(a,b) in per.items(): cty.append(dict(iso=c,period=p,pd406=ann.loc[a:b,c].mean() if c in ann else 0))
other=tot-ann[[c for c in ['CN','IN','VN','PH','BD'] if c in ann]].sum(1)
for p,(a,b) in per.items(): cty.append(dict(iso='other',period=p,pd406=other.loc[a:b].mean()))
# days per resident per landfall
R['e406']=(HImax>=40.6).sum(1); R['pdr']=R.e406*R['pop']
for c in ['CN','IN','VN','PH','BD','other']:
    for p,(a,b) in per.items():
        m=(R.season>=a)&(R.season<=b)&R.lmic&((R.iso2==c) if c!='other' else ~R.iso2.isin(['CN','IN','VN','PH','BD']))
        [x for x in cty if x['iso']==c and x['period']==p][0]['days_per_resident']=R[m].pdr.sum()/R[m]['pop'].sum() if R[m]['pop'].sum()>0 else np.nan
m=R.lmic; print('days per resident, all LMICs:',{p:round(R[m&(R.season>=a)&(R.season<=b)].pdr.sum()/R[m&(R.season>=a)&(R.season<=b)]['pop'].sum(),2) for p,(a,b) in per.items()})
CT=pd.DataFrame(cty); CT.to_csv('pilot/era5_country.csv',index=False); print(CT.round(2).to_string(),flush=True)
# ---- 3. people exposed, days per exposed person, recurrence (population at decade midpoint from GHS-POP)
rc=open('pilot/reconcile.py').read(); _ns={'np':np,'pd':pd}; exec(rc[rc.index('# population 0.5 deg'):rc.index('def series')],_ns); pop_cell=_ns['pop_cell']
L_=U[U.lm].copy(); L_['cell']=L_.pid
pe=[]
for p,(a,b) in per.items():
    x=L_[(L_.yr>=a)&(L_.yr<=b)]
    # people exposed per year: population of cells with >=1 dangerous day in that year
    cy=x[x.v>0].groupby(['yr','cell']).v.max().reset_index()
    pop_y=[pop_cell(latc[c],lonc[c],y) for c,y in zip(cy.cell,cy.yr)]; cy['popc']=pop_y
    ppl=cy.groupby('yr').popc.sum().reindex(range(a,b+1),fill_value=0).mean()/1e6; pdays=x.v.sum()/(b-a+1)/1e6
    pe.append(dict(period=p,person_days=pdays,people=ppl,days_per_person=pdays/ppl))
PE=pd.DataFrame(pe); print(PE.round(2).to_string())
rec=[]
for a,b in [(1981,1990),(1991,2000),(2001,2010),(2011,2020),(2015,2024)]:
    x=L_[(L_.yr>=a)&(L_.yr<=b)&(L_.v>0)].groupby('cell').yr.nunique(); mid=(a+b)//2+1
    pp={c:pop_cell(latc[c],lonc[c],mid) for c in x.index}
    rec.append(dict(decade=f'{a}-{b}',any=sum(pp.values())/1e6,two=sum(pp[c] for c in x.index if x[c]>=2)/1e6,three=sum(pp[c] for c in x.index if x[c]>=3)/1e6))
REC=pd.DataFrame(rec); print(REC.round(0).to_string()); PE.to_csv('pilot/era5_people.csv',index=False); REC.to_csv('pilot/era5_recurrence.csv',index=False)
# ---- 4. nights
Tn=np.full((n,30),np.nan,'float32')
for o in np.unique(off):
    r=np.where(off==o)[0]; st=(11+1)*24-o+18; Tn[r]=np.nanmin(T[r][:,st:st+30*24].reshape(len(r),30,24)[:,:,:12],2)
for thr in (26,28):
    Un=uniq((Tn>=thr).astype('float32'))
    print(f'LMIC person-nights min T>={thr}:',{p:round(Un[(Un.yr>=a)&(Un.yr<=b)&Un.lm].v.sum()/(b-a+1)/1e6) for p,(a,b) in per.items()})
Ud=uniq(((HImax>=40.6)&(Tn>=26)).astype('float32'))
print('share of dangerous person-days followed by night >=26:',{p:round(100*Ud[(Ud.yr>=a)&(Ud.yr<=b)&Ud.lm].v.sum()/U[(U.yr>=a)&(U.yr<=b)&U.lm].v.sum()) for p,(a,b) in per.items()},flush=True)
# ---- 5. regression robustness at the landfall cell
import statsmodels.formula.api as smf
LC=R.loc[lc].set_index('idx'); LC['e35']=(HImax[lc]>=35).sum(1); LC['G']=gpi.loc[LC.season].values
LC['reg']=[f'{int(np.floor(L.lat[i]/2))}_{int(np.floor(L.lon[i]/2))}' for i in LC.index]; LC['iso']=L.iso2.reindex(LC.index).values; LC['mon']=[L.time[i].month for i in LC.index]
def fit(d,f='e35~G+C(reg)',two=False):
    if two:
        import statsmodels.api as sm
        m=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':np.c_[pd.factorize(d.iso)[0],pd.factorize(d.season)[0]]})
    else: m=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.iso)[0]})
    return m.params.G,*m.conf_int().loc['G']
rows=[('main HI>=35',*fit(LC)),('main HI>=40.6',*fit(LC,'e406~G+C(reg)')),('two-way clustering',*fit(LC,two=True)),('calendar-month FE',*fit(LC,'e35~G+C(reg)+C(mon)'))]
bs=[]; yrs=LC.season.unique()
for b in range(300):
    ys=rng.choice(yrs,len(yrs)); d=pd.concat([LC[LC.season==y] for y in ys]); bs.append(smf.ols('e35~G+C(reg)',d).fit().params.G)
rows.append(('year bootstrap',np.mean(bs),np.percentile(bs,2.5),np.percentile(bs,97.5)))
for c in ['CN','IN','VN','PH','BD']: rows.append((f'without {c}',*fit(LC[LC.iso!=c])))
rows.append(('without all five',*fit(LC[~LC.iso.isin(['CN','IN','VN','PH','BD'])])))
rows.append(('LMICs only',*fit(LC[LC.lmic]))); rows.append(('high income only',*fit(LC[~LC.lmic])))
RG=pd.DataFrame(rows,columns=['test','b','lo','hi']); RG.to_csv('pilot/era5_regression.csv',index=False); print(RG.round(2).to_string(),'n',len(LC),flush=True)
pdict={p:(LC[(LC.season>=a)&(LC.season<=b)].e35.mean(),100*(LC[(LC.season>=a)&(LC.season<=b)].e406>0).mean()) for p,(a,b) in per.items()}; print('per landfall',pdict)
# placebo: same calendar windows in other years is not available in the window extraction (noted)
# ---- 6. Shapley decomposition with ERA5 local trends (1 deg monthly, 1981-2021), LMIC, 100 km, HI>=40.6, summed over landfalls
SL=np.load('data/era5h/slopes.npz'); C1=pd.read_csv('pilot/era5_cells1.csv'); k1={(int(a),int(b)):i for i,(a,b) in enumerate(zip(C1.lat,C1.lon))}
E_=R.lmic.values&(R.season.values<=2000); L2=R.lmic.values&(R.season.values>=2018)
dG=gpi.loc[2018:2024].mean()-gpi.loc[1981:2000].mean()
c1=np.array([k1.get((int(np.round(a)),int(np.round(b))),-1) for a,b in zip(R.lat,R.lon)])
print('decomposition: dG %.2f, records without 1-deg slope %d'%(dG,int(((E_|L2)&(c1<0)).sum())),flush=True)
mo_h=np.array([[ (D0+pd.Timedelta(days=int(t0)-11+k)).month-1 for k in range(NB)] for t0 in R.t0.values]); mo_h=np.repeat(mo_h,24,1)
def shifted_days(sel,sgn,var):
    i=sel[c1[sel]>=0]; ci_=c1[i][:,None]
    bT=np.nan_to_num(SL['bT'][mo_h[i],ci_]); bQ=np.nan_to_num(SL['bQ'][mo_h[i],ci_])
    t=T[i]+(sgn*bT*dG if 'T' in var else 0); e=es(Td[i]); q=0.622*e/(Ph[i]-0.378*e)
    if 'q' in var: q=q*(1+sgn*bQ*dG)
    if var=='T_RHfix': rh=RH[i]
    else: e2=q*Ph[i]/(0.622+0.378*q); rh=np.clip(100*e2/es(t),1,100)
    h=hi_c(t,rh); full=np.full((n,NB*24),np.nan,'float32'); full[i]=h; m=local_days(full)[i]
    out_=np.zeros(n); out_[i]=(m>=40.6).sum(1); return out_
popE=np.array([pop_cell(a,b,1990) for a,b in zip(R.lat,R.lon)]); popL=np.array([pop_cell(a,b,2021) for a,b in zip(R.lat,R.lon)])
dE=np.where(c1>=0,(HImax>=40.6).sum(1),np.nan); idxE=np.where(E_&(c1>=0))[0]; idxL=np.where(L2&(c1>=0))[0]
def shap(var,bootE=None,bootL=None,byyear=False):
    iE=idxE if bootE is None else bootE; iL=idxL if bootL is None else bootL
    eE=dE[iE]; eL=dE[iL]; sE=SH[var][0][iE]; sL=SH[var][1][iL]
    # E(S,P,H): S storm set, P population epoch, H climate
    def Ev(S,P,H):
        if S=='E': d=eE if H=='E' else sE; pp=(popE if P=='E' else popL)[iE]; return (d*pp).sum()/20/1e6
        d=eL if H=='L' else sL; pp=(popE if P=='E' else popL)[iL]; return (d*pp).sum()/7/1e6
    import itertools
    phi={f:0.0 for f in 'SPH'}
    for order in itertools.permutations('SPH'):
        st={'S':'E','P':'E','H':'E'}
        for f in order:
            a=Ev(st['S'],st['P'],st['H']); st[f]='L'; b=Ev(st['S'],st['P'],st['H']); phi[f]+=(b-a)/6
    tot_=Ev('L','L','L')-Ev('E','E','E'); return phi,tot_,Ev('E','E','E'),Ev('L','L','L')
SH={}
for var in ['Tq','T','q','T_RHfix']:
    SH[var]=(shifted_days(idxE,+1,var),shifted_days(idxL,-1,var))
dec=[]
for var in ['Tq','T','q','T_RHfix']:
    phi,tt,e0,e1=shap(var); bl=[]; by=[]
    for b in range(300):
        bE=rng.choice(idxE,len(idxE)); bL=rng.choice(idxL,len(idxL)); p_,t_,_,_=shap(var,bE,bL); bl.append([p_['H']/t_,p_['P']/t_,p_['S']/t_]+[p_['H'],p_['P'],p_['S']])
    yE=R.season.values[idxE]; yL=R.season.values[idxL]
    for b in range(300):
        ysE=rng.choice(range(1981,2001),20); ysL=rng.choice(range(2018,2025),7)
        bE=np.concatenate([idxE[yE==y] for y in ysE]); bL=np.concatenate([idxL[yL==y] for y in ysL]); p_,t_,_,_=shap(var,bE,bL); by.append([p_['H']/t_,p_['P']/t_,p_['S']/t_])
    bl=np.array(bl); by=np.array(by)
    dec.append(dict(var=var,early=e0,late=e1,total=tt,climate=phi['H'],population=phi['P'],storms=phi['S'],sh_clim=phi['H']/tt,sh_pop=phi['P']/tt,sh_storm=phi['S']/tt,
        clim_lo=np.percentile(bl[:,3],2.5),clim_hi=np.percentile(bl[:,3],97.5),shc_lo=np.percentile(bl[:,0],2.5),shc_hi=np.percentile(bl[:,0],97.5),shs_lo=np.percentile(bl[:,2],2.5),shs_hi=np.percentile(bl[:,2],97.5),
        shp_lo=np.percentile(bl[:,1],2.5),shp_hi=np.percentile(bl[:,1],97.5),shc_ylo=np.percentile(by[:,0],2.5),shc_yhi=np.percentile(by[:,0],97.5)))
DEC=pd.DataFrame(dec); DEC.to_csv('pilot/era5_decomposition.csv',index=False); print(DEC.round(2).to_string(),flush=True)
# leave-one-country-out climate share (T and q)
for c in ['CN','IN','VN','BD']:
    kE=idxE[R.iso2.values[idxE]!=c]; kL=idxL[R.iso2.values[idxL]!=c]; p_,t_,_,_=shap('Tq',kE,kL); print('decomposition without',c,'climate share %.0f%%'%(100*p_['H']/t_))
print('ALL DONE',flush=True)
