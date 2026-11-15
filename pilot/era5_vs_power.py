# Like-for-like comparison of ERA5 (0.25 deg, hourly) and NASA POWER (MERRA-2 based, hourly) post-landfall heat for LMIC landfalls,
# 2001-2024: same 0.5 deg cells, landfall dates, population, local days 1-30 and heat index formula.
# Part A: daily maximum heat index at every matched cell-day (NASA POWER hourly daily maxima, local solar time).
# Part B: hour-by-hour temperature and dewpoint for the most populous cell-windows, with temperature and humidity swapped between
#         the datasets to separate their contributions (two-factor Shapley).
import pandas as pd, numpy as np, os, glob, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
D0=pd.Timestamp('1981-01-01'); NB=44
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[(R.foot=='100')&R.lmic&(R.season>=2001)].reset_index(drop=True)
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); pid={k:p for k,p in zip(pts.key,pts.pid)}
R['pid']=[pid[f'{a}_{b}'] for a,b in R.cell]; R['lon']=[b for a,b in R.cell]
grp={'CN':'China','IN':'India','VN':'Viet Nam','BD':'Bangladesh','PH':'Philippines'}; R['grp']=R.iso2.map(grp).fillna('Other LMICs')
n=len(R); T=np.full((n,NB*24),np.nan,'float32'); Td=T.copy(); act={}
for i,t0 in enumerate(R.t0.values):
    for k in range(NB): act.setdefault(int(t0)-11+k,[]).append((i,k))
for d in sorted(act):
    f=f'data/era5q/{(D0+pd.Timedelta(days=d)):%Y%m%d}.npz'
    if not os.path.exists(f): continue
    z=np.load(f); ix=dict(zip(z['pid'],range(len(z['pid'])))); TT=z['T']; DD=z['Td']
    for i,k in act[d]:
        j=ix.get(R.pid[i])
        if j is not None: T[i,k*24:(k+1)*24]=TT[:,j]/100; Td[i,k*24:(k+1)*24]=DD[:,j]/100
off=np.array([int(round(lo/15)) for lo in R.lon])
def local_max(X,rows=None):
    rows=np.arange(n) if rows is None else rows; o_=off[rows]; out=np.full((len(rows),30),np.nan,'float32')
    for o in np.unique(o_):
        r=np.where(o_==o)[0]; st=12*24-o; out[r]=np.nanmax(X[r][:,st:st+720].reshape(len(r),30,24),2)
    return out
HIe=local_max(hi_c(T,np.clip(100*es(Td)/es(T),1,100)))
# ---- Part A: NASA POWER hourly daily maxima (local solar time) at the same cell-days
HIn=np.full((n,30),np.nan,'float32')
for i,r in enumerate(R.itertuples()):
    la,lo=r.cell; a=(D0+pd.Timedelta(days=int(r.t0)+1)).strftime('%Y%m%d'); fn=f'pilot/hourly_cells/{la}_{lo}_{a}.csv'
    if os.path.exists(fn):
        h=pd.read_csv(fn); v=h.HI.values[:30]; HIn[i,:len(v)]=v
ok=np.isfinite(HIe).all(1)&np.isfinite(HIn).all(1); print('matched cell-windows',int(ok.sum()),'of',n)
E=HIe[ok]; P=HIn[ok]; pop=R['pop'].values[ok][:,None]; G=R.grp.values[ok]
d=(E-P).ravel()
rows=[]
def summ(name,m):
    e=E[m]; p=P[m]; w=np.broadcast_to(pop[m],e.shape)
    rows.append(dict(group=name,cell_windows=int(m.sum()),mean_HI_ERA5=np.mean(e),mean_HI_POWER=np.mean(p),bias=np.mean(e-p),
        bias_pw=np.average(e-p,weights=w),corr=np.corrcoef(e.ravel(),p.ravel())[0,1],
        days35_ERA5=(e>=35).mean()*30,days35_POWER=(p>=35).mean()*30,days406_ERA5=(e>=40.6).mean()*30,days406_POWER=(p>=40.6).mean()*30,
        pd406_ratio=((e>=40.6)*w).sum()/max(((p>=40.6)*w).sum(),1),pd35_ratio=((e>=35)*w).sum()/max(((p>=35)*w).sum(),1),
        both406=((e>=40.6)&(p>=40.6)).sum()/max(((e>=40.6)|(p>=40.6)).sum(),1)))
summ('All LMICs',np.ones(len(E),bool))
for g in ['China','India','Viet Nam','Bangladesh','Philippines','Other LMICs']: summ(g,G==g)
A=pd.DataFrame(rows); A.to_csv('pilot/era5_vs_power_daily.csv',index=False); print(A.round(2).to_string())
print('daily maximum HI difference ERA5 minus NASA POWER, percentiles 5,25,50,75,95:',np.percentile(d,[5,25,50,75,95]).round(2))
# shift needed: constant offset that equalises person-days >= 40.6
for s in np.arange(0,3.01,0.25):
    r_=(((E-s)>=40.6)*pop).sum()/(((P)>=40.6)*pop).sum()
    print(f'ERA5 minus {s:.2f} C: person-day ratio {r_:.2f}')
np.savez('pilot/era5_vs_power_daily.npz',E=E,P=P,pop=pop[:,0],G=G)
# ---- Part B: hourly temperature and dewpoint swaps for the most populous cell-windows
fs=glob.glob('data/power_td/*.csv'); keyi={f'{a}_{b}_{int(t)}':i for i,((a,b),t) in enumerate(zip(R.cell,R.t0))}
Tn=np.full((n,NB*24),np.nan,'float32'); Tdn=Tn.copy(); sel=[]
for fn in fs:
    k=os.path.basename(fn)[:-4]; i=keyi.get(k)
    if i is None: continue
    h=pd.read_csv(fn).replace(-999,np.nan)
    if len(h)<NB*24: continue
    Tn[i]=h.T2M.values[:NB*24]; Tdn[i]=h.T2MDEW.values[:NB*24]; sel.append(i)
sel=np.array(sorted(sel)); print('hourly cell-windows',len(sel))
def HIx(Ta,Tda): return local_max(hi_c(Ta,np.clip(100*es(Tda)/es(Ta),1,100)),sel)
Ts,Tds,Tns,Tdns=T[sel],Td[sel],Tn[sel],Tdn[sel]
def HIs(Ta,Tda):
    o_=off[sel]; out=np.full((len(sel),30),np.nan,'float32'); X=hi_c(Ta,np.clip(100*es(Tda)/es(Ta),1,100))
    for o in np.unique(o_):
        r=np.where(o_==o)[0]; st=12*24-o; out[r]=np.nanmax(X[r][:,st:st+720].reshape(len(r),30,24),2)
    return out
EE=HIs(Ts,Tds); NN=HIs(Tns,Tdns); ETN=HIs(Ts,Tdns); NTE=HIs(Tns,Tds)
w=R['pop'].values[sel][:,None]; gs=R.grp.values[sel]
def pdx(X,m): return ((X[m]>=40.6)*w[m]).sum()
def tmaxday(X):
    o_=off[sel]; out=np.full((len(sel),30),np.nan,'float32')
    for o in np.unique(o_):
        r=np.where(o_==o)[0]; st=12*24-o; out[r]=np.nanmax(X[r][:,st:st+720].reshape(len(r),30,24),2)
    return out
out=[]
for name,m in [('All',np.ones(len(sel),bool))]+[(g,gs==g) for g in ['China','India','Viet Nam','Bangladesh','Philippines','Other LMICs']]:
    if m.sum()==0: continue
    a,b,c,e=pdx(EE,m),pdx(NN,m),pdx(ETN,m),pdx(NTE,m); tot=a-b
    # Shapley: temperature effect = average of (E_T,N_Td - N,N) and (E,E - N_T,E_Td); humidity effect = remainder
    tef=((c-b)+(a-e))/2; hef=((e-b)+(a-c))/2
    out.append(dict(group=name,cell_windows=int(m.sum()),pd_ERA5=a/1e6,pd_POWER=b/1e6,ratio=a/max(b,1),
        share_T=tef/tot if tot else np.nan,share_humidity=hef/tot if tot else np.nan,
        Tmax_bias=np.nanmean(tmaxday(Ts)[m]-tmaxday(Tns)[m]),Td_bias=np.nanmean(Tds[m]-Tdns[m]),T_bias=np.nanmean(Ts[m]-Tns[m])))
B=pd.DataFrame(out); B.to_csv('pilot/era5_vs_power_hourly.csv',index=False); print(B.round(2).to_string())
