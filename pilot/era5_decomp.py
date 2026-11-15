import pandas as pd, numpy as np, os, itertools, sys
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
C=pd.read_csv('pilot/era5_cells1.csv'); NC=len(C)
def es(T): return 6.112*np.exp(17.67*T/(T+243.5))
def thermo(T,Tw,P):
    Tw=np.minimum(Tw,T); e=es(Tw)-6.53e-4*(1+0.000944*Tw)*P*(T-Tw); e=np.clip(e,0.1,None); e=np.minimum(e,es(T))
    q=0.622*e/(P-0.378*e); return e,q
def rh_from(T,q,P): e=q*P/(0.622+0.378*q); return np.clip(100*e/es(T),1,100)
months=[(y,m) for y in range(1981,2022) for m in range(1,13) if os.path.exists(f'data/era5h/Tw_{y}{m:02d}.npy') and os.path.exists(f'data/era5h/Tp_{y}{m:02d}.npz')]
print('months available',len(months),flush=True)
# pass 1: daily stats
os.makedirs('data/era5h/daily',exist_ok=True)
MM={}
for y,m in months:
    out=f'data/era5h/daily/{y}{m:02d}.npz'
    if not os.path.exists(out):
        a=np.load(f'data/era5h/Tp_{y}{m:02d}.npz'); T=a['T']; P=a['P']; Tw=np.load(f'data/era5h/Tw_{y}{m:02d}.npy')
        n=min(len(T),len(Tw)); T,P,Tw=T[:n],P[:n],Tw[:n]
        e,q=thermo(T,Tw,P); RH=np.clip(100*e/es(T),1,100); HI=hi_c(T,RH)
        nd=n//24; r=lambda x:x[:nd*24].reshape(nd,24,NC)
        HId=r(HI); im=HId.argmax(1)
        np.savez(out,Tmax=r(T).max(1),HImax=HId.max(1),qmean=r(q).mean(1),RHhm=np.take_along_axis(r(RH),im[:,None,:],1)[:,0])
    d=np.load(out); MM[(y,m)]=(np.nanmean(d['Tmax'],0),np.nanmean(d['qmean'],0),np.nanmean(d['RHhm'],0))
# slopes per calendar month (needs >=25 years)
bT=np.full((12,NC),np.nan); bQ=np.full((12,NC),np.nan); bR=np.full((12,NC),np.nan)
for k in range(1,13):
    ys=[y for (y,m) in MM if m==k]
    if len(ys)<25: continue
    x=gpi.loc[ys].values; xc=x-x.mean()
    Tm=np.stack([MM[(y,k)][0] for y in ys]); Qm=np.stack([MM[(y,k)][1] for y in ys]); Rm=np.stack([MM[(y,k)][2] for y in ys])
    bT[k-1]=xc@(Tm-Tm.mean(0))/(xc@xc); bQ[k-1]=(xc@(Qm-Qm.mean(0))/(xc@xc))/Qm.mean(0); bR[k-1]=xc@(Rm-Rm.mean(0))/(xc@xc)
np.savez('data/era5h/slopes.npz',bT=bT,bQ=bQ,bR=bR)
print('months with slopes',[k+1 for k in range(12) if np.isfinite(bT[k]).any()],flush=True)
dG=gpi.loc[2018:2021].mean()-gpi.loc[1981:2000].mean(); print('dG',round(dG,3))
# pass 2: scenario daily HImax for months with slopes (only years in the two periods)
os.makedirs('data/era5h/scen',exist_ok=True)
for y,m in months:
    if not (y<=2000 or y>=2018) or not np.isfinite(bT[m-1]).any(): continue
    out=f'data/era5h/scen/{y}{m:02d}.npz'
    if os.path.exists(out): continue
    a=np.load(f'data/era5h/Tp_{y}{m:02d}.npz'); T=a['T']; P=a['P']; Tw=np.load(f'data/era5h/Tw_{y}{m:02d}.npy'); n=min(len(T),len(Tw)); T,P,Tw=T[:n],P[:n],Tw[:n]
    e,q=thermo(T,Tw,P); RH=np.clip(100*e/es(T),1,100); nd=n//24
    sgn=1.0 if y<=2000 else -1.0; dT=sgn*bT[m-1]*dG; dq=1+sgn*bQ[m-1]*dG
    res={}
    for name,(TT,qq) in {'T':(T+dT,q),'q':(T,q*dq),'Tq':(T+dT,q*dq)}.items():
        H=hi_c(TT,rh_from(TT,qq,P)); res[name]=H[:nd*24].reshape(nd,24,NC).max(1).astype('float32')
    res['T_RHfix']=hi_c(T+dT,RH)[:nd*24].reshape(nd,24,NC).max(1).astype('float32')
    np.savez(out,**res)
print('scenarios done',flush=True)
