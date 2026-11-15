# Restoration transfer test: for every county outage, the restoration phase (from the county's peak onward) is replaced by the
# mean normalised restoration curve of peer counties in the same storm and severity class, scaled to the county's own peak and
# combined with the county's own hourly weather. Peers: fastest quarter, faster half, or all peers (reference), ranked by
# outage customer-hours per customer out at peak. Exposure = customer-hours in hours with HI >= 35 C.
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl')
S=np.load('data/cmip6cty/slopes26.npz'); MOD=list(S['models']); WLs=[1.5,2.0,2.8]
def cf(r,k,wl):
    dT=S['bT'][k]*(wl-r['gobs']); dH=S['bH'][k]*(wl-r['gobs'])
    return hi_c(r['T']+dT[r['mon']][:,r['ci']],np.clip(r['RH']+dH[r['mon']][:,r['ci']],1,100))
VAR=['observed','fastQ','fastHalf','allPeers']
tot={v:0.0 for v in VAR}; wl_tot={v:np.zeros((len(MOD),3)) for v in VAR}; cov=0.0; per=[]
for r in recs:
    Ex=r['Ex'].astype('float64'); nh,nc=Ex.shape; win=r['win'][:,None]; pk=Ex.max(0); ip=Ex.argmax(0)
    R=(Ex*win).sum(0)/np.maximum(pk,1); sev=np.digitize(pk/r['cust'],[0.2,0.5,0.8])
    C=np.zeros((nc,nh))                                   # normalised curve from the peak onward
    for j in range(nc): C[j,:nh-ip[j]]=Ex[ip[j]:,j]/max(pk[j],1)
    X={v:Ex.copy() for v in VAR}; done=np.zeros(nc,bool)
    for s in range(4):
        idx=np.where(sev==s)[0]
        if len(idx)<5: continue
        for j in idx:
            peers=idx[idx!=j]; rp=R[peers]
            sets={'fastQ':peers[rp<=np.quantile(rp,0.25)],'fastHalf':peers[rp<=np.median(rp)],'allPeers':peers}
            for v,pp in sets.items():
                c=C[pp].mean(0); X[v][ip[j]:,j]=pk[j]*c[:nh-ip[j]]
            done[j]=True
    F=r['HI']>=35
    e={v:(X[v]*win*F).sum(0) for v in VAR}
    for v in VAR: tot[v]+=e[v].sum()
    cov+=e['observed'][done].sum()
    per.append(dict(storm=r['storm'],**{v:e[v].sum() for v in VAR}))
    for k in range(len(MOD)):
        for a,wl in enumerate(WLs):
            Fw=cf(r,k,wl)>=35
            for v in VAR: wl_tot[v][k,a]+=(X[v]*win*Fw).sum()
o=tot['observed']; rows=[]
for v in VAR:
    m=np.median(wl_tot[v],0)/o; lo=np.quantile(wl_tot[v],0.05,0)/o; hi=np.quantile(wl_tot[v],0.95,0)/o
    rows.append(dict(curve=v,obs_million=tot[v]/1e6,rel_obs=tot[v]/o,**{f'wl{w}':m[a] for a,w in enumerate(WLs)},**{f'wl{w}_lo':lo[a] for a,w in enumerate(WLs)},**{f'wl{w}_hi':hi[a] for a,w in enumerate(WLs)}))
T=pd.DataFrame(rows); T.to_csv('pilot/us_v5_transfer.csv',index=False); print(T.round(3).to_string())
print('share of observed exposure in counties with a peer class of >=5: %.1f%%'%(100*cov/o))
P=pd.DataFrame(per); P.to_csv('pilot/us_v5_transfer_storms.csv',index=False)
print(P.sort_values('observed',ascending=False).head(6).assign(fq=lambda d:d.fastQ/d.observed,fh=lambda d:d.fastHalf/d.observed,ap=lambda d:d.allPeers/d.observed)[['storm','observed','fq','fh','ap']].round(2).to_string())
