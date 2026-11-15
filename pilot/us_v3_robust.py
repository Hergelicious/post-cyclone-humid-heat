# Review round 4: (1) repeated-peak exposure split into two estimands with consistent bounds;
# (2) restoration performance on the whole outage curve (customer-hours per peak customer);
# (3) robustness of the time to the first hazardous hour (thresholds, model spread, storms, timing from the county peak).
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl')
S=np.load('data/cmip6cty/slopes26.npz'); A=np.load('data/cmip6cty/humandelta15.npz')
NAT=list(A['keys']); MOD=list(S['models']); WLs=[1.5,2.0,2.8]
def cf(r,kind,k,wl=None):
    if kind=='nat': dT=-A['dT'][k]*r['fac']; dH=-A['dH'][k]*r['fac']
    else: dT=S['bT'][k]*(wl-r['gobs']); dH=S['bH'][k]*(wl-r['gobs'])
    return hi_c(r['T']+dT[r['mon']][:,r['ci']],np.clip(r['RH']+dH[r['mon']][:,r['ci']],1,100))
def peak_flag(X,r,thr):
    nh=X.shape[0]; offs=r['offs']; h0=r['h0']; F=np.zeros(X.shape,bool)
    for o in np.unique(offs):
        j=np.where(offs==o)[0]; sh=(h0+o)%24; n=int(np.ceil((nh+sh)/24))*24
        P=np.full((n,len(j)),-np.inf,'float32'); P[sh:sh+nh]=np.nan_to_num(X[:,j],nan=-np.inf)
        R3=P.reshape(n//24,24,len(j)); am=R3.argmax(1); mx=R3.max(1)
        G=np.zeros(R3.shape,bool); dd,cc=np.meshgrid(np.arange(n//24),np.arange(len(j)),indexing='ij'); G[dd,am,cc]=mx>=thr
        F[:,j]=G.reshape(n,len(j))[sh:sh+nh]
    return F
def bounds(r,HI):
    """per climate: sums over counties for k=1..5 of
       A_up: out at every one of the first k peaks, upper bound (running minimum of the peak counts)
       A_lo: same, assumption-free lower bound max(0, sum n_i - (k-1)N)
       B_up: continuously without power from the 1st to the k-th peak, assumption-free upper bound (minimum over all hours between)
       B_nc: same, estimate assuming no restorations and new disconnections in the same county-hour"""
    Ex=r['Ex'].astype('float64'); nh,nc=Ex.shape; win=r['win']; on=r['onset']
    pk=peak_flag(HI,r,35)&win[:,None]&(np.arange(nh)[:,None]>=on[None])
    cinc=np.cumsum(np.clip(np.diff(Ex,axis=0,prepend=Ex[:1]),0,None),0); out=np.zeros((4,5))
    for j in range(nc):
        hs=np.where(pk[:,j])[0]
        if not len(hs): continue
        v=Ex[hs,j]; N=r['cust'][j]
        for k in range(min(5,len(v))):
            out[0,k]+=v[:k+1].min(); out[1,k]+=max(0.0,v[:k+1].sum()-k*N)
            out[2,k]+=Ex[hs[0]:hs[k]+1,j].min(); out[3,k]+=max(0.0,v[k]-(cinc[hs[k],j]-cinc[hs[0],j]))
    return out
# ---------- (1)
Bo=sum(bounds(r,r['HI']) for r in recs)
Bn=np.array([sum(bounds(r,cf(r,'nat',k)) for r in recs) for k in range(len(NAT))])
sh=1-Bn/Bo[None]; mods=np.array([k.split('|')[0] for k in NAT])
rows=[]
for i,nm in enumerate(['A_upper','A_lower','B_upper','B_noChurn']):
    for k in range(5):
        mm=[sh[mods==m,i,k].mean() for m in np.unique(mods)]
        rows.append(dict(estimand=nm,k=k+1,million=Bo[i,k]/1e6,human=sh[:,i,k].mean(),p05=np.quantile(sh[:,i,k],0.05),p95=np.quantile(sh[:,i,k],0.95),mmin=min(mm),mmax=max(mm)))
B=pd.DataFrame(rows); B.to_csv('pilot/us_v4_bounds.csv',index=False)
print('(1) bounds (million) and human share'); print(B.pivot(index='k',columns='estimand',values='million').round(2).to_string())
print(B.pivot(index='k',columns='estimand',values='human').round(2).to_string())
# ---------- (2) whole-curve restoration metric
EV=pd.read_csv('pilot/us_v3_events.csv'); EV['sev']=pd.cut(EV.peak_frac,[0,0.2,0.5,0.8,1.01])
EV['R']=EV.cust_h/EV.peak.where(EV.peak>0)
E=EV[np.isfinite(EV.R)&(EV.R>0)&EV.sev.notna()].copy()
g=E.groupby(['storm','sev'],observed=True); E['relR']=E.R/g.R.transform('median'); E['n']=g.R.transform('size'); E=E[E.n>=5]
rows=[]
for wl,rq,rqp in [(1.5,0.13,0.05),(2.0,0.29,0.13),(2.8,0.49,0.31)]:
    for meas,r_ in [('hourly',rq),('daily peaks',rqp)]:
        t=1-r_; m=E.relR<=t
        rows.append(dict(wl=wl,measure=meas,required=r_,share_events=m.mean(),share_customers=np.average(m,weights=E.peak),share_exposure=np.average(m,weights=E.hour35+1e-9)))
RG=pd.DataFrame(rows); RG.to_csv('pilot/us_v4_restoration_R.csv',index=False); print('\n(2) whole-curve metric R = customer-hours / peak customers, relative to storm-severity class median, n=%d'%len(E)); print(RG.round(2).to_string())
print('relR quantiles 10/25/50:',E.relR.quantile([.1,.25,.5]).round(2).tolist())
hi=E[E.hour35>0]; print('counties with any simultaneous exposure: n=%d, median relR %.2f, share of exposure in relR>=1: %.0f%%'%(len(hi),hi.relR.median(),100*hi.hour35[hi.relR>=1].sum()/hi.hour35.sum()))
from scipy.stats import spearmanr
print('Spearman relR vs peak customers: %.2f; relR vs relative t50: %.2f'%(spearmanr(E.relR,E.peak)[0],spearmanr(E.relR,E.t50/E.groupby(['storm','sev'],observed=True).t50.transform('median'))[0]))
E['rel50']=E.t50/E.groupby(['storm','sev'],observed=True).t50.transform('median')
# trend in R
import statsmodels.formula.api as smf
E['lR']=np.log(E.R); mdl=smf.ols('lR ~ season + C(sev)',data=E).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(E.storm)[0]})
print('trend in log R per year: %.3f (%.3f to %.3f)'%(mdl.params['season'],*mdl.conf_int().loc['season']))
pd.to_pickle(dict(E=E[['storm','sev','R','relR','rel50','peak','hour35']],trend=(mdl.params['season'],*mdl.conf_int().loc['season'])),'pilot/us_v4_R.pkl')
# ---------- (3) time to the first hazardous hour: thresholds, model spread, storms, timing from the peak
def first_after(F,Ex,start,win):
    nh,nc=Ex.shape; el=np.arange(nh)[:,None]-start[None]; ok=F&(Ex>0)&(el>=0)&win[:,None]
    return np.where(ok.any(0),np.argmax(ok,0)-start,-1)
res=[]; per_storm=[]
for thr in (34.0,35.0,36.0):
    for ref in ('onset','peak'):
        W=[];L={'obs':[]}; Lw={(k,wl):[] for k in range(len(MOD)) for wl in WLs}; ST=[]
        for r in recs:
            Ex=r['Ex'].astype('float64'); st=r['onset'] if ref=='onset' else Ex[:24*5].argmax(0)
            w=Ex[st,np.arange(Ex.shape[1])]; W.append(w); ST+= [r['storm']]*len(w)
            L['obs'].append(first_after(r['HI']>=thr,Ex,st,r['win']))
            for k in range(len(MOD)):
                for wl in WLs: Lw[(k,wl)].append(first_after(cf(r,'wl',k,wl)>=thr,Ex,st,r['win']))
        W=np.concatenate(W); ST=np.array(ST)
        def s24(Lc): Lc=np.concatenate(Lc); return np.sum(W*((Lc>=0)&(Lc<=24)))/W.sum()
        obs=s24(L['obs']); d={'thr':thr,'ref':ref,'obs':obs}
        for wl in WLs:
            v=np.array([s24(Lw[(k,wl)]) for k in range(len(MOD))]); d[f'{wl}_med']=np.median(v); d[f'{wl}_p05']=np.quantile(v,0.05); d[f'{wl}_p95']=np.quantile(v,0.95)
        res.append(d)
        if thr==35.0 and ref=='onset':
            Lo=np.concatenate(L['obs']); L2=np.median(np.array([np.concatenate(Lw[(k,2.0)]) for k in range(len(MOD))]),0)
            L2=np.array([np.concatenate(Lw[(k,2.0)]) for k in range(len(MOD))])
            in2=((L2>=0)&(L2<=24)).mean(0)>=0.5; ino=(Lo>=0)&(Lo<=24)
            gain=pd.Series(W*(in2&~ino),index=ST).groupby(level=0).sum().sort_values(ascending=False)
            print('\n(3) customers newly within 24 h at 2.0 C by storm (share of the gain):'); print((gain/gain.sum()).head(8).round(3).to_string())
            # leave-one-storm-out for the top storms
            for s in gain.index[:4]:
                m=ST!=s; print('  without %s: observed %.1f%%, 2.0 C (models majority) %.1f%%'%(s,100*np.sum(W[m]*ino[m])/W[m].sum(),100*np.sum(W[m]*in2[m])/W[m].sum()))
            per_storm=gain
LR=pd.DataFrame(res); LR.to_csv('pilot/us_v4_lead_robust.csv',index=False); print(LR.round(3).to_string())
pd.to_pickle(per_storm,'pilot/us_v4_lead_storms.pkl')
