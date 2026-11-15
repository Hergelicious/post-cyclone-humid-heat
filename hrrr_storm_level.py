# Storm-level evaluation of the forecast test: storm-bootstrap intervals for the rank correlations, distinct high-exposure episodes
# (runs of consecutive daily forecast times within a storm whose next-36 h exposure >= 1 million customer-hours), and the share of
# each detected episode's exposure that fell 12 h or more after the first alert.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
D=pd.read_csv('pilot/hrrr_eval2.csv'); X=D[D.win=='0-36 h'].reset_index(drop=True); X['event']=X.act>=1e6
W=D.pivot_table(index=['storm','k'],columns='win',values='act').reset_index()
storms=sorted(X.storm.unique()); rng=np.random.default_rng(1)
print('storms',len(storms),'forecast times',len(X),'storms with any high-exposure time',X[X.event].storm.nunique())
bs={c:[] for c in ('out','adv','clim','pers','fc')}
for _ in range(2000):
    s=rng.choice(storms,len(storms)); Z=pd.concat([X[X.storm==x] for x in s])
    for c in bs: bs[c].append(spearmanr(Z.act,Z[c])[0])
for c in bs: print(f'Spearman {c}: {spearmanr(X.act,X[c])[0]:.2f} (storm bootstrap 95% {np.nanpercentile(bs[c],2.5):.2f} to {np.nanpercentile(bs[c],97.5):.2f})')
def loso(col):
    alert=np.zeros(len(X),bool)
    for s in storms:
        tr=X[X.storm!=s]; ev=tr[tr.event]; thr=np.quantile(ev[col],0.2) if len(ev) else np.inf
        if col=='adv': thr=max(thr,1.0)
        m=(X.storm==s).values; alert[m]=X.loc[m,col].values>=thr
    return alert
res=[]
for col in ('fc','pers','clim','adv','out'):
    X['al']=loso(col); eps=[]
    for s,g in X.sort_values('k').groupby('storm'):
        ks=g.k.values; ev=g.event.values; al=g.al.values; i=0
        while i<len(ks):
            if ev[i]:
                j=i
                while j+1<len(ks) and ev[j+1] and ks[j+1]==ks[j]+1: j+=1
                # detected if an alert was issued at the episode's first forecast time or the one before it
                prev=i-1 if i>0 and ks[i-1]==ks[i]-1 else None
                det=al[i] or (prev is not None and al[prev]); early=prev is not None and al[prev]
                k0=ks[i]; w=W[(W.storm==s)&(W.k==k0)]
                late=float((w['12-24 h']+w['24-36 h']).iloc[0]/w['0-36 h'].iloc[0]) if len(w) and w['0-36 h'].iloc[0]>0 else np.nan
                eps.append(dict(storm=s,k0=k0,n=j-i+1,det=det,early=early,late=late,act=g.act.values[i:j+1].sum())); i=j+1
            else: i+=1
    E=pd.DataFrame(eps); fa_st=X[X.al&~X.event].storm.nunique(); ev_st=X[X.event].storm.unique()
    st_hit=sum(X[(X.storm==s)&X.event&X.al].shape[0]>0 for s in ev_st)
    res.append(dict(rule=col,episodes=len(E),episodes_detected=int(E.det.sum()),detected_24h_before=int(E.early.sum()),
                    exposure_share_detected=E.act[E.det].sum()/E.act.sum(),storms_with_events=len(ev_st),storms_hit=st_hit,storms_with_false_alerts=fa_st,
                    share_exposure_after_12h=np.nanmedian(E.late[E.det])))
R=pd.DataFrame(res); R.to_csv('pilot/hrrr_storm_level.csv',index=False); print(R.round(2).to_string())
