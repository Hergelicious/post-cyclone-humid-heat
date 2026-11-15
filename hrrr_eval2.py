# Prospective checks of the outage-heat forecast (HRRR, 00 UTC runs on days 0-6 after 15 landfalls, 2018-2024):
# (1) skill by lead window (0-6, 6-12, 12-24, 24-36 h) against two no-forecast baselines: persistence (hot hours in the 24 h before
#     the forecast time, same hour of day) and pre-storm climatology (mean diurnal hot hours over the 14 days before landfall);
# (2) a simple alert rule (alert when predicted exposure in the next 36 h exceeds a threshold) with the threshold chosen on the other
#     storms (leave-one-storm-out) to detect at least 80% of high-exposure forecast times there; events are forecast times whose
#     next-36 h exposure is at least 1 million customer-hours. Compared with an outage-only rule and a heat-product rule.
import numpy as np, pandas as pd, glob, os, sys, warnings
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('GAP=int'):src.index('# ---- CMIP6 fields')])
recs={r['storm']:r for r in pd.read_pickle('pilot/us_v2_recs_main.pkl')}
lf={NAMES.get(t,t):tm for t,tm in zip(L.track_id,L.time)}
W=pd.read_pickle('pilot/us_heat_warn_flags.pkl')
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
H=36; items=[]
for f in sorted(glob.glob('data/hrrr/*.npz')):
    nm=os.path.basename(f)[:-4]; z=np.load(f); r=recs[nm]; Ex=r['Ex']; HIe=r['HI']; nh=Ex.shape[0]; fl=r['fips']
    t0=lf[nm].floor('h')-pd.Timedelta(hours=24); inits=pd.to_datetime(z['inits'])
    Tf=z['T']; Df=z['D']; HIf=hi_c(Tf,np.clip(100*es(Df)/es(Tf),1,100))
    # pre-storm climatology: ERA5 hourly HI over days -14..-1 before landfall, mean hot-hour frequency by UTC hour of day
    Tp,RHp=heat_block(fl,lf[nm].normalize()-pd.Timedelta(days=14),14*24); clim=(hi_c(Tp,RHp)>=35).reshape(14,24,len(fl)).mean(0)
    for k,ti in enumerate(inits):
        i0=int((ti-t0)/pd.Timedelta(hours=1))
        if i0<24 or i0+H>=nh: continue
        O=Ex[i0]; on=O>0
        if on.sum()<3: continue
        hrs=np.arange(1,H+1); hod=(ti.hour+hrs)%24
        items.append(dict(storm=nm,k=k,O=O,on=on,fut=Ex[i0+1:i0+H+1],hot_obs=(HIe[i0+1:i0+H+1]>=35),hot_fc=(HIf[k,:H]>=35)&np.isfinite(HIf[k,:H]),
                          hot_pers=(HIe[i0-24+(hrs-1)%24]>=35),hot_clim=clim[hod],adv=W[nm][i0+1:i0+H+1].any(0)))
storms=sorted({it['storm'] for it in items})
def decay(excl):
    rr=np.concatenate([np.where(it['on'][None],it['fut']/np.where(it['O']>0,it['O'],1)[None],np.nan)[:,it['on']] for it in items if it['storm']!=excl],1)
    return np.nanmean(np.clip(rr,0,1.5),1)
DEC={s:decay(s) for s in storms}
wins={'0-6 h':(0,6),'6-12 h':(6,12),'12-24 h':(12,24),'24-36 h':(24,36),'0-36 h':(0,36)}
rows=[]
for it in items:
    d=DEC[it['storm']]; on=it['on']
    for w,(a,b) in wins.items():
        act=(it['fut'][a:b]*it['hot_obs'][a:b])[:,on].sum()
        pr=lambda hot:(it['O'][None]*d[a:b,None]*hot[a:b])[:,on].sum()
        rows.append(dict(storm=it['storm'],k=it['k'],win=w,act=act,out=it['O'][on].sum()*d[a:b].sum(),fc=pr(it['hot_fc']),pers=pr(it['hot_pers']),clim=pr(it['hot_clim'].astype(float)),
                         adv=(it['O']*it['adv'])[on].sum()))
D=pd.DataFrame(rows); D.to_csv('pilot/hrrr_eval2.csv',index=False)
print('== skill across %d forecast times (Spearman with measured exposure in each window; AUC for the top quarter)'%D[D.win=='0-36 h'].shape[0])
for w in wins:
    X=D[D.win==w]; big=X.act>=X.act.quantile(.75)
    s={c:spearmanr(X.act,X[c])[0] for c in ('out','adv','clim','pers','fc')}; a={c:roc_auc_score(big,X[c]) for c in ('out','adv','clim','pers','fc')}
    print(f'  {w:8s} Spearman: outage {s["out"]:.2f}, heat product {s["adv"]:.2f}, climatology {s["clim"]:.2f}, persistence {s["pers"]:.2f}, HRRR {s["fc"]:.2f} | AUC: outage {a["out"]:.2f}, product {a["adv"]:.2f}, clim {a["clim"]:.2f}, pers {a["pers"]:.2f}, HRRR {a["fc"]:.2f}')
# ---- alert rule, leave one storm out
X=D[D.win=='0-36 h'].reset_index(drop=True); EV=1e6; X['event']=X.act>=EV
res={}
for col in ('fc','clim','pers','out','adv'):
    alert=np.zeros(len(X),bool)
    for s in storms:
        tr=X[X.storm!=s]; ev=tr[tr.event]
        thr=np.quantile(ev[col],0.2) if len(ev) else np.inf          # detect >= 80% of training events
        if col=='adv': thr=max(thr,1.0)
        m=X.storm==s; alert[m]=X.loc[m,col]>=thr
    hit=(alert&X.event).sum(); miss=(~alert&X.event).sum(); fa=(alert&~X.event).sum()
    res[col]=dict(events=int(X.event.sum()),hits=int(hit),misses=int(miss),false_alerts=int(fa),alerts=int(alert.sum()),
                  exposure_in_hits=100*X.act[alert&X.event].sum()/X.act[X.event].sum())
    if col=='fc': X['alert_fc']=alert
R=pd.DataFrame(res).T; print('== alert rule (event: next-36 h exposure >= 1 million customer-hours; thresholds set leave-one-storm-out)'); print(R.round(1).to_string())
print('missed events (HRRR rule):'); print(X[X.event&~X.alert_fc][['storm','k','act','fc','out']].assign(act=lambda d:d.act/1e6,fc=lambda d:d.fc/1e6,out=lambda d:d.out/1e6).round(2).to_string())
R.to_csv('pilot/hrrr_alert_rule.csv')
