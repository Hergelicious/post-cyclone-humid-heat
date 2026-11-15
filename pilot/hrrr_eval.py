# Retrospective test of a forecast-based targeting rule for outage-heat exposure.
# At 00 UTC on each of days 0-6 after landfall, counties currently without power (EAGLE-I) are ranked using information available then:
# customers out now, HRRR forecast hours with HI >= 35 C over the next 36 h, an active NWS heat product, or the product of
# customers out x expected share still out (leave-one-storm-out decay) x forecast hot hours. The share of the next 36 h of
# measured simultaneous exposure (customers out x ERA5 HI >= 35 C) captured by the top-ranked counties is compared across rules.
import numpy as np, pandas as pd, glob, os
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('NAMES='):src.index('# ---- EAGLE-I rows')])
recs={r['storm']:r for r in pd.read_pickle('pilot/us_v2_recs_main.pkl')}
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False); lf={NAMES.get(t,t):tm for t,tm in zip(L.track_id,L.time)}
def es(t): return 6.112*np.exp(17.67*t/(t+243.5))
H=36; rows=[]; curves={}
# NWS heat products per county-hour (from us_heat_warnings.py inputs)
import importlib.util
W={}
if os.path.exists('pilot/us_heat_warn_flags.pkl'): W=pd.read_pickle('pilot/us_heat_warn_flags.pkl')
items=[]
for f in sorted(glob.glob('data/hrrr/*.npz')):
    nm=os.path.basename(f)[:-4]; z=np.load(f); r=recs[nm]; Ex=r['Ex']; HIe=r['HI']; nh=Ex.shape[0]
    t0=lf[nm].floor('h')-pd.Timedelta(hours=24); inits=pd.to_datetime(z['inits'])
    Tf=z['T']; Df=z['D']; HIf=hi_c(Tf,np.clip(100*es(Df)/es(Tf),1,100))
    A=W.get(nm)
    for k,ti in enumerate(inits):
        i0=int((ti-t0)/pd.Timedelta(hours=1))
        if i0<0 or i0+H>=nh: continue
        O=Ex[i0]; on=O>0
        if on.sum()<3: continue
        fut=Ex[i0+1:i0+H+1]; act=(fut*(HIe[i0+1:i0+H+1]>=35)).sum(0)
        hot_fc=np.nansum(HIf[k,:H]>=35,0).astype(float); hot_fc[np.all(np.isnan(HIf[k,:H]),0)]=np.nan
        ratio=np.where(on[None],fut/np.where(O>0,O,1)[None],np.nan)   # share still out at each lead
        adv=(A[i0+1:i0+H+1].any(0) if A is not None else np.zeros(len(O),bool))
        items.append(dict(storm=nm,k=k,O=O,act=act,hot_fc=hot_fc,HIf=HIf[k,:H],ratio=ratio,on=on,adv=adv))
storms=sorted({it['storm'] for it in items})
for s in storms:   # leave-one-storm-out decay of the share still out
    rr=np.concatenate([it['ratio'][:,it['on']] for it in items if it['storm']!=s],1); curves[s]=np.nanmean(np.clip(rr,0,1.5),1)
def capture(score,act,on,frac):
    idx=np.where(on)[0]; n=max(1,int(np.ceil(frac*len(idx)))); sc=np.nan_to_num(score[idx],nan=-1)
    top=idx[np.argsort(-sc,kind='stable')[:n]]; return act[top].sum()
res={}
for frac in (0.1,0.2):
    tot={k:0.0 for k in ['all','oracle','outage','heat','advisory','combined']}
    for it in items:
        on=it['on']; act=it['act']; tot['all']+=act[on].sum()
        pred=it['O']*np.nansum(curves[it['storm']][:,None]*(it['HIf']>=35),0)
        sc={'oracle':act,'outage':it['O'].astype(float),'heat':it['hot_fc']+1e-6*it['O'],'advisory':it['adv']*1e12+it['O'],'combined':pred}
        for k_,v in sc.items(): tot[k_]+=capture(v,act,on,frac)
    res[frac]={k:100*v/tot['all'] for k,v in tot.items() if k!='all'}
    print(f'top {int(frac*100)}% of counties without power at each forecast time: share of next-36 h simultaneous exposure captured (%)',{k:round(v,1) for k,v in res[frac].items()})
print('forecast times',len(items),'storms',len(storms),'total next-36 h exposure %.1f M customer-hours'%(sum(it['act'][it['on']].sum() for it in items)/1e6))
# skill of forecast hot hours against ERA5
# bootstrap over storms for the combined vs outage difference
rng=np.random.default_rng(1); diffs=[]
for b in range(500):
    ss=rng.choice(storms,len(storms)); tA=tC=tT=0.0
    for s in ss:
        for it in [x for x in items if x['storm']==s]:
            on=it['on']; act=it['act']; pred=it['O']*np.nansum(curves[s][:,None]*(it['HIf']>=35),0)
            tT+=act[on].sum(); tA+=capture(it['O'].astype(float),act,on,0.2); tC+=capture(pred,act,on,0.2)
    diffs.append(100*(tC-tA)/max(tT,1))
print('combined minus outage-only, top 20%%: %.1f points (95%% %.1f to %.1f, storms resampled)'%(res[0.2]['combined']-res[0.2]['outage'],*np.percentile(diffs,[2.5,97.5])))
pd.to_pickle(dict(res=res,diffs=diffs),'pilot/hrrr_eval.pkl')
# ---- timing: which forecast times (storm-days) will see large exposure? compare totals across forecast times
from scipy.stats import spearmanr
tot_act=np.array([it['act'][it['on']].sum() for it in items]); tot_out=np.array([it['O'][it['on']].sum() for it in items])
tot_pred=np.array([(it['O']*np.nansum(curves[it['storm']][:,None]*(it['HIf']>=35),0))[it['on']].sum() for it in items])
tot_adv=np.array([(it['O']*it['adv'])[it['on']].sum() for it in items])
print('across %d forecast times: Spearman with next-36 h exposure: customers out %.2f, customers out under a heat product %.2f, combined forecast %.2f'%(len(items),spearmanr(tot_act,tot_out)[0],spearmanr(tot_act,tot_adv)[0],spearmanr(tot_act,tot_pred)[0]))
big=tot_act>=np.percentile(tot_act,75)
from sklearn.metrics import roc_auc_score
print('identifying the top quarter of forecast times: AUC customers out %.2f, combined %.2f, heat product %.2f'%(roc_auc_score(big,tot_out),roc_auc_score(big,tot_pred),roc_auc_score(big,tot_adv)))
fc_hot=np.concatenate([np.nansum(it['HIf']>=35,0)[it['on']] for it in items]); 
pd.DataFrame(dict(storm=[it['storm'] for it in items],k=[it['k'] for it in items],act=tot_act,out=tot_out,pred=tot_pred,adv=tot_adv)).to_csv('pilot/hrrr_eval_times.csv',index=False)
