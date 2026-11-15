import pandas as pd, numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
src=open('pilot/power_more.py').read(); exec(src[src.index('def hi_c'):src.index('cache={}')])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
cache={}
def series(fn):
    if fn not in cache:
        s=pd.read_csv(fn).replace(-999,np.nan); s['date']=pd.to_datetime(s.YEAR.astype(str)+s.DOY.astype(str).str.zfill(3),format='%Y%j')
        q=s.QV2M/1000; p=s.PS*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX/(s.T2M_MAX+243.5)); s['HI']=hi_c(s.T2M_MAX,np.clip(100*e/es,1,100)); cache[fn]=s.set_index('date')
    return cache[fn]
def feats(lat,lon,t,wind,fn,season):
    s=series(fn); t0=t.normalize(); ant=s.loc[t0-pd.Timedelta(days=30):t0-pd.Timedelta(days=1)]
    doy=t0.dayofyear; off=(s.index.dayofyear-doy)%366; cl=s[(off>=1)&(off<=30)&(s.index.year<=2000)&(s.index.year!=season)]
    w=s.loc[t0+pd.Timedelta(days=1):t0+pd.Timedelta(days=30)]
    return dict(lat=lat,abslat=abs(lat),lon=lon,msin=np.sin(2*np.pi*doy/365),mcos=np.cos(2*np.pi*doy/365),wind=wind,
        ant_mean=ant.HI.mean(),ant_max=ant.HI.max(),ant_d35=(ant.HI>=35).sum(),clim_d35=(cl.HI>=35).groupby(cl.index.year).sum().mean(),
        clim_d406=(cl.HI>=40.6).groupby(cl.index.year).sum().mean(),clim_max=cl.HI.groupby(cl.index.year).max().mean(),gmst=gpi.get(season,np.nan),
        y406=int((w.HI>=40.6).sum()>0),y35_7=int((w.HI>=35).sum()>=7),season=season)
D=pd.read_csv('pilot/ml_train.csv'); Dx=pd.read_csv('pilot/ml_test2526.csv')
D['gmst']=D.season.map(lambda y: gpi.get(y-1,np.nan)); Dx['gmst']=Dx.season.map(lambda y: gpi.get(y-1,np.nan))
print('missing gmst',D.gmst.isna().sum(),Dx.gmst.isna().sum())
F=['abslat','lat','lon','msin','mcos','wind','ant_mean','ant_max','ant_d35','clim_d35','clim_d406','clim_max','gmst']
res=[]
for tgt in ['y406','y35_7']:
    y=D[tgt].values; p_gb=np.zeros(len(D)); p_lr=np.zeros(len(D)); p_cl=np.zeros(len(D))
    for tr,te in GroupKFold(n_splits=len(D.season.unique())).split(D,y,groups=D.season):
        gb=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,l2_regularization=1.0,random_state=0).fit(D.iloc[tr][F],y[tr]); p_gb[te]=gb.predict_proba(D.iloc[te][F])[:,1]
        lr=LogisticRegression(max_iter=2000).fit(D.iloc[tr][['clim_d406' if tgt=='y406' else 'clim_d35','ant_max','gmst']],y[tr]); p_lr[te]=lr.predict_proba(D.iloc[te][['clim_d406' if tgt=='y406' else 'clim_d35','ant_max','gmst']])[:,1]
        p_cl[te]=D.iloc[te]['clim_d406' if tgt=='y406' else 'clim_d35'].values
    for nm,p in [('gradient boosting',p_gb),('logistic (3 features)',p_lr),('climatology only (rank)',p_cl)]:
        res.append((tgt,nm,'leave-year-out',round(roc_auc_score(y,p),3),round(average_precision_score(y,p),3),round(brier_score_loss(y,np.clip(p,0,1)),3) if 'rank' not in nm else np.nan,round(y.mean(),3)))
    gb=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,l2_regularization=1.0,random_state=0).fit(D[F],y)
    px=gb.predict_proba(Dx[F])[:,1]; yx=Dx[tgt].values
    res.append((tgt,'gradient boosting','2025-2026 out-of-sample (n=%d)'%len(yx),round(roc_auc_score(yx,px),3),round(average_precision_score(yx,px),3),round(brier_score_loss(yx,px),3),round(yx.mean(),3)))
    if tgt=='y406':
        import shap; ex=shap.TreeExplainer(gb); sv=ex.shap_values(D[F]); imp=pd.Series(np.abs(sv).mean(0),index=F).sort_values(ascending=False); print('SHAP mean |value|:'); print(imp.round(3).to_string()); imp.to_csv('pilot/ml_shap_lag.csv')
        np.save('pilot/ml_shap_values_lag.npy',sv); pd.Series(p_gb).to_csv('pilot/ml_oof_p406_lag.csv',index=False)
        # calibration deciles
        q=pd.qcut(p_gb,10,duplicates='drop'); cal=pd.DataFrame({'p':p_gb,'y':y}).groupby(q,observed=True).agg(pred=('p','mean'),obs=('y','mean'),n=('y','size')); print(cal.round(3)); cal.to_csv('pilot/ml_calibration_lag.csv')
R=pd.DataFrame(res,columns=['target','model','evaluation','AUC','AP','Brier','base_rate']); print(R.to_string()); R.to_csv('pilot/ml_results_lag.csv',index=False)
