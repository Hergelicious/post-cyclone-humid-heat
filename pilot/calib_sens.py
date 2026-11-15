import pandas as pd, numpy as np
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[(R.foot=='100')].copy()
R['rh']=[np.nanmean(series(*c)['RH'][t+1:t+31]) for c,t in zip(R.cell,R.t0)]
cal=R[(R.season>=2001)&R.h406.notna()]
q=np.quantile(cal.rh,[1/3,2/3]); print('tercile cuts',q.round(1))
C=pd.read_csv('pilot/robust_calibration.csv',index_col=0)
terc=dict(dry=C.loc['dry','pd406'],mid=C.loc['mid','pd406'],humid=C.loc['humid','pd406'])
ctry={k:C.loc[k,'pd406'] for k in ['AU','CN','MX','PH','US','VN']}
E_=R[(R.season<=2000)&R.lmic].copy()
E_['t']=np.where(E_.rh<q[0],'dry',np.where(E_.rh<q[1],'mid','humid'))
base=(E_['pop']*E_.d406*0.953).sum()
alt_t=(E_['pop']*E_.d406*E_.t.map(terc)).sum()
alt_c=(E_['pop']*E_.d406*E_.iso2.map(ctry).fillna(0.953)).sum()
both=(E_['pop']*E_.d406*np.where(E_.iso2.isin(ctry),E_.iso2.map(ctry),E_.t.map(terc))).sum()
lo=(E_['pop']*E_.d406*min(terc.values())).sum(); hi=(E_['pop']*E_.d406*max(terc.values())).sum()
early=137.14; late=535.31
rows=[]
for lab,v in [('pooled 0.953',base),('humidity tercile',alt_t),('country (pooled elsewhere)',alt_c),('country, else tercile',both),('lowest tercile ratio everywhere',lo),('highest tercile ratio everywhere',hi)]:
    e=early*v/base; rows.append(dict(scheme=lab,early_change=v/base-1,early=e,growth=late/e))
O=pd.DataFrame(rows); print(O.round(3).to_string()); O.to_csv('pilot/calib_sens.csv',index=False)
print('tercile mix of early person-days', E_.assign(pd=E_['pop']*E_.d406).groupby('t').pd.sum().div((E_['pop']*E_.d406).sum()).round(3).to_dict())
print('country mix', E_.assign(pd=E_['pop']*E_.d406).groupby('iso2').pd.sum().div((E_['pop']*E_.d406).sum()).sort_values().tail(6).round(3).to_dict())
