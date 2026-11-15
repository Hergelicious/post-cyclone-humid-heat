# Robustness variants of the within county-storm restoration model (see us_restoration_heat.py)
import pandas as pd, numpy as np, statsmodels.api as sm
P=pd.read_csv('pilot/us_restoration_panel.csv'); P['sd']=P.storm+'|'+P.d.astype(str)
P=P.sort_values(['cs','d']); P['hlead']=P.groupby('cs').hmax.shift(-1)
def fit(d,xs,fe='d'):
    D=pd.get_dummies(d[fe],drop_first=True).astype(float)
    Z=pd.concat([d[['rest']+xs].astype(float).reset_index(drop=True),D.reset_index(drop=True)],axis=1); g=d.cs.values
    Z=Z-Z.groupby(g).transform('mean'); Z=Z.loc[:,Z.std()>1e-9]
    m=sm.OLS(Z.rest,Z.drop(columns='rest')).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.storm)[0]})
    k=xs[0]; return f'{100*m.params[k]:+.2f} points ({100*m.conf_int().loc[k,0]:+.2f} to {100*m.conf_int().loc[k,1]:+.2f}), P={m.pvalues[k]:.2f}, n={int(m.nobs)}'
print('base, day FE           ',fit(P,['hmax']))
print('storm-by-day FE        ',fit(P,['hmax'],'sd'))
print('excluding first day    ',fit(P[P.d>=1],['hmax']))
Q=P.dropna(subset=['hlead']); print('with next-day HI       ',fit(Q,['hmax','hlead'])); print('next-day HI (placebo)  ',fit(Q,['hlead','hmax']))
