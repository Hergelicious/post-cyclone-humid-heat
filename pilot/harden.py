import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
rng=np.random.default_rng(7)
ev=pd.read_csv('pilot/power_events_exposure.csv'); ev['reg']=(np.round(ev.lat/2)*2).astype(str)+'_'+(np.round(ev.lon/2)*2).astype(str)
lf=pd.read_csv('pilot/landfalls.csv'); ev['basin']=lf.basin.values[ev.idx]
e5=pd.read_csv('pilot/era5_events.csv'); e5['reg']=(np.round(e5.lat/2)*2).astype(str)+'_'+(np.round(e5.lon/2)*2).astype(str); e5['basin']=lf.basin.values[e5.idx]
cl=lambda d: dict(cov_type='cluster',cov_kwds={'groups':d.iso2.astype('category').cat.codes})
out=[]
# (a) Poisson (PPML) with region FE -> % change per C
for d,y,lab in [(ev,'d35','POWER HI>=35'),(ev,'d406','POWER HI>=40.6'),(e5,'tw26','ERA5 Tw>=26'),(e5,'tw28','ERA5 Tw>=28')]:
    dd=d[d.groupby('reg')[y].transform('sum')>0]
    m=smf.glm(f'{y} ~ gpi + C(reg)',data=dd,family=sm.families.Poisson()).fit(**cl(dd))
    out.append(('Poisson, region FE',lab,len(dd),f"{(np.exp(m.params.gpi)-1)*100:.0f}% per °C",f"{(np.exp(m.conf_int().loc['gpi'][0])-1)*100:.0f} to {(np.exp(m.conf_int().loc['gpi'][1])-1)*100:.0f}%"))
# (b) country cluster bootstrap of OLS coefficient
def boot(d,y,B=499):
    cs=d.iso2.unique(); g={c:d[d.iso2==c] for c in cs}; bs=[]
    for b in range(B):
        s=pd.concat([g[c] for c in rng.choice(cs,len(cs))],ignore_index=True)
        try: bs.append(smf.ols(f'{y} ~ gpi + C(reg)',data=s).fit().params.gpi)
        except Exception: pass
    return np.percentile(bs,[2.5,97.5])
for d,y,lab in [(ev,'d35','POWER HI>=35'),(ev,'d406','POWER HI>=40.6'),(e5,'tw26','ERA5 Tw>=26'),(e5,'tw28','ERA5 Tw>=28')]:
    b=smf.ols(f'{y} ~ gpi + C(reg)',data=d).fit().params.gpi; lo,hi=boot(d,y)
    out.append(('Country bootstrap (499)',lab,len(d),f"{b:.2f} days per °C",f"{lo:.2f} to {hi:.2f}"))
# (c) leave one basin out
for bsn in sorted(ev.basin.dropna().unique()):
    for d,y,lab in [(ev,'d35','POWER HI>=35'),(e5,'tw26','ERA5 Tw>=26')]:
        s=d[d.basin!=bsn]; m=smf.ols(f'{y} ~ gpi + C(reg)',data=s).fit(**cl(s))
        out.append((f'Leave out basin {bsn}',lab,len(s),f"{m.params.gpi:.2f} days per °C",f"{m.conf_int().loc['gpi'][0]:.2f} to {m.conf_int().loc['gpi'][1]:.2f}"))
R=pd.DataFrame(out,columns=['test','outcome','n','estimate','95% CI']); print(R.to_string()); R.to_csv('pilot/harden.csv',index=False)
