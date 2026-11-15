import pandas as pd, numpy as np
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
rp=0.953
C=pd.read_csv('pilot/corridor_cells.csv',parse_dates=['tpass']); lf2=lf.set_index('index')
C['season']=lf2.season.values[C.idx]; C['lmic']=lf2.lmic.values[C.idx]
def run(df,tcol):
    U={}
    for r in df.itertuples():
        s=series(r.clat,r.clon)
        if s is None: continue
        t0=np.searchsorted(dates.values,getattr(r,tcol).normalize().to_datetime64()); m=s['HI'][t0+1:t0+31]>=40.6
        p=pop_cell(r.clat,r.clon,r.season)
        dd=U.setdefault((r.clat,r.clon),{})
        for di in np.nonzero(m)[0]:
            k=t0+1+di
            if p>dd.get(k,(0,0))[0]: dd[k]=(p*rp,r.lmic)
    q=pd.DataFrame([(yr[k],v,l) for dd in U.values() for k,(v,l) in dd.items()],columns=['year','v','lmic'])
    a=q[q.lmic].groupby('year').v.sum().reindex(range(1981,2025),fill_value=0)/1e6
    return a
corr=run(C,'tpass')
# 100 km landfall radius, same uncalibrated-ratio treatment, for like-for-like comparison
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[R.foot=='100'].copy(); R['clat']=[c[0] for c in R.cell]; R['clon']=[c[1] for c in R.cell]; R['tl']=dates[R.t0.values]
rad=run(R,'tl')
out=pd.DataFrame(dict(corridor=corr,radius100=rad)); out.to_csv('pilot/corridor_annual.csv')
rng=np.random.default_rng(1)
for c in out:
    a=out[c]; e=a.loc[1981:2000].mean(); l=a.loc[2018:2024].mean()
    bs=[a.loc[rng.choice(range(2018,2025),7)].mean()/a.loc[rng.choice(range(1981,2001),20)].mean() for _ in range(2000)]
    print(c,'early',round(e,0),'mid',round(a.loc[2001:2017].mean(),0),'late',round(l,0),'ratio',round(l/e,2),np.percentile(bs,[2.5,97.5]).round(2))
print('corridor cells per landfall median',C.groupby('idx').size().median())
