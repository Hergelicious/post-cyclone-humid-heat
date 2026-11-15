import pandas as pd, numpy as np, itertools
U=pd.read_csv('pilot/robust_unique_celldays.csv'); U=U[U.lmic]
U['c']=np.where(U.iso2.isin(['CN','IN','VN','PH','BD']),U.iso2,'other')
U['per']=pd.cut(U.year,[1980,2000,2017,2024],labels=['E','M','L']); ny={'E':20,'M':17,'L':7}
A=U.groupby(['c','per'],observed=True).pd.sum().unstack()/pd.Series(ny)/1e6
R=pd.read_pickle('pilot/recon_records.pkl'); R=R[(R.foot=='100')&R.lmic]; R['c']=np.where(R.iso2.isin(['CN','IN','VN','PH','BD']),R.iso2,'other')
R['per']=pd.cut(R.season,[1980,2000,2017,2024],labels=['E','M','L'])
pp=R.groupby(['c','per'],observed=True).agg(pd_=('pd406','sum'),pop=('pop','sum'),n=('idx','nunique'))
pp['days_per_person']=pp.pd_/pp['pop']; pp['landfalls_per_yr']=pp.n/pp.index.get_level_values(1).map(ny).astype(float); pp['pop_per_landfall_M']=pp['pop']/pp.n/1e6
out=A.round(1).join(pp.days_per_person.unstack().round(2).add_prefix('dpp_')).join(pp.landfalls_per_yr.unstack().round(2).add_prefix('lf_')).join(pp.pop_per_landfall_M.unstack().round(1).add_prefix('pop_'))
print(out); out.to_csv('pilot/country_exposure.csv')
ann=U.groupby(['year','c']).pd.sum().unstack(fill_value=0).reindex(range(1981,2025),fill_value=0)/1e6; ann.to_csv('pilot/country_annual.csv')
# year-block bootstrap of decomposition (metTH)
D=pd.read_pickle('pilot/decomp_met_records.pkl'); nyD={'E':20,'L':7}
def E(d,S,P,H): s=d[d.S==S]; return (s['p'+P]*s[f'metTH_{H}']).sum()/nyD[S]/1e6
def shap(d):
    sh={f:0.0 for f in 'SPH'}
    for perm in itertools.permutations('SPH'):
        st={'S':'E','P':'E','H':'E'}; prev=E(d,st['S'],st['P'],st['H'])
        for f in perm: st[f]='L'; cur=E(d,st['S'],st['P'],st['H']); sh[f]+=(cur-prev)/6; prev=cur
    return sh
rng=np.random.default_rng(8); byy={y:g for y,g in D.groupby('season')}; ye=[y for y in byy if y<=2000]; yl=[y for y in byy if y>=2018]; B=[]
for _ in range(500):
    sub=pd.concat([byy[y] for y in list(rng.choice(ye,20))+list(rng.choice(yl,7))]); sh=shap(sub); t=sum(sh.values()); B.append([sh['H'],sh['P'],sh['S'],sh['H']/t,sh['P']/t,sh['S']/t])
B=np.array(B); print('year-block decomposition 95%: abs H,P,S',np.percentile(B[:,:3],[2.5,97.5],axis=0).round(0).tolist(),' shares',np.percentile(B[:,3:],[2.5,97.5],axis=0).round(2).tolist())
pd.DataFrame(np.percentile(B,[2.5,97.5],axis=0),columns=['H','P','S','sH','sP','sS'],index=['lo','hi']).to_csv('pilot/decomp_yearblock.csv')
