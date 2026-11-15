import pandas as pd, numpy as np, itertools
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
D=pd.read_pickle('pilot/decomp_met_records.pkl'); rp406=0.953; ny={'E':20,'L':7}; rng=np.random.default_rng(12)
gbar={'E':gpi.loc[1981:2000].mean(),'L':gpi.loc[2018:2024].mean()}
bb={}
def betas(cell,m):
    k=(cell,m)
    if k not in bb:
        s=series(*cell); sel=(mon==m)&(yr<=2024); x=gpi.loc[pd.Index(yr[sel]).unique()].values
        aT=pd.Series(s['T'][sel]).groupby(yr[sel]).mean().values; aH=pd.Series(s['RH'][sel]).groupby(yr[sel]).mean().values
        bb[k]=(np.polyfit(x,aT,1)[0],np.polyfit(x,aH,1)[0])
    return bb[k]
for H in 'EL': D[f'metTH_{H}']=0.0
vals={'E':[], 'L':[]}; bts=[]
for i,row in enumerate(D.itertuples()):
    s=series(*row.cell); w=slice(row.t0+1,row.t0+31); bT,bH=betas(row.cell,mon[row.t0]); bts.append((row.iso2,bT,bH))
    obs=(s['HI'][w]>=40.6).sum()
    for H in 'EL':
        dg=gbar[H]-gbar[row.S]
        vals[H].append(rp406*(obs if H==row.S else (hi_c(s['T'][w]+bT*dg,np.clip(s['RH'][w]+bH*dg,1,100))>=40.6).sum()))
for H in 'EL': D[f'metTH_{H}']=vals[H]
BT=pd.DataFrame(bts,columns=['iso2','bT','bH']); print('local betas median by country (T K/K, RH pts/K):'); print(BT.groupby('iso2')[['bT','bH']].median().reindex(['CN','IN','VN','PH','BD','MX']).round(2))
def E(d,S,P,H,m): s=d[d.S==S]; return (s['p'+P]*s[f'{m}_{H}']).sum()/ny[S]/1e6
def shap(d,m):
    sh={f:0.0 for f in 'SPH'}
    for perm in itertools.permutations('SPH'):
        st={'S':'E','P':'E','H':'E'}; prev=E(d,st['S'],st['P'],st['H'],m)
        for f in perm: st[f]='L'; cur=E(d,st['S'],st['P'],st['H'],m); sh[f]+=(cur-prev)/6; prev=cur
    tot=E(d,'L','L','L',m)-E(d,'E','E','E',m); return {k:v/tot for k,v in sh.items()},sh
sh,ab=shap(D,'metTH'); print('metTH shares',{k:round(v,2) for k,v in sh.items()},{k:round(v,0) for k,v in ab.items()})
ids={S:D[D.S==S].idx.unique() for S in 'EL'}; grp={i:g for i,g in D.groupby('idx')}; B=[]
for _ in range(300):
    sub=pd.concat([grp[i] for S in 'EL' for i in rng.choice(ids[S],len(ids[S]))]); a,b=shap(sub,'metTH'); B.append([a['H'],a['P'],a['S'],b['H'],b['P'],b['S']])
B=np.array(B); print('CIs',np.percentile(B,[2.5,97.5],axis=0).round(2))
for ex in ['CN','IN','ALL4']:
    d=D[~D.iso2.isin(['CN','IN','VN','PH'] if ex=='ALL4' else [ex])]; a,b=shap(d,'metTH'); print(ex,{k:round(v,2) for k,v in a.items()})
D.to_pickle('pilot/decomp_met_records.pkl')
pd.DataFrame([dict(method='metTH',climate=sh['H'],population=sh['P'],storms=sh['S'],clim_abs=ab['H'],pop_abs=ab['P'],storm_abs=ab['S'],**{f'{k}_{q}':v for j,k in enumerate(['climate','population','storms','clim_abs','pop_abs','storm_abs']) for q,v in zip(['lo','hi'],np.percentile(B[:,j],[2.5,97.5]))})]).to_csv('pilot/decomp_metTH.csv',index=False)
