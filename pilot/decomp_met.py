import pandas as pd, numpy as np, itertools
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
rng=np.random.default_rng(11); rp406=0.953; ny={'E':20,'L':7}
R=pd.read_pickle('pilot/recon_records.pkl')
D=R[(R.foot=='100')&R.lmic&((R.season<=2000)|(R.season>=2018))].copy(); D['S']=np.where(D.season<=2000,'E','L')
gbar={'E':gpi.loc[1981:2000].mean(),'L':gpi.loc[2018:2024].mean()}; print('GMST means',gbar)
beta={}
def get_beta(cell,m):
    k=(cell,m)
    if k not in beta:
        s=series(*cell); sel=(mon==m)&(yr<=2024); a=pd.Series(s['T'][sel]).groupby(yr[sel]).mean(); beta[k]=np.polyfit(gpi.loc[a.index].values,a.values,1)[0]
    return beta[k]
def clim(cell,t0,years):
    s=series(*cell)['HI']; d=dates[t0]; out=[]
    for y in years:
        tt=np.searchsorted(dates.values,pd.Timestamp(year=y,month=d.month,day=min(d.day,28)).to_datetime64()); out.append((s[tt+1:tt+31]>=40.6).sum())
    return np.mean(out)
cnt={}
for i,row in enumerate(D.itertuples()):
    s=series(*row.cell); w=slice(row.t0+1,row.t0+31); T=s['T'][w]; RH=s['RH'][w]; b=get_beta(row.cell,mon[row.t0])
    obs=(s['HI'][w]>=40.6).sum()
    for H in 'EL':
        dT=b*(gbar[H]-gbar[row.S])
        cnt[(i,H,'met')]=obs if H==row.S else (hi_c(T+dT,RH)>=40.6).sum()
        cnt[(i,H,'metdry')]=obs if H==row.S else (hi_c(T+dT,np.clip(RH-dT,1,100))>=40.6).sum()
    cE=clim(row.cell,row.t0,range(1981,2001)); cL=clim(row.cell,row.t0,range(2018,2025)); c={'E':cE,'L':cL}
    for H in 'EL': cnt[(i,H,'count')]=obs if H==row.S else max(obs+c[H]-c[row.S],0)
D=D.reset_index(drop=True)
for m in ['met','metdry','count']:
    for H in 'EL': D[f'{m}_{H}']=[cnt[(i,H,m)]*rp406 for i in range(len(D))]
D['pE']=[pop_cell(*c,1990) for c in D.cell]; D['pL']=[pop_cell(*c,2021) for c in D.cell]
D.to_pickle('pilot/decomp_met_records.pkl')
def E(d,S,P,H,m): s=d[d.S==S]; return (s['p'+P]*s[f'{m}_{H}']).sum()/ny[S]/1e6
def shap(d,m):
    sh={f:0.0 for f in 'SPH'}
    for perm in itertools.permutations('SPH'):
        st={'S':'E','P':'E','H':'E'}; prev=E(d,st['S'],st['P'],st['H'],m)
        for f in perm: st[f]='L'; cur=E(d,st['S'],st['P'],st['H'],m); sh[f]+=(cur-prev)/6; prev=cur
    tot=E(d,'L','L','L',m)-E(d,'E','E','E',m); return {k:v/tot for k,v in sh.items()},{k:v for k,v in sh.items()},E(d,'E','E','E',m),E(d,'L','L','L',m)
rows=[]
for m in ['met','metdry','count']:
    for ex in ['none','CN','IN','VN','PH','BD','ALL5']:
        d=D if ex=='none' else D[~D.iso2.isin(['CN','IN','VN','PH','BD'] if ex=='ALL5' else [ex])]
        sh,ab,e0,e1=shap(d,m); rows.append(dict(method=m,excluded=ex,climate=sh['H'],population=sh['P'],storms=sh['S'],clim_abs=ab['H'],pop_abs=ab['P'],storm_abs=ab['S'],early=e0,late=e1))
    ids={S:D[D.S==S].idx.unique() for S in 'EL'}; grp={i:g for i,g in D.groupby('idx')}; B=[]
    for _ in range(300):
        sub=pd.concat([grp[i] for S in 'EL' for i in rng.choice(ids[S],len(ids[S]))]); sh,ab,_,_=shap(sub,m); B.append([sh['H'],sh['P'],sh['S'],ab['H'],ab['P'],ab['S']])
    B=np.array(B); r=rows[[x['method'] for x in rows].index(m)]
    for j,k in enumerate(['climate','population','storms','clim_abs','pop_abs','storm_abs']): r[k+'_lo'],r[k+'_hi']=np.percentile(B[:,j],[2.5,97.5])
O=pd.DataFrame(rows); pd.set_option('display.width',250); print(O.round(2).to_string()); O.to_csv('pilot/decomp_met.csv',index=False)
