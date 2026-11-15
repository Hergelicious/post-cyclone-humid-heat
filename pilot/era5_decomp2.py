import pandas as pd, numpy as np, os, itertools
C=pd.read_csv('pilot/era5_cells1.csv'); key={(int(a),int(b)):i for i,(a,b) in enumerate(zip(C.lat,C.lon))}
D=pd.read_pickle('pilot/decomp_met_records.pkl'); D=D[(D.season<=2000)|((D.season>=2018)&(D.season<=2021))].copy()
D['ci']=[key[(int(np.round(c[0])),int(np.round(c[1])))] for c in D.cell]
base=pd.Timestamp('1981-01-01'); cache={}
def get(kind,y,m):
    k=(kind,y,m)
    if k not in cache:
        f=f'data/era5h/{"daily" if kind=="obs" else "scen"}/{y}{m:02d}.npz'
        cache[k]=dict(np.load(f)) if os.path.exists(f) else None
    return cache[k]
vals={v:[] for v in ['obs','T','q','Tq','T_RHfix']}; okl=[]
for r in D.itertuples():
    days=[base+pd.Timedelta(days=int(r.t0)+1+j) for j in range(30)]; out={v:0 for v in vals}; good=True
    for dt in days:
        o=get('obs',dt.year,dt.month); s=get('scen',dt.year,dt.month)
        if o is None or s is None or dt.day-1>=len(o['HImax']): good=False; break
        out['obs']+=o['HImax'][dt.day-1,r.ci]>=40.6
        for v in ['T','q','Tq','T_RHfix']: out[v]+=s[v][dt.day-1,r.ci]>=40.6
    okl.append(good)
    for v in vals: vals[v].append(out[v] if good else np.nan)
for v in vals: D['e_'+v]=vals[v]
D['ok']=okl; D=D[D.ok].copy(); print('records usable',len(D),'of',len(okl))
D.to_pickle('pilot/era5_decomp_records.pkl')
ny={'E':20,'L':4}
def E(d,S,P,H,col):
    s=d[d.S==S]
    if col.startswith('e_'):
        x=s['e_obs'] if H==S else s['e_'+col[2:]]
    else:
        x=s[f'{col}_{H}']
    return (s['p'+P]*x).sum()/ny[S]/1e6
def shap(d,col):
    sh={f:0.0 for f in 'SPH'}
    for perm in itertools.permutations('SPH'):
        st={'S':'E','P':'E','H':'E'}; prev=E(d,st['S'],st['P'],st['H'],col)
        for f in perm: st[f]='L'; cur=E(d,st['S'],st['P'],st['H'],col); sh[f]+=(cur-prev)/6; prev=cur
    t=sum(sh.values()); return sh,t
rng=np.random.default_rng(3); rows=[]
methods=[('ERA5, T and q shifted','e_Tq'),('ERA5, T only (q fixed)','e_T'),('ERA5, q only','e_q'),('ERA5, T shifted, RH fixed','e_T_RHfix'),('POWER, T and RH shifted (same records)','metTH'),('POWER, T only, RH fixed (same records)','met')]
for lab,col in methods:
    for ex in ['none','CN','IN','VN','BD']:
        d=D if ex=='none' else D[D.iso2!=ex]
        sh,t=shap(d,col); r=dict(method=lab,excluded=ex,early=E(d,'E','E','E',col),late=E(d,'L','L','L',col),clim=sh['H'],pop=sh['P'],storm=sh['S'],clim_share=sh['H']/t if t else np.nan)
        if ex=='none':
            byy={y:g for y,g in d.groupby('season')}; ye=[y for y in byy if y<=2000]; yl=[y for y in byy if y>=2018]; B=[]
            for _ in range(300):
                sub=pd.concat([byy[y] for y in list(rng.choice(ye,len(ye)))+list(rng.choice(yl,len(yl)))]); s_,t_=shap(sub,col); B.append([s_['H'],s_['H']/t_])
            B=np.array(B); r.update(clim_lo=np.percentile(B[:,0],2.5),clim_hi=np.percentile(B[:,0],97.5),share_lo=np.percentile(B[:,1],2.5),share_hi=np.percentile(B[:,1],97.5))
        rows.append(r)
O=pd.DataFrame(rows); pd.set_option('display.width',250); print(O.round(2).to_string()); O.to_csv('pilot/era5_decomp.csv',index=False)
