import pandas as pd, numpy as np, os
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def es(T): return 6.112*np.exp(17.67*T/(T+243.5))
D=pd.read_pickle('pilot/era5_decomp_records.pkl'); base=pd.Timestamp('1981-01-01')
need=set()
for r in D.itertuples():
    for j in range(30):
        dt=base+pd.Timedelta(days=int(r.t0)+1+j); need.add((dt.year,dt.month))
os.makedirs('data/era5h/night',exist_ok=True); print('months',len(need),flush=True)
for y,m in sorted(need):
    out=f'data/era5h/night/{y}{m:02d}.npz'
    if os.path.exists(out): continue
    a=np.load(f'data/era5h/Tp_{y}{m:02d}.npz'); T=a['T']; P=a['P']; Tw=np.load(f'data/era5h/Tw_{y}{m:02d}.npy'); n=min(len(T),len(Tw)); T,P,Tw=T[:n],P[:n],Tw[:n]
    Tw=np.minimum(Tw,T); e=np.clip(es(Tw)-6.53e-4*(1+0.000944*Tw)*P*(T-Tw),0.1,None); e=np.minimum(e,es(T)); RH=np.clip(100*e/es(T),1,100)
    nd=n//24; r_=lambda x:x[:nd*24].reshape(nd,24,-1)
    np.savez(out,Tmin=r_(T).min(1).astype('float32'),Twmin=r_(Tw).min(1).astype('float32'),HImax=r_(hi_c(T,RH)).max(1).astype('float32'))
print('daily done',flush=True)
cache={}
def get(y,m):
    if (y,m) not in cache: cache[(y,m)]=dict(np.load(f'data/era5h/night/{y}{m:02d}.npz'))
    return cache[(y,m)]
cols={'n26':[], 'n28':[], 'tw24':[], 'comp':[]}
for r in D.itertuples():
    c={k:0 for k in cols}
    for j in range(30):
        dt=base+pd.Timedelta(days=int(r.t0)+1+j); o=get(dt.year,dt.month); i=dt.day-1
        tmin=o['Tmin'][i,r.ci]; c['n26']+=tmin>=26; c['n28']+=tmin>=28; c['tw24']+=o['Twmin'][i,r.ci]>=24; c['comp']+=(tmin>=26)&(o['HImax'][i,r.ci]>=40.6)
    for k in cols: cols[k].append(c[k])
for k in cols: D[k]=cols[k]
D['day406']=D.e_obs
ny={'E':20,'L':4}; rows=[]
for k,lab in [('day406','Days with daily maximum heat index ≥40.6 °C'),('n26','Nights with minimum temperature ≥26 °C'),('n28','Nights with minimum temperature ≥28 °C'),('tw24','Nights with minimum wet-bulb temperature ≥24 °C'),('comp','Days ≥40.6 °C followed by a night ≥26 °C')]:
    e=(D[D.S=='E'].pE*D[D.S=='E'][k]).sum()/20/1e6; l=(D[D.S=='L'].pL*D[D.S=='L'][k]).sum()/4/1e6
    pr_e=D[D.S=='E'][k].mean(); pr_l=D[D.S=='L'][k].mean()
    rows.append(dict(metric=lab,early=e,late=l,ratio=l/e if e else np.nan,per_cell_early=pr_e,per_cell_late=pr_l))
# share of dangerous days whose following night stays >=26
O=pd.DataFrame(rows); print(O.round(2).to_string()); O.to_csv('pilot/night_heat.csv',index=False)
for S in 'EL': d=D[D.S==S]; print(S,'share of >=40.6 days with night >=26', round(d.comp.sum()/max(d.day406.sum(),1),2))
for c in ['CN','IN','VN','PH','BD']:
    d=D[D.iso2==c]; print(c, round((d[d.S=='E'].pE*d[d.S=='E'].n26).sum()/20/1e6,1), round((d[d.S=='L'].pL*d[d.S=='L'].n26).sum()/4/1e6,1))
