import pandas as pd, numpy as np, glob, os
s=open('pilot/cmip6_proj.py').read(); exec(s[:s.index('# ---- per model pattern scaling')])
rng=np.random.default_rng(13)
yrs_lf=A.season.values; L=A.lmic.values
d35f=(HI>=35).sum(1)
gsm=gpi.rolling(11,center=True,min_periods=5).mean()
def loadm(m,e,r):
    f=f'data/cmip6/{m}_{e}_{r}.npz'
    if not os.path.exists(f): return None
    d=dict(np.load(f)); n=min(len(d['gmst']),len(d['tasmax']),len(d['hurs']),len(d['yr'])); return {k:v[:n] for k,v in d.items()}
J=pd.read_csv('pilot/attr_jobs.csv'); rows=[]; per=[]
obs_u=uniq(HI>=40.6); obs_py=obs_u[obs_u.lmic].groupby('year').v.sum().reindex(range(2005,2025),fill_value=0)
d35_py_f=pd.Series(d35f[L]).groupby(yrs_lf[L]).sum().reindex(range(2005,2025),fill_value=0)
for (m,r),_ in J.groupby(['m','r']):
    h=loadm(m,'historical',r); n=loadm(m,'hist-nat',r)
    if h is None or n is None: continue
    for base in [(1995,2014),(1985,2014)]:
        sh=(h['yr']>=base[0])&(h['yr']<=base[1]); sn=(n['yr']>=base[0])&(n['yr']<=base[1])
        if sn.sum()<12*(base[1]-base[0]+1): continue
        dT=np.stack([h['tasmax'][sh&(h['mo']==k)].mean(0)-n['tasmax'][sn&(n['mo']==k)].mean(0) for k in range(1,13)])
        dH=np.stack([h['hurs'][sh&(h['mo']==k)].mean(0)-n['hurs'][sn&(n['mo']==k)].mean(0) for k in range(1,13)])
        dG=h['gmst'][sh].mean()-n['gmst'][sn].mean()
        fac=(gsm.loc[yrs_lf].values/gsm.loc[base[0]:base[1]].mean())[:,None]
        for scaled in [False,True]:
            sc=(gsm.loc[base[0]:base[1]].mean()/dG) if scaled else 1.0
            cfT=T-dT[mo-1,ci][:,None]*fac*sc; cfH=np.clip(RH-dH[mo-1,ci][:,None]*fac*sc,1,100); hi=hi_c(cfT,cfH)
            u=uniq(hi>=40.6); cf_py=u[u.lmic].groupby('year').v.sum().reindex(range(2005,2025),fill_value=0)
            d35c=(hi>=35).sum(1); d35_py_c=pd.Series(d35c[L]).groupby(yrs_lf[L]).sum().reindex(range(2005,2025),fill_value=0)
            rows.append(dict(model=m,member=r,base=f'{base[0]}-{base[1]}',scaled=scaled,dG=dG,frac_pd=1-cf_py.sum()/obs_py.sum(),frac_d35=1-d35_py_c.sum()/d35_py_f.sum()))
            per.append(((m,r,base,scaled),cf_py.values,d35_py_c.values))
    print(m,r,flush=True)
Rr=pd.DataFrame(rows); Rr.to_csv('pilot/attrib_members.csv',index=False)
pd.set_option('display.width',200)
print(Rr.groupby(['base','scaled','model'])[['frac_pd','frac_d35']].agg(['mean','min','max']).round(2))
# combined bootstrap: model -> member -> years (main: 1995-2014 scaled)
for base in ['1995-2014','1985-2014']:
  for scaled in [True,False]:
    sel=[p for p in per if f'{p[0][2][0]}-{p[0][2][1]}'==base and p[0][3]==scaled]
    bym={}
    for k,a,b in sel: bym.setdefault(k[0],[]).append((a,b))
    B=[]
    for _ in range(2000):
        m=rng.choice(list(bym)); a,b=bym[m][rng.integers(len(bym[m]))]; yi=rng.integers(0,20,20)
        B.append([1-a[yi].sum()/obs_py.values[yi].sum(),1-b[yi].sum()/d35_py_f.values[yi].sum()])
    B=np.array(B); print(base,'scaled' if scaled else 'unscaled','person-days',np.percentile(B[:,0],[5,50,95]).round(2),' d35 days',np.percentile(B[:,1],[5,50,95]).round(2))
