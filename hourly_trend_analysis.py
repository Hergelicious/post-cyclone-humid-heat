import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
g=pd.read_csv('pilot/gmst.csv'); g=g[g.Source=='GISTEMP'].set_index('Year').Mean; gpi=g-g.loc[1880:1900].mean()
S=pd.read_csv('pilot/hourly_trend_cells.csv'); rows=[]
for r in S.itertuples():
    d=pd.read_csv(f'pilot/power/{r.lat}_{r.lon}.csv').replace(-999,np.nan); d['date']=pd.to_datetime(d.YEAR.astype(str)+d.DOY.astype(str).str.zfill(3),format='%Y%j'); d['MO']=d.date.dt.month
    q=d.QV2M/1000; p=d.PS*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*d.T2M_MAX/(d.T2M_MAX+243.5)); d['RHa']=np.clip(100*e/es,1,100); d['HIa']=hi_c(d.T2M_MAX.values,d.RHa.values)
    for y in range(2001,2025):
        fn=f'data/hourly_trend/{r.lat}_{r.lon}_{y}.csv'
        if not os.path.exists(fn): continue
        h=pd.read_csv(fn).replace(-999,np.nan); h=h[h.MO==r.month]
        if len(h)<24*20: continue
        h['HI']=hi_c(h.T2M.values,h.RH2M.values); h['day']=h.DY
        idx=h.groupby('day').HI.idxmax(); hm=h.loc[idx]
        dd=d[(d.YEAR==y)&(d.MO==r.month)]
        rows.append(dict(iso2=r.iso2,cell=f'{r.lat}_{r.lon}',year=y,G=gpi[y],T_hmax=hm.T2M.mean(),RH_hmax=hm.RH2M.mean(),HI_h=hm.HI.mean(),q_h=h.QV2M.mean(),
                         Tmax_d=dd.T2M_MAX.mean(),RH_a=dd.RHa.mean(),HI_a=dd.HIa.mean(),q_d=dd.QV2M.mean(),n35_h=(hm.HI>=35).sum(),n35_a=(dd.HIa>=35).sum()))
X=pd.DataFrame(rows); X.to_csv('pilot/hourly_trend_table.csv',index=False)
def sl(df,v): 
    out=[]
    for c,gc in df.groupby('cell'):
        if len(gc)>=15: out.append(np.polyfit(gc.G,gc[v],1)[0])
    return np.median(out) if out else np.nan
res=[]
for iso,gi in X.groupby('iso2'):
    res.append(dict(iso2=iso,cells=gi.cell.nunique(),**{f'b_{v}':round(sl(gi,v),2) for v in ['Tmax_d','T_hmax','RH_a','RH_hmax','HI_a','HI_h','n35_a','n35_h']},q_pct_d=round(sl(gi.assign(qq=gi.q_d/gi.q_d.mean()*100),'qq'),1),q_pct_h=round(sl(gi.assign(qq=gi.q_h/gi.q_h.mean()*100),'qq'),1),
                    bias_RH=round((gi.RH_a-gi.RH_hmax).mean(),1)))
R=pd.DataFrame(res); pd.set_option('display.width',250); print(R.to_string()); R.to_csv('pilot/hourly_trend_summary.csv',index=False)
