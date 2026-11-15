import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
src=open('pilot/reconcile.py').read(); exec(src[:src.index('# ---------- 1.')])
D=pd.read_pickle('pilot/decomp_met_records.pkl')
rows=[]
for (cell,iso),g in D.groupby(['cell','iso2']):
    s=pd.read_csv(f'pilot/power/{cell[0]}_{cell[1]}.csv').replace(-999,np.nan)
    m=mon[g.t0.values]; mm=pd.Series(m).mode()[0]
    sel=(mon==mm)&(yr<=2024); Y=pd.Index(yr[sel]).unique(); x=gpi.loc[Y].values
    T=pd.Series(s.T2M_MAX.values[sel]).groupby(yr[sel]).mean().values; Q=pd.Series(s.QV2M.values[sel]).groupby(yr[sel]).mean().values
    early=(Y<=2000); 
    rows.append(dict(cell=cell,iso2=iso,month=mm,w=(g.pE+g.pL).sum(),bT=np.polyfit(x,T,1)[0],bQ=np.polyfit(x,Q,1)[0],bQpct=np.polyfit(x,Q,1)[0]/Q.mean()*100,
                     dT=T[~early].mean()-T[early].mean(),pre2001_T=T[early].mean(),post_T=T[~early].mean()))
C=pd.DataFrame(rows)
for iso in ['BD','VN','CN','IN','PH']:
    c=C[C.iso2==iso]; print(iso,'cells',len(c),'months',c.month.value_counts().head(3).to_dict(),'bT median',round(c.bT.median(),2),'IQR',c.bT.quantile([.25,.75]).round(2).tolist(),'q %/K median',round(c.bQpct.median(),1), 'Clausius~7%/K')
# step change test around MERRA-2 observing changes: compare 1981-1999 vs 2000-2024 Tmax in BD cells with ERA5 Tw
