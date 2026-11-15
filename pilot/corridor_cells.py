import pandas as pd, numpy as np, os
from global_land_mask import globe
src=open('pilot/reconcile.py').read(); exec(src[src.index('def cells_within'):src.index('def wind_R')])
P=__import__('os').path.join(__import__('os').path.dirname(__import__('huracanpy').__file__),'_data','_ibtracs_files','jtwc.csv')
d=pd.read_csv(P,low_memory=False,parse_dates=['time']); d=d[d.season>=1981]; d['lon']=((d.lon+180)%360)-180
lf=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'])
rows=[]
for i,r in lf.iterrows():
    t=d[d.track_id==r.track_id].sort_values('time'); t=t[(t.time>=r.time)&(t.time<=r.time+pd.Timedelta(hours=72))].set_index('time')[['lat','lon','wind']]
    t=t[~t.index.duplicated()]
    if len(t)>1:
        lon_un=np.unwrap(np.radians(t.lon.values)); t['lon']=np.degrees(lon_un)
        t=t.resample('3h').mean().interpolate(); t['lon']=((t.lon+180)%360)-180
    t['land']=globe.is_land(t.lat.values,t.lon.values)
    # keep until first return to water or wind<25 kt
    keep=[]
    for tt,p in t.iterrows():
        if not p.land or (pd.notna(p.wind) and p.wind<25): 
            if keep: break
            else: continue
        keep.append((tt,p.lat,p.lon))
    if not keep: keep=[(r.time,r.lat,r.lon)]
    best={}
    for tt,la,lo in keep:
        for c in cells_within(la,lo,100):
            if c not in best: best[c]=tt
    for c,tt in best.items(): rows.append((i,c[0],c[1],tt,len(keep)))
C=pd.DataFrame(rows,columns=['idx','clat','clon','tpass','npts']); C.to_csv('pilot/corridor_cells.csv',index=False)
u=C[['clat','clon']].drop_duplicates(); have=u.apply(lambda r: os.path.exists(f'pilot/power/{r.clat}_{r.clon}.csv'),axis=1)
print(len(C),'cell-landfalls; unique cells',len(u),'missing',(~have).sum()); u[~have].to_csv('pilot/corridor_missing.csv',index=False)
print(C.groupby('idx').size().describe())
