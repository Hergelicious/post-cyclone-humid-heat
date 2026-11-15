import pandas as pd, numpy as np
from global_land_mask import globe
lf=pd.read_csv('pilot/landfalls.csv')
cells=set()
def landcell(la,lo):
    c0=(int(np.round(la)),int(np.round(lo)))
    cands=[(c0[0]+a,c0[1]+b) for a in (0,-1,1) for b in (0,-1,1)]
    cands=sorted(cands,key=lambda c:(c[0]-la)**2+(c[1]-lo)**2)
    for c in cands:
        if globe.is_land(c[0],((c[1]+180)%360)-180): return c
    return c0
lf['wlat'],lf['wlon']=zip(*[landcell(a,b) for a,b in zip(lf.lat,lf.lon)])
lf['wlon']=((lf.wlon+180)%360)-180
lf.to_csv('pilot/landfalls.csv',index=False)
u=lf[['wlat','wlon']].drop_duplicates(); u.to_csv('pilot/wbt_cells.csv',index=False); print(len(u),'1-degree cells')
