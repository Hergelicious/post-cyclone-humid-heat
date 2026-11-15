import pandas as pd, numpy as np, sys
from global_land_mask import globe
import reverse_geocoder as rg
P=__import__('os').path.join(__import__('os').path.dirname(__import__('huracanpy').__file__),'_data','_ibtracs_files','jtwc.csv')
d=pd.read_csv(P,low_memory=False,parse_dates=['time'])
d=d[d.season>=1981].copy()
d['lon']=((d.lon+180)%360)-180
d['land']=globe.is_land(d.lat.values,d.lon.values)
d=d.sort_values(['track_id','time'])
d['prev_land']=d.groupby('track_id').land.shift(1)
d['prev_wind']=d.groupby('track_id').wind.shift(1)
lf=d[(d.land)&(d.prev_land==False)].copy()
lf['lf_wind']=lf[['wind','prev_wind']].max(axis=1)
lf=lf[lf.lf_wind>=34]  # tropical-storm strength or more at landfall
cc=rg.search(list(zip(lf.lat,lf.lon)),mode=1)
lf['iso2']=[c['cc'] for c in cc]
lf=lf.sort_values('lf_wind',ascending=False).drop_duplicates(['track_id','iso2'])
lf[['track_id','season','basin','time','lat','lon','lf_wind','iso2']].to_csv('pilot/landfalls.csv',index=False)
print(len(lf),'landfalls (>=34 kt, one per storm-country);', (lf.lf_wind>=64).sum(),'at >=64 kt')
print(lf.groupby(pd.cut(lf.season,[1980,1990,2000,2010,2017,2024])).size())
