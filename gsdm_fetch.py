import pandas as pd, numpy as np, os, requests
from concurrent.futures import ThreadPoolExecutor
st=pd.read_csv('data/gsdm/stations.csv',encoding='utf-8-sig')
lf=pd.read_csv('pilot/landfalls.csv')
R=6371
def hav(la1,lo1,la2,lo2):
    la1,lo1,la2,lo2=map(np.radians,(la1,lo1,la2,lo2))
    a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D=hav(lf.lat.values[:,None],lf.lon.values[:,None],st.Latitude.values[None,:],st.Longitude.values[None,:])
near=(D<=100)
st['n_lf']=near.sum(0); sel=st[st.n_lf>0]
print(len(sel),'stations within 100 km of >=1 landfall;', (near.any(1)).sum(),'landfalls with a station')
sel.to_csv('data/gsdm/stations_near.csv',index=False)
pairs=[(i,st.Code.iloc[j],D[i,j]) for i,j in zip(*np.nonzero(near))]
pd.DataFrame(pairs,columns=['lf_idx','code','km']).to_csv('data/gsdm/pairs.csv',index=False)
os.makedirs('data/gsdm/nc',exist_ok=True)
def get(code):
    fn=f'data/gsdm/nc/{code}.nc'
    if os.path.exists(fn): return 1
    u=f'https://zenodo.org/api/records/7014332/files/GSDM-WBT_{code}_19810101-20201231.nc/content'
    r=requests.get(u,timeout=60)
    if r.status_code==200: open(fn,'wb').write(r.content); return 1
    return 0
with ThreadPoolExecutor(4) as ex: print(sum(ex.map(get,sel.Code)),'downloaded')
