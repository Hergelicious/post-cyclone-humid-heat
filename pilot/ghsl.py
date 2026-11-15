import os, zipfile, numpy as np, rasterio, requests
from rasterio.windows import Window
for ep in [1980,2000,2020]:
    out=f'data/ghsl/pop_{ep}_0p1.npy'
    if os.path.exists(out): continue
    n=f'GHS_POP_E{ep}_GLOBE_R2023A_4326_30ss'
    tifp=f'data/ghsl/{n}_V1_0.tif'
    if not os.path.exists(tifp):
        u=f'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/{n}/V1-0/{n}_V1_0.zip'
        zf=f'data/ghsl/{n}.zip'
        if not os.path.exists(zf):
            with requests.get(u,stream=True,timeout=600) as r:
                with open(zf,'wb') as f:
                    for c in r.iter_content(1<<22): f.write(c)
        z=zipfile.ZipFile(zf); tif=[x for x in z.namelist() if x.endswith('.tif')][0]; z.extract(tif,'data/ghsl'); os.remove(zf)
        tifp=f'data/ghsl/{tif}'
    with rasterio.open(tifp) as src:
        H,W=src.shape; f=12; outa=np.zeros((H//f,W//f),'float32')
        for i in range(0,H//f*f,f*120):
            h=min(f*120,H//f*f-i)
            a=src.read(1,window=Window(0,i,W//f*f,h)).astype('float32'); a[a<0]=0
            outa[i//f:(i+h)//f]=a.reshape(h//f,f,W//f,f).sum(axis=(1,3))
        np.save(out,outa); np.save('data/ghsl/transform.npy',np.array(src.transform)[:6])
        print(ep,'total pop',round(outa.sum()/1e9,3),'bn',flush=True)
    os.remove(tifp)
print('done')
