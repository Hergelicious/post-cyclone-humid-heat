import urllib.request, numpy as np, time, eccodes, io
B='https://storage.googleapis.com/high-resolution-rapid-refresh'
def field(date,cyc,fh,var):
    base=f'{B}/hrrr.{date}/conus/hrrr.t{cyc:02d}z.wrfsfcf{fh:02d}.grib2'
    idx=urllib.request.urlopen(base+'.idx',timeout=60).read().decode().splitlines()
    for i,l in enumerate(idx):
        p=l.split(':')
        if p[3]==var and p[4]=='2 m above ground':
            a=int(p[1]); b=int(idx[i+1].split(':')[1])-1
            req=urllib.request.Request(base,headers={'Range':f'bytes={a}-{b}'}); data=urllib.request.urlopen(req,timeout=120).read()
            g=eccodes.codes_new_from_message(data); v=eccodes.codes_get_values(g); lat=eccodes.codes_get_array(g,'latitudes'); lon=eccodes.codes_get_array(g,'longitudes')
            ny=eccodes.codes_get(g,'Ny'); nx=eccodes.codes_get(g,'Nx'); eccodes.codes_release(g); return v.reshape(ny,nx),lat.reshape(ny,nx),lon.reshape(ny,nx),len(data)
t=time.time(); T,la,lo,n=field('20240708',0,24,'TMP'); print('TMP f24',T.shape,round(T.mean()-273.15,1),'C',n,'bytes',round(time.time()-t,1),'s')
t=time.time(); D,_,_,n=field('20240708',0,48,'DPT'); print('DPT f48',round(D.mean()-273.15,1),n,'bytes',round(time.time()-t,1),'s')
# Houston point
k=np.unravel_index(np.argmin((la-29.76)**2+((lo%360)-(360-95.37))**2),la.shape); print('Houston T f24',round(T[k]-273.15,1))
np.save('data/hrrr/latlon.npy',np.stack([la,lo]))
