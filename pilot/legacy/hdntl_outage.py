import rasterio, numpy as np, pandas as pd, glob, os, json
m=pd.read_csv('pilot/hdntl_matches.csv',parse_dates=['time']); m=m[m.dist<=100].reset_index(drop=True)
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
cache={}
def stack(serial,years):
    a=[];d=[]
    for y in years:
        for f in glob.glob(f'data/hdntl/tif/{serial:03d}_*_{y}.tif'):
            k=f
            if k not in cache:
                s=rasterio.open(f); cache[k]=(s.read().astype('float32'),pd.to_datetime([x[3:] for x in s.descriptions],format='%Y_%m_%d'))
            a.append(cache[k][0]); d.append(cache[k][1])
    if not a: return None,None
    return np.concatenate(a),d[0].append(d[1:]) if len(d)>1 else d[0]
rows=[];series=[]
for r in m.itertuples():
    if len(cache)>40: cache.clear()
    A,D=stack(r.serial,[r.time.year-1,r.time.year,r.time.year+1])
    if A is None: continue
    t=r.time.normalize(); dd=np.array((D-t).days)
    bsel=(dd>=-45)&(dd<=-2)
    if bsel.sum()<5: rows.append(dict(i=r.Index,status='no_baseline')); continue
    base=np.nanmedian(A[bsel],0); lit=base>=np.nanpercentile(base,50); lit&=base>1
    def ratio(k):
        v=A[k][lit]; b=base[lit]; ok=np.isfinite(v)&np.isfinite(b)
        return np.nan if ok.mean()<0.5 else np.nansum(v[ok])/np.nansum(b[ok])
    rb=np.array([ratio(k) for k in np.where(bsel)[0]]); rb=rb[np.isfinite(rb)]
    sd=np.std(np.log(rb)) if len(rb)>2 else np.nan
    post=np.where((dd>=0)&(dd<=60))[0]; rp=[(dd[k],ratio(k)) for k in post]; rp=[(x,y) for x,y in rp if np.isfinite(y)]
    for x,y in rp: series.append((r.Index,x,y))
    first=[(x,y) for x,y in rp if x<=7]
    if not first: rows.append(dict(i=r.Index,status='no_post',sd=sd)); continue
    thr=min(0.75,np.exp(-2.5*sd)) if np.isfinite(sd) else 0.75
    mn=min(y for x,y in first); out=mn<thr
    rec_day=np.nan; last_dark=np.nan
    if out:
        dark=[x for x,y in rp if y<thr]; last_dark=max(x for x in dark if x<=min([x for x,y in rp if y>=0.9 and x>min(dark)],default=99))
        after=[x for x,y in rp if y>=0.9 and x>last_dark]; rec_day=min(after) if after else np.nan
    rows.append(dict(i=r.Index,status='ok',sd=sd,thr=thr,min7=mn,outage=out,last_dark=last_dark,rec_day=rec_day,nb=len(rb),np7=len(first)))
O=pd.DataFrame(rows).set_index('i'); M=m.join(O); M['lmic']=M.iso2.map(inc).isin(['LIC','LMC','UMC'])
M['dur_lo']=M.last_dark+1; M['dur_hi']=M.rec_day; M['dur_mid']=np.where(M.rec_day.notna(),(M.last_dark+M.rec_day)/2,np.nan)
M.to_csv('pilot/hdntl_outage.csv',index=False); pd.DataFrame(series,columns=['i','day','ratio']).to_csv('pilot/hdntl_series.csv',index=False)
print(M.status.value_counts()); ok=M[M.status=='ok']; print('outage share',ok.outage.mean().round(2),'n',len(ok)); print(ok.groupby('lmic').outage.agg(['mean','size']))
print(ok[ok.outage][['dur_lo','dur_hi','dur_mid']].describe().round(1))
print(ok[ok.outage].sort_values('min7')[['city','time','lf_wind','dist','min7','last_dark','rec_day']].head(25).to_string())
