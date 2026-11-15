import rasterio, numpy as np, pandas as pd, glob, json, warnings
warnings.filterwarnings('ignore')
m=pd.read_csv('pilot/hdntl_matches.csv',parse_dates=['time']); m=m[m.dist<=100].reset_index(drop=True)
allm=pd.read_csv('pilot/hdntl_matches.csv',parse_dates=['time'])
wb=json.load(open('pilot/wb_countries.json'))[1]; inc={c['iso2Code']:c['incomeLevel']['id'] for c in wb}
def city_stack(serial):
    fs=sorted(glob.glob(f'data/hdntl/tif/{serial:03d}_*_20??.tif')); med=[]
    for f in fs:
        with rasterio.open(f) as s: med.append(np.nanmedian(s.read(),0))
    mm=np.nanmedian(np.stack(med),0); cand=np.flatnonzero((mm>=np.nanpercentile(mm,50))&(mm>1))
    rng=np.random.default_rng(serial); pix=np.sort(rng.choice(cand,min(len(cand),20000),replace=False))
    A=[];D=[]
    for f in fs:
        with rasterio.open(f) as s: a=s.read().reshape(s.count,-1)[:,pix]; A.append(a[:,:,None].astype('float32')); D.append(pd.to_datetime([x[3:] for x in s.descriptions],format='%Y_%m_%d'))
    return np.concatenate(A),D[0].append(D[1:])
def curve(A,D,t,lit_from=None):
    dd=np.array((D-t).days); bsel=(dd>=-45)&(dd<=-2)
    if bsel.sum()<5: return None
    base=np.nanmedian(A[bsel],0); lit=np.isfinite(base)&(base>0)
    out=[]
    for k in np.where((dd>=0)&(dd<=60))[0]:
        v=A[k][lit]; b=base[lit]; ok=np.isfinite(v)&np.isfinite(b)
        if ok.mean()>=0.5: out.append((dd[k],np.nansum(v[ok])/np.nansum(b[ok])))
    return out
rows=[];ser=[];CT=[]
for serial,g in m.groupby('serial'):
    A,D=city_stack(int(serial)); hits=allm[allm.serial==serial].time
    for r in g.itertuples():
        ev=curve(A,D,r.time.normalize())
        if ev is None: rows.append(dict(i=r.Index,status='no_baseline')); continue
        ctrl=[]
        for y in range(2012,2025):
            if y==r.time.year: continue
            try: tc=r.time.normalize().replace(year=y)
            except ValueError: continue
            if ((hits-tc).abs()<pd.Timedelta(days=75)).any(): continue
            c=curve(A,D,tc)
            if c: ctrl.append(c)
        if len(ctrl)<4: rows.append(dict(i=r.Index,status='few_controls')); continue
        cdf=pd.DataFrame([(j,d,v) for j,c in enumerate(ctrl) for d,v in c],columns=['y','day','r'])
        # control expected ratio by day (rolling +-3 d)
        exp=pd.Series({d:cdf[(cdf.day>=d-3)&(cdf.day<=d+3)].r.median() for d in range(0,61)})
        cmin7=cdf[cdf.day<=7].groupby('y').r.min(); cmin7_anom=[cdf[(cdf.y==j)&(cdf.day<=7)].apply(lambda q:q.r/exp[q.day],axis=1).min() for j in cmin7.index]
        thr=min(0.75,np.nanpercentile(cmin7_anom,5)) if len(cmin7_anom)>=4 else 0.75
        an=[(d,v/exp[d]) for d,v in ev if np.isfinite(exp[d])]
        for d,v in an: ser.append((r.Index,d,v))
        first=[v for d,v in an if d<=7]
        if not first: rows.append(dict(i=r.Index,status='no_post',ncontrol=len(ctrl))); continue
        mn=min(first); out=mn<thr; ld=rd=np.nan
        if out:
            dark=[d for d,v in an if v<thr]; d0=min(dark)
            rec=[d for d,v in an if v>=0.9 and d>d0]
            # last dark before first sustained recovery
            rd=min(rec) if rec else np.nan; ld=max([d for d in dark if (np.isnan(rd) or d<rd)])
        rows.append(dict(i=r.Index,status='ok',ncontrol=len(ctrl),thr=thr,min7=mn,outage=out,last_dark=ld,rec_day=rd,ctrl_fp=np.mean(np.array(cmin7_anom)<thr),nimg7=len(first)))
        for v in cmin7_anom: CT.append((r.Index,v))
        # image availability days 0-60
        for d,v in an: pass
O=pd.DataFrame(rows).set_index('i'); M=m.join(O); M['lmic']=M.iso2.map(inc).isin(['LIC','LMC','UMC'])
M['dur_lo']=M.last_dark+1; M['dur_hi']=M.rec_day
M.to_csv('pilot/hdntl_outage2.csv',index=False); pd.DataFrame(CT,columns=['i','ctrl_min7']).to_csv('pilot/hdntl_controls.csv',index=False); pd.DataFrame(ser,columns=['i','day','anom']).to_csv('pilot/hdntl_series2.csv',index=False)
print(M.status.value_counts()); ok=M[M.status=='ok']; print('outage share',round(ok.outage.mean(),3),'n',len(ok),' control false-positive rate',round(ok.ctrl_fp.mean(),3))
print(ok.groupby('lmic').outage.agg(['mean','sum','size']))
print(ok[ok.outage].sort_values('min7')[['city','iso2','time','lf_wind','dist','thr','min7','last_dark','rec_day']].to_string())
