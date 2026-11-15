import pandas as pd, numpy as np
s=open('pilot/cmip6_proj.py').read(); exec(s[:s.index('# ---- per model pattern scaling')])
iso=np.repeat(np.asarray(A.iso2.astype(str))[:,None],30,1).ravel()
def uniq2(mask):
    v=np.where(mask.ravel(),popw*rp406,0.0); best=np.zeros(len(ukey)); np.maximum.at(best,inv,v)
    first=np.zeros(len(ukey),int); first[inv[::-1]]=np.arange(len(inv))[::-1]
    return pd.DataFrame(dict(v=best,lmic=lm[first],iso=iso[first]))
grp=lambda df: df[df.lmic].assign(c=lambda x: np.where(x.iso.isin(['CN','IN','VN','PH','BD']),x.iso,'other')).groupby('c').v.sum()/20/1e6
obsC=grp(uniq2(HI>=40.6)); print('obs by country',obsC.round(1).to_dict())
# observed slopes per cell-month (POWER, 1981-2024) for comparison
obsb={}
def obs_beta(cidx_,m):
    k=(cidx_,m)
    if k not in obsb:
        la,lo=cells[cidx_]; s_=series(round(la,1),round(lo,1)); sel=(mon==m)&(yr<=2024); Y=pd.Index(yr[sel]).unique(); x=gpi.loc[Y].values
        obsb[k]=(np.polyfit(x,pd.Series(s_['T'][sel]).groupby(yr[sel]).mean().values,1)[0],np.polyfit(x,pd.Series(s_['RH'][sel]).groupby(yr[sel]).mean().values,1)[0])
    return obsb[k]
L=A.lmic.values; keyset=sorted(set(zip(ci[L],mo[L],np.asarray(A.iso2.astype(str))[L])))
OB=pd.DataFrame([(c,m,i,*obs_beta(c,m)) for c,m,i in keyset],columns=['c','m','iso','bT','bH'])
rows=[];proj=[]
for mname in models:
    h=load(mname,'historical'); f=load(mname,'ssp585')
    if h is None or f is None: continue
    yr_=np.concatenate([h['yr'],f['yr']]); mo_=np.concatenate([h['mo'],f['mo']]); gm=np.concatenate([h['gmst'],f['gmst']]); tx=np.concatenate([h['tasmax'],f['tasmax']]); hu=np.concatenate([h['hurs'],f['hurs']])
    gy=pd.Series(gm).groupby(yr_).mean().rolling(11,center=True,min_periods=5).mean(); ganom=gy-gy.loc[1995:2014].mean()+OBS_REF
    def slopes(yrmask):
        bT=np.zeros((12,len(cells)),'float32'); bH=np.zeros((12,len(cells)),'float32')
        for k in range(1,13):
            sel=(mo_==k)&yrmask; x=ganom.loc[yr_[sel]].values; xc=x-x.mean()
            bT[k-1]=(xc@(tx[sel]-tx[sel].mean(0)))/(xc@xc); bH[k-1]=(xc@(hu[sel]-hu[sel].mean(0)))/(xc@xc)
        return bT,bH
    full=slopes(np.ones(len(yr_),bool)); hist=slopes((yr_>=1981)&(yr_<=2024)); narrow=slopes(ganom.loc[yr_].values<=3.0)
    for c,m,i in keyset: rows.append((mname,i,full[0][m-1,c],full[1][m-1,c],hist[0][m-1,c],hist[1][m-1,c]))
    for lab,(bT,bH) in [('full',full),('narrow',narrow)]:
        for wl in [1.5,2.0,2.8]:
            dg=(wl-g_obs)[:,None]; hi=hi_c(T+bT[mo-1,ci][:,None]*dg,np.clip(RH+bH[mo-1,ci][:,None]*dg,1,100))
            g_=grp(uniq2(hi>=40.6)); proj.append(dict(model=mname,slope=lab,wl=wl,total=g_.sum(),**g_.to_dict()))
    print(mname,flush=True)
E=pd.DataFrame(rows,columns=['model','iso','bT_full','bH_full','bT_hist','bH_hist'])
ev=E.groupby('iso')[['bT_full','bH_full','bT_hist','bH_hist']].median().join(OB.groupby('iso')[['bT','bH']].median().rename(columns={'bT':'bT_obs','bH':'bH_obs'}))
ev['n_models_hist_T_below_obs']=E.groupby(['iso','model']).bT_hist.median().unstack().lt(OB.groupby('iso').bT.median(),axis=0).sum(1)
print(ev.loc[['CN','IN','VN','PH','BD']].round(2)); ev.to_csv('pilot/cmip6_eval_slopes.csv')
P=pd.DataFrame(proj); P.to_csv('pilot/cmip6_proj_country.csv',index=False)
print(P.groupby(['slope','wl'])[['total','CN','IN','VN','PH','BD','other']].median().round(0)); print('obs',obsC.round(0).to_dict())
