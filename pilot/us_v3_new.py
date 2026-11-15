# New outage analyses on the version-2 records: (1) bounds for customers without power through k hazardous daily peaks,
# (2) restoration performance against the reductions required by warming, (3) time from outage onset to the first hazardous
# overlap, observed and in other climates, (4) exposure by remaining outage share and a 72 h restoration cap,
# (5) intensity-weighted exposure (customer-degree-hours above HI 35 C) with attribution and warming-level ratios.
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl')
S=np.load('data/cmip6cty/slopes26.npz'); A=np.load('data/cmip6cty/humandelta15.npz')
NAT=list(A['keys']); MOD=list(S['models']); WLs=[1.5,2.0,2.8]
def cf(r,kind,k,wl=None):
    """heat index in another climate: kind 'nat' (realisation k) or 'wl' (model k at warming level wl)"""
    if kind=='nat': dT=-A['dT'][k]*r['fac']; dH=-A['dH'][k]*r['fac']
    else: dT=S['bT'][k]*(wl-r['gobs']); dH=S['bH'][k]*(wl-r['gobs'])
    return hi_c(r['T']+dT[r['mon']][:,r['ci']],np.clip(r['RH']+dH[r['mon']][:,r['ci']],1,100))
out={}
# ---------- (1) bounds through k peaks
cols=['up1','up2','up3','up4','up5','c2','c3','warmnight','lo1','lo2','lo3','lo4','lo5','fr1','fr2','fr3','fr4','fr5','c2lo','c3lo']
def sevsum(key): return pd.Series(sum(r['cty'][key]['sev'].sum(0) for r in recs),index=cols)
SO=sevsum('obs'); SN=pd.DataFrame({k:sevsum(k) for k in [c for c in recs[0]['cty'] if c.startswith('nat|')]})
share=(1-SN.div(SO,axis=0)).T; mm=share.groupby([c.split('|')[1] for c in share.index]).mean()
B=pd.DataFrame(dict(observed_million=SO/1e6,human_share=share.mean(),p05=share.quantile(0.05),p95=share.quantile(0.95),model_min=mm.min(),model_max=mm.max()))
B.to_csv('pilot/us_v3_bounds.csv'); print('(1) customers without power through k hazardous peaks, bounds (million) and human share'); print(B.round(3).to_string())
# ---------- per-county event table for restoration, lead time and tail analyses
ev=[]; tail=np.zeros(4); cap={}; inten={'obs':0.0}; inten_nat=np.zeros(len(NAT)); inten_wl=np.zeros((len(MOD),3)); h35_obs=0.0
cap_obs=0.0; cap_wl=np.zeros((len(MOD),3)); h35_wl=np.zeros((len(MOD),3)); comp_wl=np.zeros((len(MOD),3))
lead_cl={'obs':[]}
for r in recs:
    Ex=r['Ex'].astype('float64'); nh,nc=Ex.shape; win=r['win'][:,None]; Exw=Ex*win; HI=r['HI']; on=r['onset']
    el=np.arange(nh)[:,None]-on[None]
    pk=Ex.max(0); ipk=Ex.argmax(0)
    F=(HI>=35)
    # tail: remaining share of the county's peak
    rem=np.where(pk>0,Ex/np.maximum(pk,1),0)
    for b,(lo_,hi_) in enumerate([(0.5,1.01),(0.2,0.5),(0.1,0.2),(0,0.1)]):
        m=(rem>=lo_)&(rem<hi_)&(Ex>0); tail[b]+=(Exw*F*m).sum()
    capm=(el<72)
    h35_obs+=(Exw*F).sum(); cap_obs+=(Exw*F*capm).sum()
    inten['obs']+=(Exw*np.clip(HI-35,0,None)).sum()
    for k in range(len(NAT)):
        h=cf(r,'nat',k); inten_nat[k]+=(Exw*np.clip(h-35,0,None)).sum()
    leads_wl={}
    for k in range(len(MOD)):
        for a,wl in enumerate(WLs):
            h=cf(r,'wl',k,wl); Fw=h>=35
            h35_wl[k,a]+=(Exw*Fw).sum(); cap_wl[k,a]+=(Exw*Fw*capm).sum(); inten_wl[k,a]+=(Exw*np.clip(h-35,0,None)).sum()
            # first hazardous hour with customers still out, after onset
            fo=np.where(((Fw)&(Ex>0)&(el>=0)&win).any(0),np.argmax((Fw)&(Ex>0)&(el>=0)&win,0)-on,-1)
            leads_wl.setdefault(wl,[]).append(fo)
    fo=np.where((F&(Ex>0)&(el>=0)&win).any(0),np.argmax(F&(Ex>0)&(el>=0)&win,0)-on,-1)
    # restoration milestones after the peak: hours from onset to the first hour at or below 50% and 10% of the peak
    for j in range(nc):
        x=Ex[:,j]; p=ipk[j]
        def first_below(fr):
            k=np.where(x[p:]<=fr*pk[j])[0]; return (p+k[0]-on[j]) if len(k) else np.nan
        ev.append(dict(storm=r['storm'],season=r['season'],fips=r['fips'][j],cust=r['cust'][j],peak=pk[j],peak_frac=pk[j]/r['cust'][j],
            t50=first_below(0.5),t90=first_below(0.1),cust_h=float(Exw[:,j].sum()),hour35=float((Exw[:,j]*F[:,j]).sum()),
            lead=fo[j],out_at_onset=float(Ex[on[j],j]),**{f'lead_wl{wl}':np.median([L[j] for L in leads_wl[wl][j::1][:0]] or [np.nan]) for wl in []}))
    for wl in WLs:
        Lw=np.array(leads_wl[wl])  # models x counties
        for j in range(nc): ev[-nc+j][f'lead_wl{wl}']=np.median(np.where(Lw[:,j]>=0,Lw[:,j],np.nan)) if (Lw[:,j]>=0).mean()>=0.5 else -1
EV=pd.DataFrame(ev); EV.to_csv('pilot/us_v3_events.csv',index=False)
# ---------- (2) restoration performance
EV['sev']=pd.cut(EV.peak_frac,[0,0.2,0.5,0.8,1.01],labels=['5-20%','20-50%','50-80%','>80%'])
E2=EV[np.isfinite(EV.t50)&(EV.t50>0)].copy()
E2['rel50']=E2.t50/E2.groupby(['storm','sev']).t50.transform('median'); E2['n_str']=E2.groupby(['storm','sev']).t50.transform('size')
E3=E2[E2.n_str>=5]
E9=EV[np.isfinite(EV.t90)&(EV.t90>0)].copy(); E9['rel90']=E9.t90/E9.groupby(['storm','sev']).t90.transform('median'); E9['n_str']=E9.groupby(['storm','sev']).t90.transform('size'); E9=E9[E9.n_str>=5]
req={1.5:0.13,2.0:0.29,2.8:0.49}; req_pk={1.5:0.05,2.0:0.13,2.8:0.31}
print('\n(2) restoration: county-events in storm-by-severity strata with >=5 counties:',len(E3),'(t50),',len(E9),'(t90)')
rows=[]
for wl in WLs:
    for nm,rq in [('hourly',req[wl]),('daily peaks',req_pk[wl])]:
        thr=1-rq
        rows.append(dict(wl=wl,measure=nm,required_reduction=rq,share_events_t50=(E3.rel50<=thr).mean(),share_customers_t50=np.average(E3.rel50<=thr,weights=E3.peak),
                         share_events_t90=(E9.rel90<=thr).mean(),share_customers_t90=np.average(E9.rel90<=thr,weights=E9.peak),
                         share_exposure_t50=np.average(E3.rel50<=thr,weights=E3.hour35+1e-9)))
RG=pd.DataFrame(rows); RG.to_csv('pilot/us_v3_restoration_gap.csv',index=False); print(RG.round(2).to_string())
q=E3.groupby('sev').rel50.quantile([0.1,0.25,0.5]).unstack(); print('relative time to 50% restoration within storm and severity, quantiles:'); print(q.round(2).to_string())
print('overall quantiles rel50 10/25/50%:',E3.rel50.quantile([0.1,0.25,0.5]).round(2).tolist(),' rel90:',E9.rel90.quantile([0.1,0.25,0.5]).round(2).tolist())
print('median hours onset to 50% / 90% restoration by severity:'); print(EV.groupby('sev')[['t50','t90']].median().round(0).to_string())
# trend in restoration speed by year within severity strata
import statsmodels.formula.api as smf
E2=E2[E2.sev.notna()].copy(); E2['lt']=np.log(E2.t50); m=smf.ols('lt ~ season + C(sev)',data=E2).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(E2.storm)[0]})
print('trend in log time to 50%% restoration per year: %.3f (%.3f to %.3f), n=%d'%(m.params['season'],*m.conf_int().loc['season'],len(E2)))
out['trend']=(m.params['season'],*m.conf_int().loc['season'])
# ---------- (3) lead time from onset to first hazardous overlap
def lead_tab(col):
    L=EV[col].values; w=EV.out_at_onset.values; ok=L>=0
    bins=[(-0.5,0.5,'at onset'),(0.5,6,'1-6 h'),(6,12,'6-12 h'),(12,24,'12-24 h'),(24,48,'24-48 h'),(48,1e9,'>48 h')]
    d={b[2]:w[ok&(L>b[0])&(L<=b[1])].sum()/w.sum() for b in bins}; d['no overlap']=w[~ok].sum()/w.sum()
    lw=L[ok]; ww=w[ok]; o=np.argsort(lw); c=np.cumsum(ww[o])/ww.sum(); d['median_h']=lw[o][np.searchsorted(c,0.5)]
    return d
LT=pd.DataFrame({'observed':lead_tab('lead'),**{f'{wl} C':lead_tab(f'lead_wl{wl}') for wl in WLs}}).T
LT.to_csv('pilot/us_v3_leadtime.csv'); print('\n(3) share of customers out at onset by time to first hazardous hour while still out (customer-weighted)'); print(LT.round(3).to_string())
# ---------- (4) tail and cap
print('\n(4) share of hour35 exposure by remaining share of county peak: >50%, 20-50%, 10-20%, <10%:',(tail/tail.sum()).round(3).tolist())
print('restoring all customers by 72 h after onset removes %.0f%% of observed hour35 exposure; at 1.5/2.0/2.8 C (median over models) exposure with the cap is %s of observed'%(100*(1-cap_obs/h35_obs),np.round(np.median(cap_wl,0)/h35_obs,2).tolist()))
print('uncapped warming ratios',np.round(np.median(h35_wl,0)/h35_obs,2).tolist())
# ---------- (5) intensity
hs=1-inten_nat/inten['obs']; mmn=pd.Series(hs).groupby([k.split('|')[0] for k in NAT]).mean()
print('\n(5) customer-degree-hours above HI 35 C: %.0f million; human share %.0f%% (5-95%% %.0f-%.0f; model means %.0f-%.0f); warming ratios %s'%(inten['obs']/1e6,100*hs.mean(),100*np.quantile(hs,0.05),100*np.quantile(hs,0.95),100*mmn.min(),100*mmn.max(),np.round(np.median(inten_wl,0)/inten['obs'],2).tolist()))
pd.to_pickle(dict(hs=hs,inten_wl=inten_wl,inten_obs=inten['obs'],tail=tail,cap_obs=cap_obs,cap_wl=cap_wl,h35_wl=h35_wl,h35_obs=h35_obs,trend=out['trend']),'pilot/us_v3_misc.pkl')
