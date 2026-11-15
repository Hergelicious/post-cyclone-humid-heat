import sys; sys.path.insert(0,'pilot')
from style import *
import pandas as pd, numpy as np
rng=np.random.default_rng(4); GREY='#7d7c78'; C_MID=mpl.colors.to_hex(cm.cividis(0.45))
# ================= Fig 2
ann=pd.read_csv('pilot/robust_annual_lmic.csv',index_col=0).iloc[:,0]
D=pd.read_csv('pilot/decomp_metTH.csv').iloc[0]; Dm=pd.read_csv('pilot/decomp_met.csv'); cnt=Dm[(Dm.method=='count')&(Dm.excluded=='none')].iloc[0]; tonly=Dm[(Dm.method=='met')&(Dm.excluded=='none')].iloc[0]
fig,axs=plt.subplots(1,3,figsize=(183*MM,62*MM),gridspec_kw=dict(width_ratios=[1.35,1.1,0.9],wspace=0.75,left=0.08,right=0.955,bottom=0.2,top=0.9))
ax=axs[0]; ax.bar(ann.index,ann.values,0.8,color=C_LMIC,alpha=0.45,lw=0,label='Annual')
mm=ann.rolling(10).mean(); ax.plot(mm.index,mm.values,color=C_LMIC,lw=1.2,label='10-yr running mean')
for (a,b) in [(1981,1991),(1992,2002),(2003,2013),(2014,2024)]:
    v=ann.loc[a:b].mean(); ax.plot([a-0.4,b+0.4],[v,v],color=C_HIC,lw=1.2); ax.text((a+b)/2,v+60,f'{v:.0f}',ha='center',fontsize=5.5)
ax.plot([],[],color=C_HIC,lw=1.2,label='11-yr block mean')
ax.set_xlim(1980,2025); ax.set_ylabel('LMIC person-days with HI ≥ 40.6 °C\nwithin 100 km (million yr$^{-1}$)'); ax.legend(frameon=False,loc='upper left',fontsize=5.5); panel(ax,'a',x=-0.3)
ax=axs[1]; early=D.early if 'early' in D else cnt.early
early=cnt.early; late=cnt.late
steps=[('1981–\n2000',early,None,None,None,'t'),('Climate',D.clim_abs,D.clim_abs_lo,D.clim_abs_hi,D.climate,'i'),('Popu-\nlation',D.pop_abs,D.pop_abs_lo,D.pop_abs_hi,D.population,'i'),('Storm\nset',D.storm_abs,D.storm_abs_lo,D.storm_abs_hi,D.storms,'i'),('2018–\n2024',late,None,None,None,'t')]
cum=0
for i,(lab,v,lo,hi,sh,k) in enumerate(steps):
    if k=='t': ax.bar(i,v,0.6,color=GREY,lw=0); ax.text(i,v+15,f'{v:.0f}',ha='center',fontsize=5.5); cum=v if i==0 else cum
    else:
        ax.bar(i,v,0.6,bottom=cum,color=C_LMIC,lw=0); ax.errorbar(i,cum+v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor='#333333',elinewidth=0.6,capsize=1.5)
        ax.text(i+0.35,cum+v/2,f'{sh*100:.0f}%',fontsize=5.5,va='center'); cum+=v
ax.set_xticks(range(5)); ax.set_xticklabels([s[0] for s in steps],fontsize=5.5); ax.set_ylim(0,1000); ax.set_xlim(-0.5,4.8)
ax.set_ylabel('LMIC person-days with HI ≥ 40.6 °C\nwithin 100 km (million yr$^{-1}$)'); panel(ax,'b',x=-0.3)
ax=axs[2]
LO=pd.read_csv('pilot/robust_loo_exposure.csv').set_index('excluded'); CA=pd.read_csv('pilot/corridor_annual.csv',index_col=0)
def yb(a,late=(2018,2024),early=(1981,2000)):
    r=[a.loc[rng.choice(range(late[0],late[1]+1),late[1]-late[0]+1)].mean()/a.loc[rng.choice(range(early[0],early[1]+1),early[1]-early[0]+1)].mean() for _ in range(2000)]
    return a.loc[late[0]:late[1]].mean()/a.loc[early[0]:early[1]].mean(),*np.percentile(r,[2.5,97.5])
rows=[('100 km, 2018–24 vs 1981–2000',*yb(ann)),('100 km, 2014–24 vs 1981–91',*yb(ann,(2014,2024),(1981,1991))),('Track corridor',*yb(CA.corridor))]
for c,n in [('CN','China'),('IN','India'),('VN','Viet Nam'),('PH','Philippines'),('BD','Bangladesh')]: rows.append((f'Without {n}',LO.loc[c,'ratio'],np.nan,np.nan))
yy=np.arange(len(rows))[::-1]
for y,(lab,v,lo,hi) in zip(yy,rows):
    ax.plot(v,y,'o',color=C_LMIC if np.isfinite(lo) else C_MID,ms=3.5,mec='white',mew=0.4)
    if np.isfinite(lo): ax.plot([lo,min(hi,14)],[y,y],color=C_LMIC,lw=0.8)
    ax.text(15,y,f'{v:.1f}×',va='center',fontsize=5.5,clip_on=False)
ax.axvline(1,color='#999999',lw=0.5,ls=':'); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows],fontsize=5.5); ax.set_xscale('log'); ax.set_xlim(0.8,14); ax.set_xticks([1,2,4,8]); ax.set_xticklabels(['1','2','4','8'])
ax.set_xlabel('LMIC exposure ratio,\nlate to early period'); ax.tick_params(axis='y',length=0); panel(ax,'c',x=-1.45)
fig.savefig('figs/fig2_v4.png'); fig.savefig('figs/fig2_v4.pdf'); plt.close(fig)
# ================= Fig 3 outages
H=pd.read_csv('pilot/outage_heat.csv',parse_dates=['time']); S=pd.read_csv('pilot/hdntl_series2.csv'); M=pd.read_csv('pilot/hdntl_outage2.csv',parse_dates=['time'])
s2=open('pilot/power_more.py').read(); exec(s2[s2.index('def hi_c'):s2.index('cache={}')])
def hi_series(la,lo):
    s=pd.read_csv(f'pilot/power/{la}_{lo}.csv').replace(-999,np.nan); q=s.QV2M.values/1000; p=s.PS.values*10; e=q*p/(0.622+0.378*q); es=6.112*np.exp(17.67*s.T2M_MAX.values/(s.T2M_MAX.values+243.5))
    return pd.Series(hi_c(s.T2M_MAX.values,np.clip(100*e/es,1,100)),index=pd.date_range('1981-01-01',periods=len(s)))
fig=plt.figure(figsize=(183*MM,95*MM)); gs=fig.add_gridspec(2,3,width_ratios=[1,1,1.25],wspace=0.42,hspace=0.6,left=0.07,right=0.84,bottom=0.1,top=0.93)
cases=[('Bhubaneswar','2019-05-03','Fani, Bhubaneswar 2019'),('Calcutta','2020-05-20','Amphan, Kolkata 2020'),('Houston','2024-07-08','Beryl, Houston 2024'),('NewOrleans','2021-08-29','Ida, New Orleans 2021')]
for k,(city,d,title) in enumerate(cases):
    ax=fig.add_subplot(gs[k//2,k%2]); r=M[(M.city==city)&(M.time.dt.strftime('%Y-%m-%d')==d)].iloc[0]; s=S[S.i==r.name if False else S.i==M.index[(M.city==city)&(M.time.dt.strftime('%Y-%m-%d')==d)][0]]
    hi=hi_series(np.round(r.clat*2)/2,np.round(r.clon*2)/2); t=r.time.normalize(); w=hi[t:t+pd.Timedelta(days=40)]; days=(w.index-t).days
    ax.bar(days,(w>=35).astype(float)*0.08,0.9,bottom=-0.1,color=C_HIC,lw=0); ax.bar(days,(w>=40.6).astype(float)*0.08,0.9,bottom=-0.1,color=C_MID,lw=0)
    s=s[s.day<=40]; ax.plot(s.day,s.anom,'o-',color=C_LMIC,ms=2.5,lw=0.7,mec='white',mew=0.3)
    ax.axhline(1,color='#999999',lw=0.5,ls=':'); ax.set_ylim(-0.12,1.5); ax.set_xlim(-1,40); ax.set_title(title,fontsize=6.5,loc='left')
    if k%2==0: ax.set_ylabel('Night-light ratio\n(vs seasonal expectation)')
    if k//2==1: ax.set_xlabel('Days after landfall')
    panel(ax,'abcd'[k],x=-0.28 if k%2==0 else -0.15,y=1.08)
ax=fig.add_subplot(gs[:,2]); ok=H.copy()
for lm,col,lab in [(True,C_LMIC,'LMIC cities'),(False,C_HIC,'High-income cities')]:
    q=ok[ok.lmic==lm]; ax.scatter(q.min7,q.hi35_14+rng.uniform(-0.25,0.25,len(q)),s=7,color=col,alpha=0.75,lw=0,label=lab)
ax.axvline(0.6,color='#333333',lw=0.6,ls='--'); ax.set_xlabel('Lowest night-light ratio in first week'); ax.set_ylabel('Days with HI ≥ 35 °C in 14 days after landfall'); ax.set_xlim(0.1,1.6)
import matplotlib.patches as mp
h=[mp.Patch(color=C_LMIC,label='LMIC city'),mp.Patch(color=C_HIC,label='High-income city'),mpl.lines.Line2D([],[],color='#333333',ls='--',lw=0.6,label='Outage threshold'),mp.Patch(color=C_HIC,label='HI 35–40.6 °C (a–d)'),mp.Patch(color=C_MID,label='HI ≥ 40.6 °C (a–d)')]
ax.legend(handles=h,frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=5.5)
panel(ax,'e',x=-0.2,y=1.02)
fig.savefig('figs/fig3_v4.png'); fig.savefig('figs/fig3_v4.pdf'); plt.close(fig)
# ================= Fig 4 projections + attribution
P=pd.read_csv('pilot/cmip6_proj.csv'); Cb=pd.read_csv('pilot/cmip6_combined.csv',index_col=0); PJ=pd.read_csv('pilot/recon_projections.csv',index_col=0)
po=pd.read_pickle('pilot/obs_peryear_2005.pkl'); ob=[po.loc[rng.choice(range(2005,2025),20)].mean() for _ in range(2000)]
fig,axs=plt.subplots(1,2,figsize=(183*MM,66*MM),gridspec_kw=dict(width_ratios=[1.35,1],wspace=0.45,left=0.1,right=0.82,bottom=0.17,top=0.92))
ax=axs[0]; xs=np.arange(4); labs=['Observed\n2005–2024','1.5 °C','2.0 °C','2.8 °C']
ax.plot([0,0],np.percentile(ob,[5,95]),color=GREY,lw=3,solid_capstyle='butt'); ax.plot(0,po.mean(),'o',color='black',ms=3.5)
for j,wl in enumerate([1.5,2.0,2.8]):
    x=j+1; v=P[P.wl==wl].lmic.values
    ax.plot([x-0.12]*2,[Cb.loc['p5',str(wl)],Cb.loc['p95',str(wl)]],color=C_LMIC,lw=3,alpha=0.35,solid_capstyle='butt')
    ax.scatter(x+0.08+rng.uniform(-0.05,0.05,len(v)),v,s=5,color=C_LMIC,lw=0,alpha=0.8)
    ax.plot(x-0.12,np.median(v),'o',color=C_LMIC,ms=3.5,mec='white',mew=0.4)
    ax.plot(x+0.3,PJ.loc[f'{wl}_c','LMIC'],'s',color=C_HIC,ms=3,mec='black',mew=0.3); ax.plot(x+0.3,PJ.loc[f'{wl}_d','LMIC'],'v',color=C_HIC,ms=3,mec='black',mew=0.3)
ax.plot([],[],'s',color=C_HIC,ms=3,mec='black',mew=0.3,label='Observed local trend, RH fixed'); ax.plot([],[],'v',color=C_HIC,ms=3,mec='black',mew=0.3,label='Observed local trend, RH −1 pt K$^{-1}$')
ax.scatter([],[],s=5,color=C_LMIC,label='CMIP6 models (n = 26)'); ax.plot([],[],color=C_LMIC,lw=3,alpha=0.35,label='Models × years, 5–95%')
ax.set_xticks(xs); ax.set_xticklabels(labs); ax.set_xlim(-0.4,3.5); ax.set_ylim(0,1800); ax.set_ylabel('LMIC person-days with HI ≥ 40.6 °C\nwithin 100 km (million yr$^{-1}$)'); ax.legend(frameon=False,loc='upper left',fontsize=5.3); panel(ax,'a',x=-0.17)
ax=axs[1]; AT=pd.read_csv('pilot/attribution.csv'); A=AT[AT['var']=='T+RH constrained'].reset_index(drop=True)
yy=np.arange(len(A))[::-1]
ax.scatter(A.frac_attr*100,yy+0.12,s=14,color=C_LMIC,lw=0,label='Person-days ≥ 40.6 °C')
ax.scatter(A.d35_frac*100,yy-0.12,s=14,color=C_HIC,marker='D',lw=0,label='Days ≥ 35 °C per landfall')
ax.axvline(0,color='#999999',lw=0.5,ls=':'); ax.set_yticks(yy); ax.set_yticklabels(A.model,fontsize=6); ax.set_xlim(-20,80); ax.tick_params(axis='y',length=0)
ax.set_xlabel('Share of 2005–2024 LMIC post-landfall heat\nattributable to anthropogenic forcing (%)'); ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=5.5); panel(ax,'b',x=-0.5)
fig.savefig('figs/fig4_v4.png'); fig.savefig('figs/fig4_v4.pdf'); plt.close(fig); print('ok')
