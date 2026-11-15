import sys; sys.path.insert(0,'pilot')
from style import *
import pandas as pd, numpy as np
rng=np.random.default_rng(11)
per=['1981-2000','2001-2017','2018-2024']; ny={'1981-2000':20,'2001-2017':17,'2018-2024':7}
U=pd.read_csv('pilot/spatial_unique.csv'); O=pd.read_csv('pilot/decomposition.csv',index_col=0)
# ---------- Fig 2
fig,axs=plt.subplots(1,2,figsize=(183*MM,62*MM),gridspec_kw=dict(wspace=0.5,left=0.08,right=0.86,bottom=0.16,top=0.92))
ax=axs[0]; x=np.arange(3); foots=[('50','50 km',0.05),('100','100 km',0.3),('wind','Wind-scaled',0.55),('200','200 km',0.8)]
for k,(f,lab,v) in enumerate(foots):
    s=U[(U.foot.astype(str)==f)&(U.thr==406)&(U.lmic==True)].iloc[0]; vals=[s[p] for p in per]
    ax.bar(x+(k-1.5)*0.2,vals,0.19,color=mpl.colors.to_hex(cm.cividis(v)),edgecolor='white',linewidth=0.4,label=lab)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_yscale('log'); ax.set_ylim(20,4000); ax.set_yticks([30,100,300,1000,3000]); ax.set_yticklabels(['30','100','300','1,000','3,000'])
ax.set_ylabel('Unique person-days with HI ≥ 40.6 °C\nafter landfall in LMICs (million yr$^{-1}$)')
ax.legend(frameon=False,loc='upper left',fontsize=5.5,title='Footprint',title_fontsize=5.5); panel(ax,'a',x=-0.22)
ax=axs[1]; o=O.loc[406]
steps=[('1981–2000',o.model_early,'total'),('Climate',o.shapley_H,'inc'),('Population',o.shapley_P,'inc'),('Storm\nsampling',o.shapley_S,'inc'),('2018–2024',o.model_late,'total')]
cum=0; cols={'total':'#7d7c78','inc':C_LMIC}
for i,(lab,v,kind) in enumerate(steps):
    if kind=='total': ax.bar(i,v,0.6,color=cols['total'],edgecolor='white',linewidth=0.4); ax.text(i,v+15,f'{v:.0f}',ha='center',fontsize=5.5); cum=v if i==0 else cum
    else: ax.bar(i,v,0.6,bottom=cum,color=cols['inc'],edgecolor='white',linewidth=0.4); ax.text(i,cum+v+15,f'+{v:.0f}',ha='center',fontsize=5.5); cum+=v
ax.set_xticks(range(5)); ax.set_xticklabels([s[0] for s in steps],fontsize=6); ax.set_ylabel('Modelled person-days with HI ≥ 40.6 °C\nafter landfall in LMICs (million yr$^{-1}$)'); ax.set_ylim(0,720)
panel(ax,'b',x=-0.22)
fig.savefig('figs/fig2_exposure_v2.png'); fig.savefig('figs/fig2_exposure_v2.pdf'); plt.close(fig)
# ---------- Fig 4 with bootstrap bands
ev=pd.read_csv('pilot/power_events_exposure.csv'); a=ev[ev.season>=2005]
lv=['Observed\n2005–2024','1.5 °C','2.0 °C','2.8 °C']; k=['1.5','2.0','2.8']
fig,axs=plt.subplots(1,2,figsize=(183*MM,58*MM),gridspec_kw=dict(wspace=0.4,left=0.08,right=0.87,bottom=0.17,top=0.92))
for ax,(m,ttl) in zip(axs,[('d406','Days with HI ≥ 40.6 °C'),('tw28','Days with Tw ≥ 28 °C')]):
    for j,(lm,col,labx) in enumerate([(True,C_LMIC,'LMICs'),(False,C_HIC,'High income')]):
        s=a[a.lmic==lm]; xs=np.arange(4)+(j-0.5)*0.2
        cols_c=[m]+[f'p{q}_c_{m}' for q in k]; cols_d=[m]+[f'p{q}_d_{m}' for q in k]
        B=np.array([[s.sample(len(s),replace=True,random_state=int(rng.integers(1e9)))[[c,d]].mean().values for c,d in zip(cols_c,cols_d)] for _ in range(300)])  # 300 x 4 x 2
        lo=np.percentile(B.min(2),2.5,axis=0); hi=np.percentile(B.max(2),97.5,axis=0); mid=np.array([s[[c,d]].mean().mean() for c,d in zip(cols_c,cols_d)])
        ax.fill_between(xs,lo,hi,color=col,alpha=0.18,lw=0)
        ax.plot(xs,mid,'o-',color=col,ms=3.5,mec='white',mew=0.4,lw=0.8,label=labx)
        ax.text(xs[-1]+0.12,mid[-1],f'{mid[-1]:.1f}',fontsize=5.5,va='center')
    ax.set_xticks(range(4)); ax.set_xticklabels(lv); ax.set_ylabel(ttl+'\nin 30 days after landfall (mean per landfall)')
axs[1].legend(frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0)); panel(axs[0],'a',x=-0.2); panel(axs[1],'b',x=-0.2)
fig.savefig('figs/fig4_projections_v2.png'); fig.savefig('figs/fig4_projections_v2.pdf'); plt.close(fig)
print('ok')
