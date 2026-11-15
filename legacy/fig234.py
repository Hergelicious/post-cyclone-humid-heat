import sys; sys.path.insert(0,'pilot')
from style import *
import pandas as pd, numpy as np
ev=pd.read_csv('pilot/power_events_exposure.csv')
ev['wet']=ev.rain4>=100; ev['wethot']=ev.wet&(ev.hot21>=7)
per=['1981-2000','2001-2017','2018-2024']; ny={'1981-2000':20,'2001-2017':17,'2018-2024':7}
# ---------- Fig 2: exposure and water pathway
fig,axs=plt.subplots(1,2,figsize=(183*MM,58*MM),gridspec_kw=dict(wspace=0.6,left=0.08,right=0.87,bottom=0.13,top=0.92))
ax=axs[0]; x=np.arange(3); w=0.36
for k,(lm,col,lab) in enumerate([(True,C_LMIC,'LMICs'),(False,C_HIC,'High income')]):
    v=[ev[(ev.period==p)&(ev.lmic==lm)].pd_d406.sum()/ny[p]/1e6 for p in per]
    b=ax.bar(x+(k-0.5)*w,v,w*0.92,color=col,label=lab,edgecolor='white',linewidth=0.4)
    for xi,vi in zip(x+(k-0.5)*w,v): ax.text(xi,vi+12,f'{vi:.0f}',ha='center',fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylabel('Person-days with HI ≥ 40.6 °C\nwithin 30 days of landfall (million yr$^{-1}$)')
ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.0,1.0)); panel(ax,'a',x=-0.22)
ax=axs[1]
for k,(lm,col,lab) in enumerate([(True,C_LMIC,'LMICs'),(False,C_HIC,'High income')]):
    ps=[];lo=[];hi=[]
    for p in per:
        s=ev[(ev.period==p)&(ev.lmic==lm)]; a,b_,c=wilson(s.wethot.sum(),len(s)); ps.append(100*a);lo.append(100*b_);hi.append(100*c)
    xx=x+(k-0.5)*0.18
    ax.errorbar(xx,ps,yerr=[np.array(ps)-lo,np.array(hi)-ps],fmt='o',ms=4,color=col,mec='white',mew=0.4,elinewidth=0.6,capsize=0,label=lab)
    ax.plot(xx,ps,color=col,lw=0.8,alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_ylabel('Landfalls with storm rain\nfollowed by heat (%)'); ax.set_ylim(0,40)
ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1.0,1.0)); panel(ax,'b',x=-0.2)
fig.savefig('figs/fig2_exposure_water.png'); fig.savefig('figs/fig2_exposure_water.pdf'); plt.close(fig)
# ---------- Fig 3: recovery capacity
c=pd.read_csv('pilot/country_capacity.csv').dropna(subset=['deficit'])
c=c[(c.n_ind>=4)&(c.pd35_yr>=1e4)].copy()
fig,ax=plt.subplots(figsize=(120*MM,80*MM)); fig.subplots_adjust(left=0.13,right=0.76,bottom=0.14,top=0.97)
c['y']=np.log10(c.pd35_yr); sz=8+c.landfalls*2.2
sc=ax.scatter(c.deficit,c.y,s=sz,c=c.elec,cmap='cividis',vmin=30,vmax=100,edgecolor='white',linewidth=0.4,zorder=3)
pos={'China':(0.13,9.25),'India':(0.33,8.75),'Viet Nam':(0.02,8.85,'right'),'Japan':(0.02,8.25,'right'),'Philippines':(0.33,8.35),'Bangladesh':(0.33,7.95),
     'United States':(0.02,7.75,'right'),'Mexico':(0.02,7.3,'right'),'Myanmar':(0.36,6.95),'Mozambique':(0.62,7.45),'Somalia, Fed. Rep.':(0.86,6.95),'Madagascar':(0.8,6.0),'Haiti':(0.8,5.4),'Vanuatu':(0.5,4.75)}
short={'United States':'USA','Somalia, Fed. Rep.':'Somalia'}
for r in c.itertuples():
    if r.name in pos:
        p=pos[r.name]; ha=p[2] if len(p)>2 else 'left'
        ax.annotate(short.get(r.name,r.name),(r.deficit,r.y),xytext=(p[0] if ha=='left' else -0.02+p[0]*0, p[1]),textcoords='data' if ha=='left' else 'data',
                    fontsize=5.5,va='center',ha=ha,arrowprops=dict(arrowstyle='-',lw=0.35,color='#7d7c78',shrinkA=0,shrinkB=2))
ax.set_xlabel('Recovery capacity deficit index (0 = highest capacity)'); ax.set_ylabel('Person-days with HI ≥ 35 °C after landfall,\n2013–2024 (yr$^{-1}$)')
ax.set_yticks([4,5,6,7,8,9]); ax.set_yticklabels(['10$^4$','10$^5$','10$^6$','10$^7$','10$^8$','10$^9$']); ax.set_xlim(-0.22,1.0); ax.set_ylim(3.8,9.5)
cax=fig.add_axes([0.81,0.55,0.025,0.38]); cb=fig.colorbar(sc,cax=cax); cb.set_label('Electricity access (%)',fontsize=6); cb.outline.set_linewidth(0.4)
for i,n in enumerate([1,10,40]):
    fig.add_artist(plt.Line2D([0.825],[0.38-i*0.07],marker='o',ms=np.sqrt(8+n*2.2),color='#7d7c78',mec='white',transform=fig.transFigure,ls=''))
    fig.text(0.855,0.38-i*0.07,f'{n}',fontsize=6,va='center')
fig.text(0.81,0.45,'Landfalls',fontsize=6)
fig.savefig('figs/fig3_capacity.png'); fig.savefig('figs/fig3_capacity.pdf'); plt.close(fig)
# ---------- Fig 4: projections
a=ev[ev.season>=2005]
lv=['Observed\n2005–2024','1.5 °C','2.0 °C','2.8 °C']; k=['1.5','2.0','2.8']
fig,axs=plt.subplots(1,2,figsize=(183*MM,58*MM),gridspec_kw=dict(wspace=0.4,left=0.08,right=0.87,bottom=0.17,top=0.92))
for ax,(m,ttl) in zip(axs,[('d406','Days with HI ≥ 40.6 °C'),('tw28','Days with Tw ≥ 28 °C')]):
    obs={'d406':'d406','tw28':'tw28'}[m]
    for j,(lm,col,labx) in enumerate([(True,C_LMIC,'LMICs'),(False,C_HIC,'High income')]):
        s=a[a.lmic==lm]; xs=np.arange(4)+(j-0.5)*0.2
        mid=[s[obs].mean()]+[s[f'p{q}_c_{m}'].mean() for q in k]; low=[s[obs].mean()]+[s[f'p{q}_d_{m}'].mean() for q in k]
        ax.vlines(xs[1:],low[1:],mid[1:],color=col,lw=2.2,alpha=0.55)
        ax.plot(xs,mid,'o-',color=col,ms=3.5,mec='white',mew=0.4,lw=0.8,label=labx)
        ax.text(xs[-1]+0.12,mid[-1],f'{mid[-1]:.1f}',fontsize=5.5,va='center',color='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(lv); ax.set_ylabel(ttl+'\nin 30 days after landfall (mean per landfall)')
axs[1].legend(frameon=False,loc='upper left',bbox_to_anchor=(1.02,1.0)); panel(axs[0],'a',x=-0.2); panel(axs[1],'b',x=-0.2)
fig.savefig('figs/fig4_projections.png'); fig.savefig('figs/fig4_projections.pdf'); plt.close(fig)
print('ok')
