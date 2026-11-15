# Figure: measured power outages and humid heat after US landfalls (EAGLE-I + hourly ERA5), 2015-2024
import pandas as pd, numpy as np, sys
exec(open('pilot/style.py').read())
import matplotlib.patches as mp
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
R=pd.read_csv(f'pilot/us_outage_storms_{TAG}.csv'); R=R[R.counties.notna()].copy()
C=pd.read_pickle(f'pilot/us_outage_curves_{TAG}.pkl'); FS=C['FS']; cur=C['curves']; P=pd.read_csv(f'pilot/us_outage_profile_{TAG}.csv')
c406=mpl.colors.to_hex(cm.cividis(0.05)); c35=mpl.colors.to_hex(cm.cividis(0.55)); cno='#d9d8d4'
fig=plt.figure(figsize=(180*MM,118*MM)); gs=fig.add_gridspec(2,2,width_ratios=[1.15,1],height_ratios=[1,1],hspace=0.55,wspace=0.5,left=0.15,right=0.93,top=0.95,bottom=0.09)
# a, storms
ax=fig.add_subplot(gs[0,0]); S=R[R.cust_h35>=1e6].sort_values('cust_h35'); y=np.arange(len(S))
a=S.cust_h406/1e6; b=(S.cust_h35-S.cust_h406)/1e6; c=(S.cust_h-S.cust_h35)/1e6
ax.barh(y,a,color=c406,lw=0,height=0.7); ax.barh(y,b,left=a,color=c35,lw=0,height=0.7); ax.barh(y,c,left=a+b,color=cno,lw=0,height=0.7)
ax.set_yticks(y); ax.set_yticklabels([f'{s} ({int(v)})' for s,v in zip(S.storm,S.season)],fontsize=5.6); ax.set_xlim(0,600)
ax.set_xlabel('Customer‑hours without power, days 0–30 (million)'); ax.tick_params(axis='y',length=0); panel(ax,'a',x=-0.30)
ax.legend(handles=[mp.Patch(color=c406,label='HI ≥ 40.6 °C day'),mp.Patch(color=c35,label='HI 35–40.6 °C day'),mp.Patch(color=cno,label='Cooler day')],frameon=False,loc='lower right',fontsize=5.3)
# b, by day since landfall
ax=fig.add_subplot(gs[0,1]); Q=P[P.day<=14]
ax.bar(Q.day,Q.cust_h35,color=c35,lw=0,width=0.8); ax.bar(Q.day,Q.cust_h-Q.cust_h35,bottom=Q.cust_h35,color=cno,lw=0,width=0.8)
ax.set_xlabel('Day after landfall'); ax.set_ylabel('Customer‑hours (million)'); ax.set_xticks([0,3,7,10,14]); panel(ax,'b',x=-0.22)
ax2=ax.twinx(); q=Q[Q.day<=8]; ax2.plot(q.day,100*q.share_hot,'o-',color=c406,ms=2.5,lw=0.8); ax2.set_ylim(0,100); ax2.set_ylabel('Share on HI ≥ 35 °C days (%)'); ax2.spines['right'].set_visible(True)
ax.legend(handles=[mp.Patch(color=c35,label='HI ≥ 35 °C day'),mp.Patch(color=cno,label='Cooler day'),mpl.lines.Line2D([],[],color=c406,marker='o',ms=2.5,lw=0.8,label='Share')],frameon=False,loc='upper right',fontsize=5.3)
# c, human influence
ax=fig.add_subplot(gs[1,0]); nat=[c for c in R if c.startswith('E35|nat|')]; mods=sorted(set(c.split('|')[2] for c in nat))
for j,(thr,col,mk) in enumerate([('35',c35,'D'),('406',c406,'o')]):
    o=R[f'E{thr}|obs'].sum()
    for i,m in enumerate(mods):
        v=[100*(1-R[c.replace('E35',f'E{thr}')].sum()/o) for c in nat if c.split('|')[2]==m]
        yy=len(mods)-i+(0.15 if j==0 else -0.15); ax.scatter(v,[yy]*len(v),s=8,color=col,marker=mk,lw=0); ax.plot(np.mean(v),yy,'|',color='black',ms=6)
    allv=np.array([100*(1-R[c.replace('E35',f'E{thr}')].sum()/o) for c in nat]); yy=0+(0.15 if j==0 else -0.15)
    ax.plot([np.percentile(allv,5),np.percentile(allv,95)],[yy,yy],color=col,lw=1); ax.plot(allv.mean(),yy,mk,color=col,ms=3.5)
ax.set_yticks(list(range(len(mods),0,-1))+[0]); ax.set_yticklabels(mods+['All, 5–95%'],fontsize=5.6); ax.tick_params(axis='y',length=0)
ax.axvline(0,color='#999999',lw=0.5,ls=':'); ax.set_xlim(-10,100); ax.set_xlabel('Share of hot‑day customer‑hours attributable\nto human influence (%)'); panel(ax,'c',x=-0.30)
ax.legend(handles=[mpl.lines.Line2D([],[],color=c35,marker='D',lw=0,ms=3,label='HI ≥ 35 °C'),mpl.lines.Line2D([],[],color=c406,marker='o',lw=0,ms=3,label='HI ≥ 40.6 °C'),mpl.lines.Line2D([],[],color='black',marker='|',lw=0,ms=6,label='Model mean')],frameon=False,loc='upper left',fontsize=5.3)
# d, restoration requirement (HI>=35)
ax=fig.add_subplot(gs[1,1]); storms=list(R.storm)
def agg(k): return sum(cur[(s,k,35)] for s in storms if (s,k,35) in cur)
i1=int(np.argmin(np.abs(FS-1))); obs=agg('obs'); ref=obs[i1]
natk=[c.split('|',1)[1] for c in nat]; natc=np.mean([agg(k) for k in natk],0)
lines=[('Natural only',natc,'#9a9993','--')]
for wl,col in [(1.5,cm.cividis(0.75)),(2.0,cm.cividis(0.45)),(2.8,cm.cividis(0.1))]:
    ms=sorted(set(c.split('|')[2] for c in R if c.startswith(f'E35|wl{wl}|'))); lines.append((f'{wl} °C',np.median([agg(f'wl{wl}|{m}') for m in ms],0),col,'-'))
lines.insert(1,('Observed',obs,'black','-'))
for lab,v,col,ls in lines: ax.plot(FS,100*v/ref,color=col,lw=0.9,ls=ls,label=lab)
ax.axhline(100,color='#999999',lw=0.5,ls=':'); ax.axvline(1,color='#999999',lw=0.5,ls=':')
for lab,v,col,ls in lines[2:]:
    fx=float(np.interp(ref,v,FS)); ax.plot(fx,100,'o',color=col,ms=3)
ax.set_xlim(0.3,1.3); ax.set_ylim(0,170); ax.set_xlabel('Outage duration relative to observed'); ax.set_ylabel('Hot‑day customer‑hours (% of observed)')
ax.legend(frameon=False,fontsize=5.3,loc='upper left'); panel(ax,'d',x=-0.22)
fig.savefig('figs/fig_us_outage.png'); fig.savefig('figs/fig_us_outage.pdf'); print('ok')
