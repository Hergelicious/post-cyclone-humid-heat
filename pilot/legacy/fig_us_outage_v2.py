# Fig. 4 (v2): measured power outages and simultaneous humid heat after US landfalls (EAGLE-I + hourly ERA5), 2015-2024
import pandas as pd, numpy as np, sys
exec(open('pilot/style.py').read())
import matplotlib.patches as mp
TAG=sys.argv[1] if len(sys.argv)>1 else 'main'
R=pd.read_csv(f'pilot/us_v2_storms_{TAG}.csv'); R=R[R.counties.notna()].copy()
C=pd.read_pickle(f'pilot/us_v2_curves_{TAG}.pkl'); FS=C['FS']; cur=C['curves']
c406=mpl.colors.to_hex(cm.cividis(0.05)); c35=mpl.colors.to_hex(cm.cividis(0.55)); cng=mpl.colors.to_hex(cm.cividis(0.9))
fig=plt.figure(figsize=(180*MM,175*MM)); gs=fig.add_gridspec(3,2,width_ratios=[1.15,1],height_ratios=[1,1,1],hspace=0.62,wspace=0.5,left=0.15,right=0.97,top=0.965,bottom=0.085)
M=1e6
# a, simultaneous exposure by storm
ax=fig.add_subplot(gs[0,0]); S=R[R['hour35|obs']>=1e6].sort_values('hour35|obs'); y=np.arange(len(S))
a=S['hour406|obs']/M; b=(S['hour35|obs']-S['hour406|obs'])/M
ax.barh(y+0.18,a,color=c406,lw=0,height=0.36); ax.barh(y+0.18,b,left=a,color=c35,lw=0,height=0.36)
ax.barh(y-0.18,S['night26|obs']/M,color=cng,lw=0,height=0.36)
ax.set_yticks(y); ax.set_yticklabels([f'{s} ({int(v)})' for s,v in zip(S.storm,S.season)],fontsize=5.6); ax.set_xlim(0,140)
ax.set_xlabel('Customer‑hours without power (million)'); ax.tick_params(axis='y',length=0); panel(ax,'a',x=-0.30)
ax.legend(handles=[mp.Patch(color=c406,label='Hour with HI ≥ 40.6 °C'),mp.Patch(color=c35,label='Hour with HI 35–40.6 °C'),mp.Patch(color=cng,label='Night hour ≥ 26 °C')],frameon=False,loc='lower right',fontsize=5.3)
# b, customers without power through k or more hazardous daily peaks: bounds from aggregate county counts
ax=fig.add_subplot(gs[0,1]); k=np.arange(1,6); B=pd.read_csv('pilot/us_v3_bounds.csv',index_col=0)
up=[B.loc[f'up{i}','observed_million'] for i in k]; lo=[B.loc[f'lo{i}','observed_million'] for i in k]; fr=[B.loc[f'fr{i}','observed_million'] for i in k]
ax.bar(k-0.18,up,width=0.36,color=c35,lw=0); ax.bar(k+0.18,lo,width=0.36,color=c406,lw=0); ax.scatter(k+0.18,fr,s=7,color='black',zorder=3,lw=0)
ax.set_xticks(k); ax.set_xticklabels(['≥1','≥2','≥3','≥4','≥5']); ax.set_xlabel('Hazardous daily peaks spent without power'); ax.set_ylabel('Customers (million)'); panel(ax,'b',x=-0.22)
ax.legend(handles=[mp.Patch(color=c35,label='Upper bound'),mp.Patch(color=c406,label='Lower estimate, no churn'),mpl.lines.Line2D([],[],color='black',marker='o',lw=0,ms=2.5,label='Assumption‑free lower bound')],frameon=False,loc='upper right',fontsize=5.3)
# c, human influence on simultaneous exposure
ax=fig.add_subplot(gs[1,0]); nat=sorted({c.split('|',2)[2] for c in R if c.startswith('hour35|nat|')}); mods=sorted(set(x.split('|')[0] for x in nat))
cpk=mpl.colors.to_hex(cm.cividis(0.3))
for j,(met,col,mk) in enumerate([('hour35',c35,'D'),('peak35',cpk,'o')]):
    o=R[f'{met}|obs'].sum()
    for i,m in enumerate(mods):
        vv=[100*(1-R[f'{met}|nat|{x}'].sum()/o) for x in nat if x.split('|')[0]==m]
        yy=len(mods)-i+(0.15 if j==0 else -0.15); ax.scatter(vv,[yy]*len(vv),s=8,color=col,marker=mk,lw=0); ax.plot(np.mean(vv),yy,'|',color='black',ms=6)
    allv=np.array([100*(1-R[f'{met}|nat|{x}'].sum()/o) for x in nat]); yy=(0.15 if j==0 else -0.15)
    ax.plot([np.percentile(allv,5),np.percentile(allv,95)],[yy,yy],color=col,lw=1); ax.plot(allv.mean(),yy,mk,color=col,ms=3.5)
ax.set_yticks(list(range(len(mods),0,-1))+[0]); ax.set_yticklabels(mods+['All, 5–95%'],fontsize=5.6); ax.tick_params(axis='y',length=0)
ax.axvline(0,color='#999999',lw=0.5,ls=':'); ax.set_xlim(-10,100); ax.set_xlabel('Share attributable to human influence (%)'); panel(ax,'c',x=-0.30)
ax.legend(handles=[mpl.lines.Line2D([],[],color=c35,marker='D',lw=0,ms=3,label='Hours with HI ≥ 35 °C'),mpl.lines.Line2D([],[],color=cpk,marker='o',lw=0,ms=3,label='Hazardous daily peaks'),mpl.lines.Line2D([],[],color='black',marker='|',lw=0,ms=6,label='Model mean')],frameon=False,loc='upper left',fontsize=5.3,ncol=3,columnspacing=0.8,handletextpad=0.3)
ax.set_ylim(-0.6,len(mods)+1.3)
# d, restoration requirement for simultaneous exposure
ax=fig.add_subplot(gs[1,1]); storms=list(R.storm)
def agg(k_,m='hour35'): return sum(cur[(s,k_,m)] for s in storms if (s,k_,m) in cur)
i1=int(np.argmin(np.abs(FS-1))); obs=agg('obs'); ref=obs[i1]
natc=np.mean([agg(f'nat|{x}') for x in nat],0)
lines=[('Natural only',natc,'#9a9993','--'),('Observed',obs,'black','-')]
for wl,col in [(1.5,cm.cividis(0.75)),(2.0,cm.cividis(0.45)),(2.8,cm.cividis(0.1))]:
    ms=sorted(set(c.split('|')[2] for c in R if c.startswith(f'hour35|wl{wl}|'))); lines.append((f'{wl} °C',np.median([agg(f'wl{wl}|{m}') for m in ms],0),col,'-'))
for lab,v,col,ls in lines: ax.plot(FS,100*v/ref,color=col,lw=0.9,ls=ls,label=lab)
ax.axhline(100,color='#999999',lw=0.5,ls=':'); ax.axvline(1,color='#999999',lw=0.5,ls=':')
for lab,v,col,ls in lines[2:]:
    fx=float(np.interp(ref,v,FS)); ax.plot(fx,100,'o',color=col,ms=3)
ax.set_xlim(0.3,1.3); ax.set_ylim(0,300); ax.set_xlabel('Outage duration relative to observed'); ax.set_ylabel('Customer‑hours with HI ≥ 35 °C\n(% of observed)')
ax.legend(frameon=False,fontsize=5.3,loc='upper left'); panel(ax,'d',x=-0.22)
# e, restoration performance within storm and severity class against the reductions required by warming
ax=fig.add_subplot(gs[2,0]); EV=pd.read_csv('pilot/us_v3_events.csv'); EV['sev']=pd.cut(EV.peak_frac,[0,0.2,0.5,0.8,1.01])
E2=EV[np.isfinite(EV.t50)&(EV.t50>0)&EV.sev.notna()].copy(); E2['rel']=E2.t50/E2.groupby(['storm','sev']).t50.transform('median'); E2=E2[E2.groupby(['storm','sev']).t50.transform('size')>=5]
for wcol,lab,col in [(None,'County outages','black'),('peak','Customers out at peak',cm.cividis(0.55)),('hour35','Customer‑hours with HI ≥ 35 °C',cm.cividis(0.9))]:
    x=np.sort(E2.rel.values) if wcol is None else E2.rel.values[np.argsort(E2.rel.values)]
    w=np.ones(len(x)) if wcol is None else E2[wcol].values[np.argsort(E2.rel.values)]
    ax.plot(x,100*np.cumsum(w)/w.sum(),color=col,lw=0.9,label=lab)
for f,col in [(0.87,cm.cividis(0.75)),(0.71,cm.cividis(0.45)),(0.51,cm.cividis(0.1))]: ax.axvline(f,color=col,lw=0.7,ls='--')
ax.set_xlim(0,2); ax.set_ylim(0,100); ax.set_xlabel('Time to restore half of the peak outage,\nrelative to the median county of the same storm and severity')
ax.set_ylabel('Cumulative share (%)'); ax.legend(frameon=False,fontsize=5.3,loc='lower right'); panel(ax,'e',x=-0.30)
# f, time from outage onset to the first hazardous hour while still without power
ax=fig.add_subplot(gs[2,1]); LT=pd.read_csv('pilot/us_v3_leadtime.csv',index_col=0)
cats=[('≤12 h',['at onset','1-6 h','6-12 h']),('12–24 h',['12-24 h']),('24–48 h',['24-48 h']),('>48 h',['>48 h']),('None',['no overlap'])]
xx=np.arange(len(cats))
for i,(row,col) in enumerate([('observed','black'),('2.0 C',cm.cividis(0.45)),('2.8 C',cm.cividis(0.1))]):
    v=[100*LT.loc[row,c].sum() for _,c in cats]; ax.bar(xx+(i-1)*0.26,v,width=0.26,color=col,lw=0,label={'observed':'Observed','2.0 C':'2.0 °C','2.8 C':'2.8 °C'}[row])
ax.set_xticks(xx); ax.set_xticklabels([c for c,_ in cats]); ax.set_xlabel('Time from outage onset to first hazardous hour'); ax.set_ylabel('Customers out at onset (%)')
ax.set_ylim(0,62); ax.legend(frameon=False,fontsize=5.3,loc='upper center',ncol=3,columnspacing=0.8,handlelength=1.2); panel(ax,'f',x=-0.22)
fig.savefig('figs/fig_us_outage_v2.png'); fig.savefig('figs/fig_us_outage_v2.pdf'); print('ok')
