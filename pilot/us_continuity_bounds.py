# Bounds on the share of simultaneous exposure (customers out in hours with HI >= 35 C) borne by customers already without power
# for at least D hours, from aggregate county counts: (i) onset assumption (all customers out at t have been out since the county
# onset); (ii) upper bound: running minimum of customers out over the previous D hours; (iii) lower bound: customers out at t minus
# all increases in customers out during the previous D hours (each increase treated as newly disconnected customers).
import numpy as np, pandas as pd
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl'); out={D:np.zeros(3) for D in (24,48,72)}; tot=0.0
for r in recs:
    Ex=r['Ex'].astype('float64'); nh,nc=Ex.shape; win=((r['tt']>=-24)&(r['tt']<720))[:,None]; F=(r['HI']>=35)&win; tot+=(Ex*F).sum()
    inc=np.clip(np.diff(Ex,axis=0,prepend=Ex[:1]),0,None); cinc=np.cumsum(inc,0)
    el=np.arange(nh)[:,None]-r['onset'][None]
    for D in (24,48,72):
        a=(Ex*F*(el>=D)).sum()
        rm=np.full_like(Ex,0.0)
        for t in range(D,nh): rm[t]=Ex[t-D:t+1].min(0)
        b=(rm*F).sum()
        lo=np.zeros_like(Ex); lo[D:]=np.clip(Ex[D:]-(cinc[D:]-cinc[:-D]),0,None); c=(lo*F).sum()
        out[D]+=[a,b,c]
for D,v in out.items(): print(f'>= {D} h: onset assumption {100*v[0]/tot:.0f}%, running-minimum bound {100*v[1]/tot:.0f}%, increments lower bound {100*v[2]/tot:.0f}%')
