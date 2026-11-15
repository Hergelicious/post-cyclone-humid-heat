# Sensitivity of simultaneous exposure to possibly unreported outage intervals in EAGLE-I (a missing record is read as zero).
# (a) internal gaps: runs of 1-23 zero hours inside an outage, bounded by customers out on both sides with the later value at least
#     half the earlier, are filled by linear interpolation; (b) abrupt endings: outages that end with a one-hour drop to zero from at
#     least 20% of the county peak are extended by a linear decline over a further 24 h.
import pandas as pd, numpy as np
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl')
base=0.0; fa=0.0; fb=0.0; nf=0; nab=0; nev=0; ab_storms={}
for r in recs:
    Ex=r['Ex'].astype('float64'); win=r['win'][:,None]; F=(r['HI']>=35); nh,nc=Ex.shape
    base+=(Ex*win*F).sum(); A=Ex.copy(); Bx=Ex.copy()
    for j in range(nc):
        x=Ex[:,j]; pk=x.max()
        if pk<=0: continue
        nev+=1; nz=np.where(x>0)[0]
        for a,b in zip(nz[:-1],nz[1:]):
            g=b-a-1
            if 1<=g<=23 and x[b]>=0.5*x[a]:
                A[a+1:b,j]=np.interp(np.arange(a+1,b),[a,b],[x[a],x[b]]); nf+=1
        last=nz[-1]
        if x[last]>=0.2*pk and last+1<nh:
            nab+=1; ab_storms[r['storm']]=ab_storms.get(r['storm'],0)+1; e=min(nh,last+25)
            Bx[last+1:e,j]=x[last]*(1-np.arange(1,e-last)/24)
    fa+=(A*win*F).sum(); fb+=(Bx*win*F).sum()
print(f'county outages {nev}; internal gaps filled {nf}; abrupt endings {nab} ({100*nab/nev:.1f}% of county outages)')
print(f'hour35 exposure: base {base/1e6:.1f} M; with gaps filled {fa/1e6:.1f} M ({100*(fa/base-1):+.1f}%); with abrupt endings extended {fb/1e6:.1f} M ({100*(fb/base-1):+.1f}%)')
print('storms with most abrupt endings:',sorted(ab_storms.items(),key=lambda x:-x[1])[:6])
pd.Series(dict(county_outages=nev,gaps_filled=nf,abrupt=nab,base=base,gapfill=fa,abrupt_ext=fb)).to_csv('pilot/us_missing_check.csv')
