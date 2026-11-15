import pickle, numpy as np, pandas as pd
P=pickle.load(open('pilot/attrib_per.pkl','rb')); obs=P['obs']; d35=P['d35']; rng=np.random.default_rng(13)
rows=[]
for k,a,b in P['per']: rows.append(dict(model=k[0],member=k[1],cf_pd=a,cf_d35=b))
R=pd.DataFrame(rows)
out={}
for lab,fact,col in [('d35',d35,'cf_d35'),('pd',obs,'cf_pd')]:
    full=np.array([1-c.sum()/fact.sum() for c in R[col]]); R['f_'+lab]=full
    mm=R.groupby('model')['f_'+lab].mean()
    v_model=mm.var(ddof=1); v_real=R.groupby('model')['f_'+lab].var(ddof=1).mean()
    vy=[]
    for c in R[col]:
        B=[];
        for _ in range(1000):
            yi=rng.integers(0,20,20); B.append(1-c[yi].sum()/fact[yi].sum())
        vy.append(np.var(B,ddof=1))
    v_year=np.mean(vy); tot=v_model+v_real+v_year
    # one-factor ranges (5-95)
    print(lab,'central',round(full.mean(),3),'model means',mm.round(2).to_dict())
    print('  variance shares: model',round(v_model/tot,2),'realisation',round(v_real/tot,2),'year',round(v_year/tot,2),' sd',round(np.sqrt(v_model),3),round(np.sqrt(v_real),3),round(np.sqrt(v_year),3))
    out[lab]=dict(central=full.mean(),sd_model=np.sqrt(v_model),sd_real=np.sqrt(v_real),sd_year=np.sqrt(v_year),share_model=v_model/tot,share_real=v_real/tot,share_year=v_year/tot,mm_min=mm.min(),mm_max=mm.max())
pd.DataFrame(out).T.to_csv('pilot/attrib_variance.csv'); print(R.groupby('model').member.apply(list).to_dict())
