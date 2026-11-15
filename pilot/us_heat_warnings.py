# Were NWS heat products active when customers were without power in hazardous heat?
# Archived watches, warnings and advisories (Iowa Environmental Mesonet) for heat phenomena (HT heat advisory, EH excessive heat,
# XH extreme heat) are matched to counties through the forecast zone polygon containing each county's representative point,
# and compared hour by hour with the measured simultaneous exposure (EAGLE-I outages x hourly ERA5 heat index).
import pandas as pd, numpy as np, json, os, io, urllib.request, time
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
src=open('pilot/us_outage_heat.py').read(); exec(src[src.index('NAMES='):src.index('# ---- EAGLE-I rows')])
recs=pd.read_pickle('pilot/us_v2_recs_main.pkl')
pts=pd.read_csv('data/era5q/points.csv',dtype={'key':str}); cty=pts[pts.kind=='county'].copy(); cty['fips']=cty.key.str.zfill(5)
lat_of=dict(zip(cty.fips,cty.lat)); lon_of=dict(zip(cty.fips,cty.lon))
ST={'01':'AL','12':'FL','13':'GA','22':'LA','28':'MS','37':'NC','45':'SC','48':'TX','51':'VA','47':'TN','72':'PR','05':'AR','24':'MD','10':'DE','11':'DC','34':'NJ','36':'NY','42':'PA','54':'WV','21':'KY','40':'OK','29':'MO','17':'IL','18':'IN','39':'OH','09':'CT','25':'MA','44':'RI','33':'NH','50':'VT','23':'ME','26':'MI','55':'WI','19':'IA','20':'KS','31':'NE','35':'NM','27':'MN'}
os.makedirs('data/nws',exist_ok=True)
def get(url,f):
    if not os.path.exists(f):
        for a in range(4):
            try: urllib.request.urlretrieve(url,f); break
            except Exception as e: print('retry',url,e); time.sleep(5)
    return f
states=sorted({ST.get(f[:2],'') for r in recs for f in r['fips']}-{''})
# zone polygons -> county point mapping
zone_of={}
for s_ in states:
    g=json.load(open(get(f'https://mesonet.agron.iastate.edu/api/1/nws/ugcs.geojson?state={s_}',f'data/nws/ugcs_{s_}.json')))
    Z=[(ft['properties'].get('ugc') or ft.get('id'),shape(ft['geometry'])) for ft in g['features'] if ft.get('geometry') and str(ft['properties'].get('ugc') or ft.get('id'))[2:3]=='Z']
    if not Z: continue
    tree=STRtree([z[1] for z in Z])
    for f in [x for x in cty.fips if ST.get(x[:2])==s_]:
        p=Point(lon_of[f],lat_of[f]); hit=[i for i in tree.query(p) if Z[i][1].contains(p)]
        if hit: zone_of[f]=Z[hit[0]][0]
print('counties mapped to zones',len(zone_of))
L=pd.read_csv('pilot/landfalls.csv',parse_dates=['time'],keep_default_na=False); US=L[L.iso2.isin(['US','PR'])&(L.season>=2015)]
lfmap={NAMES.get(t,t):tm for t,tm in zip(US.track_id,US.time)}
FLAGS={}
rows=[]; tot={k:0.0 for k in ['h35','h35_adv','h406','h406_adv','pk35','pk35_adv','h35_lt406','h35_lt422']}
for r in recs:
    lf=lfmap[r['storm']].floor('h'); t0=lf-pd.Timedelta(hours=24); nh=r['Ex'].shape[0]
    sts=sorted({ST.get(f[:2],'') for f in r['fips']}-{''})
    a=(t0-pd.Timedelta(days=2)).strftime('%Y-%m-%dT%H:%MZ'); b=(t0+pd.Timedelta(hours=nh)).strftime('%Y-%m-%dT%H:%MZ')
    url='https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py?accept=csv&location_group=states&'+'&'.join(f'states={s_}' for s_ in sts)+f'&sts={a}&ets={b}'
    W=pd.read_csv(get(url,f"data/nws/ww_{r['storm']}_{r['season']}.csv"))
    W=W[W.phenomena.isin(['HT','EH','XH'])&W.significance.isin(['Y','W'])&W.ugc.notna()]
    W['i']=pd.to_datetime(W.utc_issue); W['e']=pd.to_datetime(W.utc_expire)
    hrs=t0+pd.to_timedelta(np.arange(nh),unit='h'); A=np.zeros(r['Ex'].shape,bool)
    for j,f in enumerate(r['fips']):
        z=zone_of.get(f)
        if z is None: continue
        for w in W[W.ugc==z].itertuples(): A[:,j]|=(hrs>=w.i)&(hrs<w.e)
    FLAGS[r['storm']]=A
    win=((r['tt']>=-24)&(r['tt']<720))[:,None]; E=r['Ex']*win; HI=r['HI']
    d=dict(storm=r['storm'],season=r['season'],h35=(E*(HI>=35)).sum(),h35_adv=(E*(HI>=35)*A).sum(),h406=(E*(HI>=40.6)).sum(),h406_adv=(E*(HI>=40.6)*A).sum(),
           pk35=(E*r['flags']['peak35']).sum(),pk35_adv=(E*r['flags']['peak35']*A).sum(),h35_lt406=(E*(HI>=35)*(HI<40.6)).sum(),h35_lt422=(E*(HI>=35)*(HI<42.2)).sum(),
           n_products=int(len(W)),mapped=float(np.mean([f in zone_of for f in r['fips']])))
    rows.append(d)
pd.to_pickle(FLAGS,'pilot/us_heat_warn_flags.pkl')
D=pd.DataFrame(rows); D.to_csv('pilot/us_heat_warnings.csv',index=False); t=D.sum(numeric_only=True)
print('share of simultaneous customer-hours (HI>=35) under an active heat advisory or warning: %.1f%%'%(100*t.h35_adv/t.h35))
print('share at HI>=40.6 under an active product: %.1f%%; customer-days at hazardous peaks under a product: %.1f%%'%(100*t.h406_adv/max(t.h406,1),100*t.pk35_adv/t.pk35))
print('share of simultaneous exposure with HI below 40.6 C: %.1f%%; below 42.2 C: %.1f%%'%(100*t.h35_lt406/t.h35,100*t.h35_lt422/t.h35))
print(D.sort_values('h35',ascending=False).head(10)[['storm','season','h35','h35_adv','pk35','pk35_adv','n_products','mapped']].assign(h35=lambda x:x.h35/1e6,h35_adv=lambda x:x.h35_adv/1e6,pk35=lambda x:x.pk35/1e6,pk35_adv=lambda x:x.pk35_adv/1e6).round(2).to_string())
