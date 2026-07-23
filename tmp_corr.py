import csv, math
from collections import defaultdict
from ppo_training.cases import ThreeByThreeCaseFactory
from trajectory_conflicts import route_ids_conflict
from collections import Counter

# gather cases metadata for ep1-20
raw=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=False)
fix=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=True)
meta={}
for ep in range(1,21):
    r=raw(); f=fix()
    req=[(v[3]) for v in r.requests]
    plans=f.plans
    fplen=[len(pl.route_options[0].intersections) for pl in plans]
    ff=[]
    for pl in plans:
        rt=pl.route_options[0]
        ff.append(sum(rt.execution_times)+len(rt.edges)*pl.road_time)
    cnt=Counter()
    for pl in plans:
        cnt.update(pl.route_options[0].intersections)
    conflict=0; poss=0
    for i in range(len(plans)):
        for j in range(i+1,len(plans)):
            set_i=set(plans[i].route_options[0].intersections)
            set_j=set(plans[j].route_options[0].intersections)
            for intr in set_i & set_j:
                poss += 1
                ti=list(plans[i].route_options[0].intersections).index(intr)
                tj=list(plans[j].route_options[0].intersections).index(intr)
                ri=plans[i].route_options[0].traversals[ti].route_id
                rj=plans[j].route_options[0].traversals[tj].route_id
                if route_ids_conflict(ri,rj):
                    conflict += 1
    # shared resources and first conflict pairs
    shared=sum(1 for c in cnt.values() if c>1)
    meta[ep] = {
        'release_span': max(req)-min(req),
        'len_sum': sum(fplen),
        'ff_sum': sum(ff),
        'npath_sum': sum(len(p.route_options) for p in r.plans),
        'shared':shared,
        'conflict_ratio': 0 if poss==0 else conflict/poss,
    }

# read one file's episode relative gaps
for fpath in ['output/ppo_n3/plan_run/eval_lr0p0001_ep64_rand_mb32.csv','output/ppo_n3/plan_run/eval_lr0p0001_act2_ep64_rand_mb32.csv','output/ppo_n3/plan_run/eval_lr0p0001_normAbs_ep64_rand_mb32.csv','output/ppo_n3/plan_run/eval_lr0p0001_skipTriv_ep64_rand_mb32.csv']:
    bad=[]
    with open(fpath,newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ep=int(row['episode']); rg=float(row['relative_gap']);
            if math.isfinite(rg) and rg>1e-9:
                bad.append((ep,rg,float(row['ppo_cost']),float(row['exact_cost'])) )
    print('\n',fpath.split('/')[-1], 'bad eps', [b[0] for b in bad])
    for ep,rg,pc,ec in bad:
        mm=meta[ep]
        print(' ep',ep,'gap',f'{rg:.3%}','meta',mm)