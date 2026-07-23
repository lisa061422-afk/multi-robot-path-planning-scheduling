import csv, math
from collections import Counter,defaultdict
from ppo_training.cases import ThreeByThreeCaseFactory
from trajectory_conflicts import route_ids_conflict


def case_meta(seed=20260721,episodes=20):
    raw=ThreeByThreeCaseFactory(seed=seed, randomize=True, n_robots=3, fix_shortest_paths=False)
    fixed=ThreeByThreeCaseFactory(seed=seed, randomize=True, n_robots=3, fix_shortest_paths=True)
    out={}
    for ep in range(1,episodes+1):
        r=raw(); f=fixed()
        raw_case=r
        fixed_plans=f.plans
        req=[(int(v),e,x,alpha) for v,e,x,alpha in raw_case.requests]
        npath=[len(pl.route_options) for pl in raw_case.plans]
        length=[len(pl.route_options[0].intersections) for pl in fixed_plans]
        ff=[]
        for pl in fixed_plans:
            rt=pl.route_options[0]
            ff.append(sum(rt.execution_times)+len(rt.edges)*pl.road_time)
        counts=Counter()
        for pl in fixed_plans:
            counts.update(pl.route_options[0].intersections)
        shared= {i:c for i,c in counts.items() if c>1}
        # contention counts
        conflict=0; poss=0
        for i in range(len(fixed_plans)):
            for j in range(i+1,len(fixed_plans)):
                int_i=set(fixed_plans[i].route_options[0].intersections)
                int_j=set(fixed_plans[j].route_options[0].intersections)
                ov=int_i & int_j
                if ov:
                    for intr in ov:
                        poss+=1
                        ti=list(fixed_plans[i].route_options[0].intersections).index(intr)
                        tj=list(fixed_plans[j].route_options[0].intersections).index(intr)
                        ri=fixed_plans[i].route_options[0].traversals[ti].route_id
                        rj=fixed_plans[j].route_options[0].traversals[tj].route_id
                        if route_ids_conflict(ri,rj):
                            conflict+=1
        out[ep]={
            'requests':req,
            'npath_sum':sum(npath),
            'len_sum':sum(length),
            'ff_sum':sum(ff),
            'shared_cnt':len(shared),
            'overlap_cnt_pairs': sum(1 for _ in []),
            'conflict_pairs':conflict,
            'conflict_possible':poss,
            'conflict_ratio': None if poss==0 else conflict/poss,
            'release_span': max([x[3] for x in req])-min([x[3] for x in req]),
        }
    return out

meta=case_meta()

bad_episodes=[]
with open('output/ppo_n3/plan_run/eval_lr0p0001_ep64_rand_mb32.csv',newline='',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        ep=int(r['episode']); rg=float(r['relative_gap'])
        if math.isfinite(rg) and rg>1e-9:
            bad_episodes.append(ep)

print('bad episodes in baseline file:',bad_episodes)
keys=['npath_sum','len_sum','ff_sum','shared_cnt','conflict_ratio','release_span']
good=[ep for ep in meta if ep not in bad_episodes]
for k in keys:
    bg=[meta[e][k] for e in bad_episodes]
    gg=[meta[e][k] for e in good]
    # conflict_ratio may be None
    if k=='conflict_ratio':
        bg=[x for x in bg if x is not None]; gg=[x for x in gg if x is not None]
    print(k,'bad mean',sum(bg)/len(bg) if bg else None, 'good mean',sum(gg)/len(gg) if gg else None, 'bad median', sorted(bg)[len(bg)//2] if bg else None, 'good median', sorted(gg)[len(gg)//2] if gg else None)
print('count bad',len(bad_episodes),'count good',len(good))