import csv, math
from collections import Counter
from ppo_training.cases import ThreeByThreeCaseFactory
from trajectory_conflicts import route_ids_conflict

def meta(seed=20260721,episodes=20):
    raw=ThreeByThreeCaseFactory(seed=seed, randomize=True, n_robots=3, fix_shortest_paths=False)
    fixed=ThreeByThreeCaseFactory(seed=seed, randomize=True, n_robots=3, fix_shortest_paths=True)
    out={}
    for ep in range(1,episodes+1):
        rq=raw(); fp=fixed()
        req=[(int(v),e,x,alpha) for v,e,x,alpha in rq.requests]
        npath=[len(pl.route_options) for pl in rq.plans]
        fplen=[]; ff=[]
        for pl in fp.plans:
            rt=pl.route_options[0]
            fplen.append(len(rt.intersections))
            ff.append(sum(rt.execution_times)+len(rt.edges)*pl.road_time)
        cnt=Counter()
        for pl in fp.plans:
            cnt.update(pl.route_options[0].intersections)
        shared={i for i,c in cnt.items() if c>1}
        conflict=0; poss=0
        for i in range(len(fp.plans)):
            for j in range(i+1,len(fp.plans)):
                int_i=fp.plans[i].route_options[0]
                int_j=fp.plans[j].route_options[0]
                for intr in set(int_i.intersections)&set(int_j.intersections):
                    ti=int_i.intersections.index(intr); tj=int_j.intersections.index(intr)
                    ri=int_i.traversals[ti].route_id; rj=int_j.traversals[tj].route_id
                    poss +=1
                    if route_ids_conflict(ri,rj): conflict +=1
        out[ep]={
            'requests':req,'npath_sum':sum(npath),'len_sum':sum(fplen),'ff_sum':sum(ff),
            'shared_cnt':len(shared),'conflict_ratio':None if poss==0 else conflict/poss,
            'release_span':max(x[3] for x in req)-min(x[3] for x in req),
            'conflict_pairs':conflict,'conflict_possible':poss
        }
    return out

m=meta()

def analyze(file):
    bad=[]
    with open(file,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rg=float(r['relative_gap'])
            ep=int(r['episode'])
            if math.isfinite(rg) and rg>1e-9:
                bad.append(ep)
    keys=['npath_sum','len_sum','ff_sum','shared_cnt','conflict_ratio','release_span']
    good=[i for i in m if i not in bad]
    print('\n',file,'bad',bad)
    for k in keys:
        b=[m[e][k] for e in bad if m[e][k] is not None]
        g=[m[e][k] for e in good if m[e][k] is not None]
        if not b: continue
        print(k,'bad',sum(b)/len(b), 'good',sum(g)/len(g),'bad_cnt',len(b),'good_cnt',len(g))

for f in ['output/ppo_n3/plan_run/eval_lr0p0001_act2_ep64_rand_mb32.csv','output/ppo_n3/plan_run/eval_lr0p0001_normAbs_ep64_rand_mb32.csv','output/ppo_n3/plan_run/eval_lr0p0001_skipTriv_ep64_rand_mb32.csv']:
    analyze(f)