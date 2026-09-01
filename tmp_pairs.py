from collections import Counter,defaultdict
from ppo_training.cases import ThreeByThreeCaseFactory
from trajectory_conflicts import route_ids_conflict

def analyze(ep):
    f=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=True)
    for i in range(ep):
        c=f()
    plans=c.plans
    req=c.requests
    # pair shared intersections and route conflicts
    print('episode',ep,'requests',req)
    for i in range(3):
        ri=plans[i].route_options[0]; inter_i=list(ri.intersections)
        for j in range(i+1,3):
            rj=plans[j].route_options[0]; inter_j=list(rj.intersections)
            shared=[x for x in inter_i if x in inter_j]
            shared=set(shared)
            print(' pair',i+1,j+1,'shared',sorted(shared),'count',len(shared))
            if shared:
                for s in sorted(shared):
                    t1=inter_i.index(s); t2=inter_j.index(s)
                    rid1=ri.traversals[t1].route_id; rid2=rj.traversals[t2].route_id
                    print('  I{}: rID ({},{}) conflict={}'.format(s,rid1,rid2, route_ids_conflict(rid1,rid2)))
    # first intersection/time
    for p in plans:
        print('  N{} first I{} alpha{}'.format(p.vehicle_id,p.route_options[0].intersections[0],c.requests[p.vehicle_id-1][3]))
    print('---')

for ep in [2,8,12]:
    analyze(ep)