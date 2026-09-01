from collections import defaultdict
from ppo_training.cases import ThreeByThreeCaseFactory

rf=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=False)
bad={2,8,12}
for ep in range(1,21):
    c=rf(); fp=c
    # requests from fixed-shortest? use c.requests
    req=[(v,e,x,a) for v,e,x,a in c.requests]
    first_int=[]
    first_time=[]
    for p in c.plans:
        rt=p.route_options[0] if p.route_options else None
        if rt is None:
            first_int.append(None); first_time.append(None)
        else:
            first_int.append(rt.intersections[0])
            first_time.append(p.route_options[0])
    # For our purpose use requests alpha
    first_by_time={} 
    
    first_map=defaultdict(list)
    for (veh,entr,exit,a), p in zip(c.requests,c.plans):
        first_map[(p.route_options[0].intersections[0],a)].append(veh)
    # also count same intersection irrespective time
    map2=defaultdict(int)
    for (inter,a),vs in first_map.items():
        map2[inter]+=len(vs)
    dup=sum(1 for v in map2.values() if v>1)
    if ep in bad:
        print('ep',ep,'requests',c.requests,'first_inter',[(v,p.route_options[0].intersections[0],a) for v,p,a in zip(range(1,4),c.plans,[x[3] for x in c.requests])], 'dup_first_inter',dup, 'first_map',dict(first_map))