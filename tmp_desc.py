from ppo_training.cases import ThreeByThreeCaseFactory

def describe(eps):
    rf=ThreeByThreeCaseFactory(seed=20260721, randomize=True, n_robots=3, fix_shortest_paths=False)
    fs=ThreeByThreeCaseFactory(seed=20260721, randomize=True, n_robots=3, fix_shortest_paths=True)
    for ep in range(1,max(eps)+1):
        if ep not in eps: 
            _=rf(); _=fs();
            continue
        rq=rf(); fp=fs()
        print('--- episode',ep,'---')
        for i,(veh,en,ex,rel) in enumerate(rq.requests,1):
            # raw options count
            raw_plan=rf.plans[i-1] if False else None
        for p in rq.plans:
            pass
        for p_fix in fp.plans:
            idx=p_fix.vehicle_id
            raw_raw=None
        for ii in range(len(rq.plans)):
            r=fp.plans[ii]
            rr=r.route_options[0]
            print(f'N{r.vehicle_id}: ent->ext {r.entrance}->{r.exit} | alpha0={rq.requests[ii][3]}')
            print('  options_all=',len(rq.plans[ii].route_options),' shortest_intersections=',rr.intersections)
            print('  rIDs:', [tr.route_id for tr in rr.traversals], 'turns:', [tr.turn for tr in rr.traversals])
            print('  freeflow=',sum(rr.execution_times)+len(rr.edges)*r.road_time)

defeps=[2,8,12]
describe(defeps)