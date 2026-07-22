import re, random
from pathlib import Path
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter
from main import make_relaxed_vehicle_plans, route_free_time
from coarse_scheduler import apply_relaxed_entrance_headway, search_dynamic_codesign_dfs_bb

set_trajectory_conflict_filter(False)
tmap=TrafficMap.paper_3x3()
Dt=3.0
T_headway=2.0

ports=[]
for line in tmap.describe_ports():
    m=re.search(r'port\s+(\d+)',line)
    if m:
        ports.append(int(m.group(1)))
ports=sorted(set(ports))
od_all=[(e,x) for e in ports for x in ports if e!=x and len(tmap.route_options(e,x))>1]

def hop(route):
    return len(route.intersections)

def run_case(trips):
    plans=make_relaxed_vehicle_plans(tmap,trips,Dt=Dt)
    plans=apply_relaxed_entrance_headway(plans,headway=T_headway)
    res=search_dynamic_codesign_dfs_bb(plans,branch_and_bound=True,verbose=False)
    return res,plans

def check(res,plans):
    out=[]
    for p,sel_t in zip(plans,res.best_node.route_candidates):
        sel_idx=sel_t[0]
        opts=p.route_options
        ids=[o.id for o in opts]
        shortest=tmap.shortest_route_option(p.entrance,p.exit,road_time=Dt)
        sidx=ids.index(shortest.id)
        if sel_idx==sidx:
            continue
        dh=hop(opts[sel_idx])-hop(shortest)
        if dh>=2:
            de=route_free_time(opts[sel_idx],road_time=Dt)-route_free_time(shortest,road_time=Dt)
            out.append((p.vehicle_id,p.entrance,p.exit,sel_idx,sidx,dh,de,opts[sel_idx].id,shortest.id,hop(opts[sel_idx]),hop(shortest)))
    return out

random.seed(99)
for t in range(1,8000):
    trips=[(1,*random.choice(od_all),0.0),(2,*random.choice(od_all),0.0),(3,*random.choice(od_all),0.0)]
    res,plans=run_case(trips)
    diffs=check(res,plans)
    if diffs:
        print('FOUND',t,trips)
        print('best_g',res.best_g,'delay',res.best_node.g_delay,'path',res.best_node.g_path)
        for row in diffs:
            vid,e,x,sel,sidx,dh,de,selid,sid,hsel,hshort=row
            print(f'V{vid} P{e}->{x}: sel_id={selid} short_id={sid}, hop {hshort}->{hsel}(delta{dh}) extra={de}')
        break
    if t%500==0:
        print('trial',t)
else:
    print('NOT FOUND')
