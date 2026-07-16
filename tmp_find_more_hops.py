import re, random
from pathlib import Path
from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter
from main import make_relaxed_vehicle_plans, route_free_time
from coarse_scheduler import apply_relaxed_entrance_headway, search_dynamic_codesign_dfs_bb, write_interactive_solution_html

set_trajectory_conflict_filter(False)
tmap = TrafficMap.paper_3x3()
Dt=3.0
T_headway=2.0
lambdas=[0.0,0.1,0.2,0.5,1.0]

ports=[]
for line in tmap.describe_ports():
    m = re.search(r'port\s+(\d+)', line)
    if m:
        ports.append(int(m.group(1)))
ports = sorted(set(ports))

od_all=[(e,x) for e in ports for x in ports if e!=x and len(tmap.route_options(e,x))>1]


def hop(route):
    return len(route.intersections)

def run_case(trips, lam):
    plans = make_relaxed_vehicle_plans(tmap, trips, Dt=Dt)
    plans = apply_relaxed_entrance_headway(plans, headway=T_headway)
    res = search_dynamic_codesign_dfs_bb(
        plans,
        lambda_path=lam,
        branch_and_bound=True,
        verbose=False,
    )
    return res, plans

def check(res, plans):
    diffs=[]
    for p, sel_t in zip(plans, res.best_node.route_candidates):
        sel_idx = sel_t[0]
        opts = p.route_options
        ids = [o.id for o in opts]
        shortest = tmap.shortest_route_option(p.entrance, p.exit, road_time=Dt)
        sidx = ids.index(shortest.id)
        if sel_idx == sidx:
            continue
        dh = hop(opts[sel_idx]) - hop(shortest)
        if dh >= 1:
            de = route_free_time(opts[sel_idx], road_time=Dt) - route_free_time(shortest, road_time=Dt)
            diffs.append((p.vehicle_id,p.entrance,p.exit,sel_idx,sidx,dh,de,opts[sel_idx].id,shortest.id,hop(opts[sel_idx]),hop(shortest)))
    return diffs

random.seed(2026)
for lam in lambdas:
    print('--- lambda',lam,'---')
    found=False
    for t in range(1,8000):
        trips=[
            (1,*random.choice(od_all),0.0),
            (2,*random.choice(od_all),0.0),
            (3,*random.choice(od_all),0.0),
        ]
        res, plans = run_case(trips, lam)
        diffs = check(res, plans)
        if diffs:
            print('FOUND lam',lam,'at trial',t)
            print('trips',trips)
            print('best_g',res.best_g,'delay',res.best_node.g_delay,'path',res.best_node.g_path)
            print('route_candidates',res.best_node.route_candidates)
            for row in diffs:
                vid,e,x,sel,sidx,dh,de,selid,sid,hsel,hshort = row
                print(f'  V{vid} P{e}->P{x}: sel_id={selid},short_id={sid},hop {hshort}->{hsel}(delta{dh}),extra={de:.6f}')
            suffix = str(lam).replace('.', '_')
            out = Path(f'output/counterexample_3x3_relaxed_hop_lam{suffix}')
            out.mkdir(parents=True, exist_ok=True)
            html = out / 'relaxed_interactive_solution.html'
            write_interactive_solution_html(
                res,
                html,
                plans=plans,
                tmap=tmap,
                max_terminal_paths=300,
                max_tree_nodes=8000,
                lambda_path=lam,
            )
            with open(out / 'case_summary.txt','w',encoding='utf-8') as f:
                f.write(f'lambda={lam}\n')
                f.write(f'trips={trips}\n')
                f.write(f'best_g={res.best_g}\n')
                f.write(f'best_delay={res.best_node.g_delay}\n')
                f.write(f'best_path={res.best_node.g_path}\n')
                f.write(f'route_candidates={res.best_node.route_candidates}\n')
                for row in diffs:
                    vid,e,x,sel,sidx,dh,de,selid,sid,hsel,hshort=row
                    f.write(f'V{vid} P{e}->{x}: sel_id={selid} hop={hshort}->{hsel} delta={dh}, short_id={sid}, extra={de}\n')
            print('saved',html)
            found=True
            break
        if t % 500 == 0:
            print('  trial',t)
    if found:
        break
    else:
        print('not found for lambda',lam)
