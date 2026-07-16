from traffic_map import TrafficMap
from trajectory_conflicts import set_trajectory_conflict_filter
from main import make_relaxed_vehicle_plans, route_free_time
from coarse_scheduler import apply_relaxed_entrance_headway, search_dynamic_codesign_dfs_bb, write_interactive_solution_html

Dt=3.0
T_headway=2.0
lam=0.1
set_trajectory_conflict_filter(False)
tmap = TrafficMap.paper_3x3()
trips=[
    (1,3,9,0.0),
    (2,12,9,0.0),
    (3,12,5,0.0),
]
plans = make_relaxed_vehicle_plans(tmap, trips, Dt=Dt)
plans = apply_relaxed_entrance_headway(plans, headway=T_headway)
res = search_dynamic_codesign_dfs_bb(plans, lambda_path=lam, branch_and_bound=True, verbose=False)
print('best_g', res.best_g)
print('best_delay', res.best_node.g_delay)
print('best_path', res.best_node.g_path)
print('route_candidates', res.best_node.route_candidates)

for p, sel_t in zip(plans,res.best_node.route_candidates):
    sel_idx = sel_t[0]
    opts = p.route_options
    shortest = tmap.shortest_route_option(p.entrance,p.exit,road_time=Dt)
    sidx = [o.id for o in opts].index(shortest.id)
    extra = route_free_time(opts[sel_idx],road_time=Dt)-route_free_time(shortest,road_time=Dt)
    print(f'V{p.vehicle_id} P{p.entrance}->{p.exit}: sel_idx={sel_idx}(id={opts[sel_idx].id},hops={len(opts[sel_idx].intersections)}), shortest_idx={sidx}(id={shortest.id},hops={len(shortest.intersections)}), extra={extra:.6f}')
    print('  shortest intersections', shortest.intersections)
    print('  selected intersections', opts[sel_idx].intersections)

from pathlib import Path
out = Path('output/counterexample_3x3_relaxed_more_intersections_hop2_lam0_1')
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
print('saved', html)
with open(out / 'case_summary.txt', 'w', encoding='utf-8') as f:
    f.write('lambda=0.1\n')
    f.write(f'requests={trips}\n')
    f.write(f'best_g={res.best_g}\n')
    f.write(f'best_delay={res.best_node.g_delay}\n')
    f.write(f'best_path={res.best_node.g_path}\n')
    f.write(f'route_candidates={res.best_node.route_candidates}\n')
    f.write('vehicle1 V3->9 selected nonshortest with +2 hops\n')
print('saved', out / 'case_summary.txt')
