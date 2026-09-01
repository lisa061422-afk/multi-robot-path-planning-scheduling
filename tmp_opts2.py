from ppo_training.cases import ThreeByThreeCaseFactory
rf=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=False)
# maybe lengths
fs=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=True)
rows=[]
for ep in range(1,21):
    rq=rf(); fp=fs()
    opts=[len(p.route_options) for p in rq.plans]
    len_each=[len(p.route_options[0].intersections) for p in fp.plans]
    ff_each=[]
    for p in fp.plans:
        ro=p.route_options[0]
        ff_each.append(sum(ro.execution_times)+len(ro.edges)*p.road_time)
    rows.append((ep,sum(opts),sum(len_each),sum(ff_each)))
print(rows)
for k in ['sum','len','ff']:
    pass