from ppo_training.cases import ThreeByThreeCaseFactory

raw=ThreeByThreeCaseFactory(seed=20260721,randomize=True,n_robots=3,fix_shortest_paths=False)
opts=[]
for ep in range(1,21):
    c=raw()
    n=[len(p.route_options) for p in c.plans]
    opts.append((ep,sum(n),n))
print('all sums',opts)
print('max sum',max(s for _,s,_ in opts),'min',min(s for _,s,_ in opts),'mean',sum(s for _,s,_ in opts)/len(opts))
print('ep2', [o for o in opts if o[0] in (2,3,4,5,6,7,8,12,15,17,18)])