# PPOè°ƒå‚æ‰§è¡Œè®¡åˆ’ï¼ˆæŒ‰åºå•å˜é‡å®žéªŒï¼‰

ç›®æ ‡ï¼šåœ¨ `fixed shortest path + scheduling` åœºæ™¯ä¸‹ï¼Œç”¨æœ€å°æ”¹åŠ¨æ‰¾å‡ºæ›´ç¨³çš„æ³›åŒ–å‚æ•°ï¼ˆä»¥éšæœºéªŒè¯é›†ä¸ºä¸»ï¼‰ã€‚

åŽŸåˆ™ï¼ˆå…ˆéµå®ˆï¼‰ï¼š
1. æ¯ä¸€è½®åªæ”¹ **ä¸€ä¸ªè¶…å‚æ•°ç»„**ï¼ˆå…¶ä½™å‚æ•°ä¿æŒä¸å˜ï¼‰ã€‚
2. åŒä¸€ç»„å®žéªŒå…ˆå›ºå®šéšæœºç§å­ï¼ˆé»˜è®¤ 20260721ï¼‰ï¼Œé¿å…ç»“æžœå™ªå£°æ¥è‡ªéšæœºæ€§ã€‚
3. æ¯æ¬¡éƒ½è¾“å‡ºï¼š`fixed-case` æœ€ç»ˆ gapã€`randomized eval` çš„ `mean / median / max relative gap`ã€`exact_solved`ã€‚
4. è‹¥å•è½®å®žéªŒæ— æ”¹å–„ï¼Œå›žé€€åˆ°ä¸Šä¸€ä¼˜ç»“æžœï¼Œä¸åšæ— åºå åŠ ã€‚

## å½“å‰åŸºçº¿ï¼ˆä½œä¸º Step 0ï¼‰
- Model: `--actor-hidden-layers 3 --hidden-dim 256`
- è®­ç»ƒ: `--fixed-case --fix-shortest-paths --updates 20 --episodes-per-update 8 --n-robots 3`
- è¯„ä¼°: `--exact --episodes 20 --randomize --fix-shortest-paths --n-robots 3`

## å®žéªŒé¡ºåºï¼ˆå»ºè®®ï¼‰

### Step 1ï¼šå…ˆè°ƒ LRï¼ˆå…¶ä»–å‚æ•°å›ºå®šï¼‰
- å€™é€‰ï¼š`3e-4`ï¼ˆåŸºçº¿ï¼‰, `1e-4`, `6e-4`
- ä¸æ”¹ actor æ·±åº¦ï¼ˆå…ˆæ²¿ç”¨åŸºçº¿ 3å±‚ï¼‰
- æ¯æ¬¡è®­ç»ƒå®Œåšéšæœºè¯„ä¼°ï¼ˆ20 episodesï¼‰

### Step 2ï¼šå†è°ƒ Entropyï¼ˆåœ¨ step1 æœ€ä¼˜ LR ä¸‹ï¼‰
- å€™é€‰ï¼š`0.02`, `0.01`ï¼ˆåŸºçº¿ï¼‰, `0.005`
- ä¿æŒ step1 æœ€ä¼˜ LR

### Step 3ï¼šå†è°ƒ æ¯è½®é‡‡æ · episode æ•°ï¼ˆå›ºå®š step1/step2 æœ€ä¼˜å€¼ï¼‰
- å€™é€‰ï¼š`8`, `16`ï¼ˆåŸºçº¿ï¼‰, `32`
- ä¿æŒå›ºå®šå­¦ä¹ çŽ‡å’Œ entropy

### Step 4ï¼šå›ºå®šæœ€ä¼˜ LR/Entropy/episodes åŽï¼Œå°è¯• critic ç»“æž„ï¼ˆå¯é€‰ï¼‰
- å…ˆåªæ”¹ `StateValueCritic hidden_dim` ä¸Žå±‚æ•°ï¼ˆå¦‚ 128/256ï¼‰ï¼Œè§‚å¯Ÿæ˜¯å¦æ”¹å–„ `max gap`

## è®°å½•è¡¨ï¼ˆæ‰§è¡Œæ—¶é€è¡Œå¡«ï¼‰

| Step | é…ç½® | fixed-case gap | éšæœºeval mean gap | éšæœºeval median gap | éšæœºeval max gap | exact_solved |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (baseline) | actor-hidden-layers=3, hidden-dim=256 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 1.1 | LR=3e-4 (å›ºå®š) | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 1.2 | LR=1e-4 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 1.3 | LR=6e-4 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 2.1 | Entropy=0.02 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 2.2 | Entropy=0.01ï¼ˆåŸºçº¿ï¼‰ | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 2.3 | Entropy=0.005 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 3.1 | episodes/update=8ï¼ˆåŸºçº¿ï¼‰ | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 3.2 | episodes/update=16 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 3.3 | episodes/update=32 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.1 | critic_hidden_dim=128 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.2 | critic_hidden_dim=256ï¼ˆåŸºçº¿ï¼‰ | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.3 | critic_hidden_dim=512 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.4 | critic_hidden_layers=1 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.5 | critic_hidden_layers=2ï¼ˆåŸºçº¿ï¼‰ | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 4.6 | critic_hidden_layers=3 | 0.000000 | 216.308% | 8.348% | 1829.896% | 20/20 |
| 5 | training mode=随机case（去掉 --fixed-case） | 0.000000 | 15.810% | 0.000% | 212.791% | 20/20 |
| 5.1 | 随机case, episodes/update=64, minibatch=32 | 0.000000 | 5.171% | 0.000% | 103.419% | 20/20 |
| 5.2 | 随机case, episodes/update=128, minibatch=32 | 0.000000 | 5.171% | 0.000% | 103.419% | 20/20 |
| 5.3 | 随机case, episodes/update=64, minibatch=16 | 0.000000 | 5.171% | 0.000% | 103.419% | 20/20 |
| 5.4 | 随机case, episodes/update=64, minibatch=16, entropy=0.005 | 0.000000 | 5.171% | 0.000% | 103.419% | 20/20 |

## ç»Ÿä¸€æ‰§è¡Œå‘½ä»¤æ¨¡æ¿

```powershell
# è®­ç»ƒ
.\\.venv\\Scripts\\python.exe -m ppo_training.train --fixed-case --updates 20 --episodes-per-update 8 --n-robots 3 --fix-shortest-paths --actor-hidden-layers 3 --hidden-dim 256 --learning-rate {LR} --entropy-coef {ENT} --checkpoint output\ppo_n3\plan_run\ppo_seed20260721_lr{LR}_ent{ENT}.pt --metrics-csv output\ppo_n3\plan_run\metrics_lr{LR}_ent{ENT}.csv

# è¯„ä¼°ï¼ˆå›ºå®š seedï¼‰
.\\.venv\\Scripts\\python.exe -m ppo_training.evaluate --checkpoint output\ppo_n3\plan_run\ppo_seed20260721_lr{LR}_ent{ENT}.pt --episodes 20 --exact --n-robots 3 --randomize --fix-shortest-paths
```

## æ¬æ¬¡æµè¯•è¡¨ (2026-07-22 æ¥æä¾)

### è®­ç»ƒè®¾ç½®
- updates: 20
- episodes/update: 64
- minibatch: 32
- n-robots: 3
- fixed-shortest-path + randomize eval
- seed: 20260721
- actor: hidden_layers=3, hidden_dim=256

### ç»“æžœå¯¹æ¯ (N=3, random eval 20 cases)

| Config | update/ckpt | mean_gap | median_gap | max_gap | exact_solved |
| --- | --- | --- | --- | --- | --- |
| lr=1e-4 | `ppo_lr0p0001_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20（finite gap=14/20，exact_cost=0 的 case 会导致 inf） |
| lr=6e-4 | `ppo_lr0p0006_ep64_rand_mb32.pt` | 2368.092% | 1554.961% | 8938.018% | 20/20（finite gap=14/20） |
| lr=1e-4, critic_hidden_dim=128 | `ppo_lr0p0001_crt128_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, critic_hidden_dim=512 | `ppo_lr0p0001_crt512_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, critic_hidden_layers=1 | `ppo_lr0p0001_crtH1_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, critic_hidden_layers=3 | `ppo_lr0p0001_crtH3_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, actor_hidden_layers=2 | `ppo_lr0p0001_act2_ep64_rand_mb32.pt` | 15.810% | 0.000% | 212.791% | 20/20 |
| lr=1e-4, actor_hidden_layers=4 | `ppo_lr0p0001_act4_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, reward norm=absmax | `ppo_lr0p0001_normAbs_ep64_rand_mb32.pt` | 15.905% | 0.000% | 214.679% | 20/20 |
| lr=1e-4, skip-trivial-cases | `ppo_lr0p0001_skipTriv_ep64_rand_mb32.pt` | 15.810% | 0.000% | 212.791% | 20/20 |
| lr=1e-4 + linear entropy schedule (0.05→0.001) | `ppo_lr0p0001_entlin0p05to0p001_ep64_rand_mb32.pt` | 5.171% | 0.000% | 103.419% | 20/20 |
| lr=1e-4, updates=50, ep/update=64 | `ppo_lr0p0001_ep64_rand_mb32_u50.pt` | 5.171% | 0.000% | 103.419% | 20/20 |

### çŸ­è¿°
- lr=6e-4 å¼ç»“æžœæ˜¾è‘—è¿œä¼¼ï¼å°ä¸èƒ½ä¼å¤„ç§ï¼ŒæŽ§åˆ¶è¯»å¼ã
- critic_hidden_dim ååŒ–ï¼128 vs 512ï¼åœ¨å½“åè®¾ç½®ä¸ä¸æŽ¨å¨è¯¡å¥½ã

è¯´æ˜Žï¼šæ¯ä¸ªå®žéªŒç‹¬ç«‹ç›®å½•ï¼Œé¿å…è¦†ç›–ï¼›å®žéªŒåŽæŠŠç»“æžœå†™å…¥ä¸Šè¡¨ã€‚


