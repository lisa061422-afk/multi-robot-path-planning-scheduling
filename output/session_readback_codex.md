# Codex 页面记录（可在登录后随时读取）

最近一次会话里你让我确认的关键点（已落库）：

1. 训练是否分集成回合（episodic）
- 是的，训练是按 episode 收集轨迹，再做批量 PPO 更新（不是每一步立刻更新）。

2. 一次 episode 什么时候结束
- 在 `DecisionTreeEnv` 中，`reset()` 后进入 `while not terminated`，直到叶子节点 `is_leaf=True` 触发 `terminated=True`。
- 训练里还有安全上限：当 `len(episode_steps) >= max_decisions_per_episode`（默认 200）会抛错终止。

3. Actor / Critic 是什么
- Actor：给当前状态 `state` 和每个可行动作特征 `branch_actions` 计算分数（logits）。
  - 输入：`state`（全局状态）+ `branch_actions`（每条可选分支特征）
  - 结构：共享 MLP（`BranchScoringActor`）或轻量 GNN 分支打分（`BranchScoringActorGNN`）
  - 输出：每个合法分支一个 logit，经过 Categorical 采样/取 argmax
- Critic：输入状态 `state`，输出标量 V(s)（当前状态价值估计）。

4. 损失与训练
- Actor PPO 损失：裁剪目标比率 loss（`-min(r_t*A_t, clip(r_t)*A_t)` 的均值）
- Critic 损失：value MSE（`(V - target)^2`）
- 有熵正则：`entropy_coef * entropy`（鼓励探索）

5. RL 实际在学什么
- 学每个决策节点下的分支选择策略（本质是路径/调度联合树搜索中的分支决策）。
- 奖励定义为路径代价增量的负值，所以学习目标是降低最终总成本。

6. 是否在学习“最优解”
- 纯 PPO 不是直接监督最优解。
- 但有可选的 BC warm-start：可用 exact 最优决策轨迹做预训练（`--bc-pretrain-episodes`）。

其它你关心的“登录后可读”说明
- 当前训练/日志文件已在代码仓库更新并推送：`origin/agent/path-planning`。
- 这份记录文件会随着仓库同步；你登录后可直接读取到该文件内容。

当前提交：
- `6abbdde Refine PPO training/evaluation options and add contention-aware logging`

