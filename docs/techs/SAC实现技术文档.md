# SAC 实现 —— 技术文档

> 记录 SAC 基线（off-policy Soft Actor-Critic，DMControl walker-walk 像素）的选型、设计、接口与运行方式。源码 `src/sac/`；复盘见 `docs/notes/SAC算法实现开发复盘.md`；超参锚点见 CURL 论文 Table 3（walker 行）。

## 一、需求复述

需求《实现Soft Actor-Critic算法.md》：在 AC 基础上实现 SAC；保持训练/测试/保存设计；超参数按 Table 3；代码放 `src/sac/`。交付边界已在开发阶段与用户对齐：**完整 off-policy SAC**、**同构网络但参数各自独立（方案 B）**、**训练/评估统一 center crop（无 random crop）**、**温度 α 可学习**（init 0.1 / target −6）、**time-limit 截断按 truncated 处理**。网络规格见 `docs/reports/模型架构报告.md` §3.2。

## 二、总体数据流

```
帧(100,100,3)uint8 → center crop 84×84 → 3帧栈(uint8,9,84,84)
env.step×action_repeat → 块 reward / next 帧(3,84,84) → ReplayBuffer(cap 100K)
eval_interval 节点：独立 env mean action 评估 → latest/best + eval_history.jsonl
```

**决策**：walk 无动作时 time-limit=1000 env steps 触发 `last()`（无物理终止）。每收集一 action-repeat 块做一次更新：

```
sample batch(512) → α' = stop_grad(exp(α))
y = r + γ(1−done)(min(Q′_t1,Q′_t2) − α'·logπ′)        # done 恒 False（截断≠终止）
ℓ_Q = mean((Q1−y)²) + mean((Q2−y)²)                    # Q1,Q2 行为头；critic backward + soft_update
ã,logπ = reparam(π(.|s))                              # 带梯度
ℓ_π = mean(α'·logπ − min(Q1(s,ã),Q2(s,ã)))
ℓ_α = mean(−log α·(logπ + target_entropy)),  H0 = −6
梯度独立：optimizer critic(Q1,Q2 全部参数) / actor(独立) / α(log_alpha)
```

## 三、模块与接口

| 模块 | 关键接口 |
|---|---|
| `networks.py` | `Actor(act_dim=6)`：`forward(obs)→(μ,log_std)`（log_std clamp[-5,2]，零初始）；`Critic(act_dim=6)`：共享 body `forward(obs,action)→(Q1,Q2)`；`soft_update(target,source,tau)`；`reparam_sample(mu,log_std)→(a,logπ)`；`tanh_normal_log_prob` |
| `pipeline.py` | `FrameStack.reset/push/state()`→(9,84,84) uint8；`center_crop(frame)`→84×84；`obs_to_tensor(obs)`→float32×(1/255) |
| `buffer.py` | `ReplayBuffer(capacity,act_dim)`：`.add(obs,act,reward,next_frame_u8,done)` / `.sample(batch)→(obs,act,rew,next_obs,done)`；obs (9,84,84) uint8 + next 帧 (3,84,84)，sample 拼 `concat(obs[:,FRAME_C:], next_f)` |
| `train.py` | `run_block(env,action,repeat)→(r,last,used)`、`select_action(actor,state,sample)`→(6,) float64、`train_step(...)`（Q/actor/α + soft_update target）、`evaluate(actor,repeat,episodes,seed)`（独立 env）、`save_checkpoint` |
| `eval_demo.py` | `--dir data/sac-*`（默认 `best.pt`，`--latest` 切换），`--headless` 不弹窗 |
| `viewer.py` | `PyGameViewer(size_hw,title).show(frame_hwc_uint8)/close()`；pygame 延迟 import |

熵/分布工具沿用 tanh-squashed Gaussian：`tanh_normal_log_prob` 为高斯 log-prob 减 Jacobian `log(1−a²)`（atanh 前对 a 做 ±(1−1e-6) 数值保护）；重参数采样走 `reparam_sample`，使 μ/log_std 可回传。

## 四、网络规格（方案 B —— 参数各自独立）

报告 §3.2 SAC 图语义：actor 与 critic "共享 encoder"指架构同构(同一 C_dmc→h(1024))，**非**共享可训练参数。critic Q(s,a) 必须出现在 actor 目标里：若共享卷积参数，actor 那步 backward 会把策略梯度污染进 Q 头。故：

```
Actor(独立参数):
  body: Conv(9→32,k3,s2)+ReLU → Conv(32→32,k3)×3+ReLU → flatten 39200
        → Linear 39200→1024 + ReLU → Linear 1024→256 + ReLU
  p:    μ: Linear(256→6)；log_std: Linear(256→6, 置零 init, 输出 clamp[-5,2])

Critic(独立参数, 单实例双头):
  body: 同 Actor.body（另建参数）
  q1:    Linear(1030→256)+ReLU → Linear(256→1)      # 输入 concat(h(1024), a(6))
  q2:    同上独立权重实例

target critic = Critic() 同构实例；每 critic 更新后 soft_update(τ=0.01)
Adam 两个 lr=1e-3（actor / critic）；α 用 Adam(lr=1e-4, betas=(.5,.999))
```

forward 一次 actor / 一次 critic / 一次 target 三分量（每 collect 一次 train_step）。检查点只存 actor（meta 记温度实值等），供演示载入。

## 五、训练循环与存档

`train.py main()` 主循环（内层每块、外层每次 episode / 评估节点）：

```
while env_steps < max_env_steps:
    budget = (max_env_steps − env_steps) // action_repeat；≤0 break
    ts = env.reset()；stack.reset(渲染首帧)
    while (not ts.last()) and nblocks < budget:      # 一个 episode 或预算尾部
        state = stack.state()                          # (9,84,84) uint8
        if env_steps < initial_steps:  a = random_action()
        else:                          a = select_action(actor, state, sample=True)  # π 采样
        r_sum,last,used = run_block(env, a, action_repeat)
        frame = render_frame(env)；stack.push(frame)
        buffer.add(state, a.float32, r_sum, center_crop(frame).transpose(CHW), done=False)
        env_steps += used；ep_return += r_sum
        if env_steps ≥ initial_steps and len(buffer) ≥ batch_size:
            train_step(...)                            # Q + soft → actor → α，逐块
        if last: break
    episodes += 1；打印 ep 摘要
    if env_steps ≥ next_eval:                          # 在训练 episode 完全结束后触发
        returns = evaluate(actor, repeat, eval_episodes, seed)  # 独立 env，不打断
        mean = mean(returns)；存档 latest/best + eval_history 追加
        next_eval = min(env_steps + eval_interval, max_env_steps)
DONE 打印 best_mean/out；收尾 env.close()
```

存档目录 `data/sac-YYYYmmdd-HHMMSS[-smoke]` 内 `latest.pt`、`best.pt`、`eval_history.jsonl`。checkpoint 结构：`{"model": actor.state_dict(), "meta": {seed, env_steps, episodes, mean_return, eval_episodes, action_repeat, temperature, lr, alpha_lr, gamma, tau, batch_size, initial_steps}}`。

评估用**独立 env**（同 seed）：避免与训练侧共用 env 时评估 reset 打断正在进行的训练 episode（off-policy 逐块更新，评估节点可能落在 episode 中途）。

CLI 参数（`train.py`，默认值=Table 3 walker 行）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--seed` | 0 | torch/np/env 随机种子 |
| `--action-repeat` | 2 | 决策块跨度（Table 3） |
| `--max-env-steps` | 100000 | 训练环境步上限 |
| `--eval-interval` | 10000 | 评估/存档间隔（环境步） |
| `--eval-episodes` | 10 | 每节点评估局数（Table 3） |
| `--lr` | 1e-3 | actor/critic 学习率 |
| `--alpha-lr` | 1e-4 | log α 学习率 |
| `--gamma` | 0.99 | 折扣（按块） |
| `--tau` | 0.01 | target Q EMA 系数 |
| `--init-temperature` | 0.1 | α 初值 exp(log α) |
| `--initial-steps` | 1000 | 纯随机探索填充步 |
| `--batch-size` | 512 | 采样批大小 |
| `--replay-size` | 100000 | 环形 buffer 容量 |
| `--smoke` | off | 短跑覆盖（见下） |

smoke 覆盖：`max=120, initial=50, replay=1024, batch=16, eval_interval=120, eval_episodes=1`（把完整 warmup+batch512 压到秒级，仍走通"收集→更新→eval→存档"）。

## 六、运行与实测（seed=0，.venv CPU）

```
.venv\Scripts\python.exe src\sac\train.py                          # 完整 100K 环境步
.venv\Scripts\python.exe src\sac\train.py --smoke                  # 短跑自检 ≈ 30s
.venv\Scripts\python.exe src\sac\eval_demo.py --dir data\sac-<ts>            # 弹窗演示
.venv\Scripts\python.exe src\sac\eval_demo.py --dir data\sac-<ts> --headless # 无头
```

smoke 实测（seed=0）：

```
[sac-train] ep=1 env_steps=120 ep_return=10.701 q=0.2982 pi=-1.036 temp=0.09982
[sac-train] EVAL env_steps=120 mean_return=18.455 best=18.455 saved=...
[sac-demo]  DONE env_steps=1000 blocks=500 return=18.455
```

退出码 0、无 warning；同 seed 重跑逐字节一致。CPU 单次 SAC train_step 约 batch64≈0.84s / batch512≈4.06s（三份卷积网络），全量 100K 明显长于 AC，可据机器预算调 batch／replay。随机/初始策略回报 ≈40-50，判断学习以 `EVAL mean_return` 上行突破该基线为准。

## 七、边界与可扩展点

- **center crop 固定**（无 random crop）：offset=8 硬编码于 pipeline，训练/评估一致；随机 crop 留待 CURL 阶段。
- **time-limit 截断 ≠ 终止**：walker time-limit 恰 1000 env steps 无物理终止，`done` 恒 False；代码保留 done 通道，供其它域区分 terminated/truncated 时置 True 不 bootstrap。
- **target Q 软更新（soft EMA τ=0.01）**：作用在 critic 全部参数（含 encoder body），非定期 hard copy。
- **α 自动调节**：目标熵 −6（=−act_dim），log α 逐块优化，默认不固定温度。
- **重参数策略梯度**：ã=tanh(μ+σε) 把梯度转移到可导的 μ/σ，无 score-function 方差问题。
- **弹窗依赖显示设备**；`--headless` 供无头/CI。终端输出全 ASCII（规避中文 Windows 乱码）。
- C_dmc 卷积与 AC/CURL 同构，不自建跨算法抽象；CURL+SAC 将在 z(50) 表征上另建 actor/critic 头 + 对比/EMA 机制。

## 八、正确性审查要点录（实现期自查）

1. next_obs 拼接用 obs 后 2 帧 + next 新帧（共 3 帧），勿按 STACK_C 取 6 帧 — 已由 `FRAME_C:` 常量规避并 smoke 通过。
2. Q 值参与 actor 目标，critic/actor 独立 body 防梯度污染（方案 B 结构保证）。
3. α 对 Bellman 目标 stop-grad：`y` 用 `alpha.detach()`，温度仅由 `ℓ_α` 训练。
4. 日志 float 均 `.detach()`，避免 `float(tensor_with_grad)` 的 UserWarning。
5. eval 用独立 env，不打断训练 episode；训练侧 env/stack 在自身 episode 内干净推进。
