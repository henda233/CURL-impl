# SAC 算法实现 —— 开发复盘

## 一、开发内容概述

### 1.1 任务背景与目标

本阶段按需求文档《实现Soft Actor-Critic算法.md》交付 RL 基线 **SAC**：作用于 DMControl 连续控制 `walker/walk`（walker-walk）像素观测。需求：在 AC 基础上实现 SAC、保持训练/测试/保存设计、超参按原文 Table 3、代码放 `src/sac/`。

对照需求做了一次关键澄清：AC 是 on-policy + 无温度形态，需求笼统说"只需改模型"并不成立——replay buffer、random exploration、双 Q、target Q EMA、自动温度正是 AC 阶段按 on-policy 语义弃用的 off-policy 机制项，实现 SAC 必须恢复。故交付 = **AC 同款骨架（100K 环境步、`data/sac-<时间戳>/`、latest/best 双档、eval_demo、smoke） + 完整 off-policy SAC**。

### 1.2 与用户对齐的五项决策（文字问答，未用 ask 工具）

1. **机制范围 = 完整 off-policy SAC**：replay buffer + initial random exploration + 双 Q + target Q EMA + 自动温度，而非 on-policy 字面"只改模型"。
2. **输入增强 = 无 random crop**：与 AC 完全一致（训练/评估统一 center crop 84×84），随机 crop 属 CURL 对比学习配套，留待 CURL 阶段，避免把增强变量混入 SAC 基线。
3. **网络 = 同构共享、参数分离（方案 B）**：actor 与 critic 各自持有同构 C_dmc→h(1024) 参数，**架构相同但参数各自独立**（文档语义如此）。critic 为双 Q 独立头。
4. **温度 α = 可学习**：log α 由 Adam(0.5, 0.999) lr 1e-4 优化，init exp(0.1)，目标熵 −act_dim=−6（照 Table 3 Initial temperature 0.1 / lr(α) 1e-4）。
5. **其余默认 = 确认 AC 移植**：time-limit 截断视为 truncated（Q 继续 bootstrap，不置 done）、每 action-repeat 块一次更新、存档/演示/smoke 设计照 AC。

### 1.3 交付物与验收口径

`src/sac/` 六个文件，零新增第三方依赖：

| 文件 | 职责 |
|---|---|
| `networks.py` | 方案 B 三体：Actor（body+μ/log_std）、Critic（共享自建 body+双 Q 头，concat(h,a)）、soft_update EMA、tanh-squash 工具 |
| `pipeline.py` | center crop + 3 帧栈 **uint8**（省内存供 buffer 存）、`obs_to_tensor` 归一化 |
| `buffer.py` | 环形 ReplayBuffer（capacity 100K），obs 帧栈 (9,84,84) uint8 + next 新帧 (3,84,84)，sample 时拼成 next_obs |
| `train.py` | off-policy SAC 主循环 + eval/checkpoint；CLI 全参 + `--smoke` |
| `eval_demo.py` | 载入 `best.pt`/`latest.pt`，mean action 渲染演示（`--headless` 无头自检） |
| `viewer.py` | pygame 弹窗逐帧渲染（延迟 import，无头安全） |

验收（AGENTS 数据导向）：短跑自检输出确定数值、退出码 0、存档三件套齐全、载入演示跑通；同 seed 两次运行逐字节一致。

## 二、开发决策

### 2.1 网络三体（方案 B 落地）

架构报告 §3.2 的 SAC 结构与论文 "actor and critic use the same encoder" 都不要求共享可训练参数；相反，**共享会引入污染**——critic Q 值出现在 actor 损失里（`L_π = α log π(ã) − Q(s,ã)`，ã 重参数采样），若 actor 与 Q 头共用卷积参数，actor 那步 backward 会把策略梯度送进 Q 头。故按方案 B 参数各自独立：

```
Actor:   body: (9,84,84)→C_dmc(39200)→1024+ReLU→256+ReLU → μ(6)|log_std(6)
Critic:  自建 body 同 actor；其上 Q1/Q2 独立头：concat(h(1024), a(6))=1030→256→1
target = Critic 的同构实例，经 soft_update(τ=0.01) 由 critic 拉近
log_std 前向 clamp [-5,2]，置零初始化（σ=1）
```

Adam 两个（lr 1e-3）：actor 参数一组、critic 参数一组（target 只被 EMA 更新不随优化器）。

### 2.2 架构选型定性：为何体量是 AC 的 ~3 份

AC 一个共享 body + 双头；SAC 是 actor + critic + target/critic 三份网络体量。尽管检查点只存 actor（供演示），训练期显存/CPU 均三方同驻。内存与算力开销显著大于 AC（实测见 §3/三）。

### 2.3 动作分布与重参数梯度

沿用 AC 的 tanh-squashed Gaussian。重参数采样 `u = μ + σε; a = tanh(u)` 直接带梯度：`reparam_sample(mu, log_std) → (a, logπ)`。策略梯度经 ã 回传（reparam，非 REINFORCE score function），不需对 log π 单独 `detach`——actor loss `α logπ(ã) − minQ(s,ã)` 中 α logπ 项对 log_std 保梯度以调整探索。

### 2.4 决策语义与计数口径（同 AC）

环境步 = `env.step`；决策块 = 一个 action 作用 action_repeat=2 次；块 reward = 块内奖励和；γ=.99 按块折扣；100K 环境步上限；walker time-limit = 1000 env steps（探针实测 zero-action 恰 1000 步触发 `last()`，无真终止）。

### 2.5 truncated 处理：与 AC 的差异

AC 是 on-policy，TD 末位按终止补 0。SAC off-policy 下 time-limit 截断不置 done：`y = r + γ(1−done)(minQ′ − α log π′)`，永不把截断当终止——让 Q 继续 bootstrap。由于 walker 从无物理终止，`done` 恒 False，代码仍保留 done 通道（截断/终止语义区分），为 buffer 通用性留位。

### 2.6 更新节律与 replay

每收集完一个 action-repeat 块就做一次更新（on-policy 的"每 episode 更新"在 off-policy 下替换为"每块更新"）：从 replay 抽 batch（默认 512）→
① 双 Q loss ℓQ = ((Q₁−y)²)₁ + ((Q₂−y)²)₂ → update critic → soft_update target；
② actor loss Lπ = mean(α logπ(ã) − min(Q₁,Q₂)) → update actor（reparam 梯度）；
③ 温度 loss ℒα = −mean(log α·(logπ(ã)+H₀))，H₀=−6 → update α；
温度 agent 端取 `exp(logα).detach()`（stop-grad），不带入 Bellman/actor 目标。

`initial_steps=1000`（Table 3）内纯随机填充 buffer 不更新；随机充分探索覆盖初始 episode。

### 2.7 观测管线与 buffer 的 uint8 压缩

与 AC 同款 **center crop 84×84**（去掉 8px 边框、训练/评估一致）+ 3 帧栈；SAC 需存大量过渡样本，buffer 里 obs 帧栈存 **uint8 (9,84,84)**、另存 **next 新帧 (3,84,84)**。sample 时 `next_obs = concat(obs[:,FRAME_C:], next_f)`（丢最旧 3 通道、压入新帧），避免整份双倍重复存储，采样前 `obs_to_tensor` ×(1/255) 归一。100K 容量 uint8 仍约 (9+3)·84·84×2×100K ≈ 17 GB → 提供 `--replay-size` 可调、CPU 上建议小容量。

### 2.8 存档与"最新/最优"语义（同 AC）

每 `eval_interval`（默认 10K 环境步）用独立 eval env + mean action 评估 `eval_episodes`（默认 10）局：`latest.pt` 每节点覆写、`best.pt` 仅创新高时覆写、`eval_history.jsonl` 每节点追加一行 meta。目录 `data/sac-<时间戳>`；`--smoke` 加后缀。**checkpoint 只存 actor 网络**（架构 body 仅 actor 一份 + meta），供演示载入即走。

### 2.9 评估用独立 env，不打断训练 episode

AC 里 collect/eval 共用 env——on-policy 下两者都在 episode 边界，天然互不污染。SAC 每块更新后随时可能命中 eval_interval，若共用 env 去评估会把当前训练 episode 的中途轨迹打断、并以评估 reset 污染收集侧。改为 `evaluate()` 每次开闭**独立 suite env（同 seed）**，两者解耦，训练侧 env 保持自身 episode 完整性。

## 三、实测结果（短跑自检）

### 3.1 环境/算力探针

- walker-walk：zero-action 恰 **1000 env steps** 触发 `last()`（纯 time-limit，无真终止）；action (6,)，界 [-1,1] float64；渲染 (100,100,3) uint8。
- CPU 单次 SAC train_step（3 份卷积体）：batch64 ≈0.84s、batch512 ≈4.06s——据此把 smoke 压到 batch16 秒级。

### 3.2 训练链路 smoke（seed=0，max=120 / init=50 / batch=16 / replay=1024）

```
[sac-train] ep=1 env_steps=120 ep_return=10.701 ep_env=120 q=0.2982 pi=-1.036 temp=0.09982
[sac-train] EVAL env_steps=120 mean_return=18.455 best=18.455 saved=...
[sac-train] DONE env_steps=120 episodes=1 best_mean=18.455 out=...
```

120 env steps → 一次收集 episode（前半 warmup，后半更新）→ 一次 eval → 三件套落盘，退出码 0、无 warning。同 seed 重跑数值完全一致（10.701 / 0.2982 / 18.455 / 0.09982）。

### 3.3 演示链路（seed=0，载 best.pt，headless）

```
[sac-demo] DONE env_steps=1000 blocks=500 return=18.455
```

checkpoint（~160MB actor state_dict）roundtrip 正常，mean action 完整走完一个 episode。弹窗路径依赖显示设备，本轮未在无头环境验证 viewer。

## 四、迭代中修正的真实 bug（复盘重点）

1. **buffer 帧通道序 bug**：首版 `next_f` 存 HWC 撞 `(84,84,3)` 形状——定 next_f 存 center_crop 后转 **CHW (3,84,84)** uint8，与帧栈通道序一致。
2. **next_obs 拼接索引 bug**：误用 `obs[:, STACK_C-FRAME_C:]`（得 6 通道）——实应保留 `obs[:, FRAME_C:]`（末两帧，丢最旧 3 通道并入新 3），否则 next 观测 6 通道撞 Conv 首层 9 通道。以字面 `FRAME_C:` 常量规避。
3. **eval/env 状态耦合**：初版用训练 env 评估会打断当前训练 episode 并 reset 污染收集侧——改独立 eval env 修正。

## 五、经验与衔接

### 5.1 经验

- **"先探后写"再次生效**：CPU train_step 算力探针直接圈定了 smoke 的步数上限，避免超时空转；walker `last()` 只在 time-limit 出现，确认 SAC 的 truncated 处理必要性。
- **"架构相同≠参数共享"落实到 clean code**：共享 Q 进入 actor 损失梯度污染是像素 SAC 的经典坑，方案 B 从结构上根除，代价是 3 倍体量与更长训练时长（本机无 N 卡，CPU）。
- **uint8 压缩 buffer 是像素 off-policy 隐性刚需**：不压缩则 100K replay 双份 RGB 图达数十 GB，sample 亦慢；单帧增量存储把推理拼接集中到一处采样时，内存减半且不易错（代价 next 观测在 sample 现算）。

### 5.2 后续衔接（用户本地运行）

```
.venv\Scripts\python.exe src\sac\train.py            # 完整 100K 环境步（默认 Table3 超参）
.venv\Scripts\python.exe src\sac\train.py --smoke    # 回归自检（秒级）
.venv\Scripts\python.exe src\sac\eval_demo.py --dir data\sac-<时间戳>            # 弹窗演示 best.pt
.venv\Scripts\python.exe src\sac\eval_demo.py --dir data\sac-<时间戳> --headless # 无头自检
```

注意：SAC 每 env 步一次 batch=512 的 3× 卷积前向，CPU 全量训练明显长于 AC（约数小时~10 小时量级）；可 `--batch-size 128/256` 提速、按内存调 `--replay-size`。判断学习看 `EVAL mean_return` 上行突破随机基线 ~40-50（Table 3 口径下 SAC 像素样本效率应显著高于 AC）。进入 CURL 阶段时，SAC 基线的 replay/target-EMA/温度/同构 encoder 即 CURL+SAC 的差异基线（仅增表征尾巴 + 对比/EMA），变量隔离干净。
