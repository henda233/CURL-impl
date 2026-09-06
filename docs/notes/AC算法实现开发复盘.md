# AC 算法实现 —— 开发复盘

## 一、开发内容概述

### 1.1 任务背景与目标

本阶段按需求文档《实现Actor-Critic算法.md》交付 RL 基线 **AC**：作用于 DMControl 连续控制 `walker/walk`（walker-walk）的像素观测。硬性要求：按文献 100K 步训练；训练中定时把模型存到 `data/ac-当前时间/`（同时维护"最新"与"最优"两档）；训练后提供载入参数、渲染演示的测试代码；代码放 `src/ac/`；不必考虑跨算法复用，可复用相同卷积架构；超参数见文献 Table 3。

### 1.2 与用户对齐的三项决策

1. **算法形态 = on-policy Advantage AC（V 头即 critic）**：弃 off-policy 无温度 SAC 形态；V 作 critic，TD(0) advantage `A = r + γV′ − V` 同时作策略梯度权重与 critic 目标，无 replay/EMA/warmup。
2. **交付边界 = 代码 + 短跑自检**：完整 100K 训练由用户本地执行；本阶段用短跑验证"训练→更新→评估→存档→载入→演示"整链路。
3. **输入增强 = 无 random crop**：训练与评估统一 center crop 84×84；随机 crop 属文献 CURL 对比学习配套，留待 CURL+X 阶段，避免把增强变量混入 AC 基线。

### 1.3 交付物与验收口径

`src/ac/` 五个文件，零新增第三方依赖：

| 文件 | 职责 |
|---|---|
| `networks.py` | 共享 C_dmc + h(1024)，actor(μ, log_std) 与 V 头分叉；tanh-squash 分布工具 |
| `pipeline.py` | center crop + 3 帧栈 + ×1/255 → (9,84,84) float32 |
| `train.py` | on-policy AC 主循环 + eval/checkpoint；CLI 全参 + `--smoke` |
| `eval_demo.py` | 载入 `best.pt`/`latest.pt`，mean action 渲染演示（`--headless` 无头自检） |
| `viewer.py` | pygame 弹窗逐帧渲染（延迟 import，无头安全） |

验收（AGENTS 数据导向）：短跑自检输出确定数值、退出码 0、存档三件套齐全、载入演示跑通；同 seed 两次运行逐字节一致。

## 二、开发决策

### 2.1 网络：共享卷积与特征层，actor/V 头分叉

照架构报告 §3.1 与论文附录"actor and critic share the same encoder"：

```
obs(9,84,84) → C_dmc(39200) → Linear 39200→1024 + ReLU = h(1024)
  actor: p_head(1024→256+ReLU) → μ(6) | log_std(6，weight/bias 显式置 0)
  value: v_head(1024→256+ReLU → 256→1) = V(s)
```

log_std 前向 clamp 至 [-5,2] 防发散；actor 与 V 的梯度经共享 body 同一次 backward 相加（A2C 常规）。

### 2.2 动作分布：tanh-squashed Gaussian

动作界 [-1,1]。训练采样 `a = tanh(μ + σε)`，评估取 `tanh(μ)`。策略梯度所需 `log π(a|s)` 用闭式 squashed-Gaussian：高斯 log-prob 减 Jacobian 项 `log(1−a²)`，`atanh` 前对 a 做 ±(1−1e-6) 数值保护。rollout 每决策块一次 no-grad forward 得分布；update 时**重算** `log π(a|s;θ)`（a 固定、μ/σ 带梯度）保证策略梯度可回传。

### 2.3 时间口径与块折扣

术语：**环境步** = `env.step`（physics step）；**决策块** = 一个 action 连续作用 `action_repeat=2` 次。主循环按环境步计数，默认 `max_env_steps=100_000`（文献 DMControl100k；Table 3 walker 行 repeat=2）。一条 TD transition 的 reward = 块内各环境步奖励之和，γ=.99 按块折扣（DMControl 像素基线 wrapper 惯例）；episode return 以未折扣和报告（DMControl 惯例）。walker episode time limit = 1000 环境步 → 每 episode ≈ 500 决策块；rollout 受 `env_budget` 整块截断，截断尾部视作终止。

### 2.4 更新节律：每 episode 一次 TD(0) 更新

100K 环境步 ≈ 200 个 episode。on-policy 样本只用一次：每 episode 结束后对其全部 transition 做一次梯度步（`epochs=1` 默认，可调多 epoch 复用）。critic 目标 `y = r + γV′·(1−done)`（末位/终止补 0）；target 与 advantage 全程 detach，`L_π = −mean(A·log π)` 与 `L_V = mean((V−y)²)` 分别回传。默认无熵正则（`entropy_coef=0`，贴合"无温度 AC"语义）；σ 坍缩可调 `--entropy-coef`。

### 2.5 超参数映射（Table 3 → AC 基线，walker 取值）

采用：action repeat 2、渲染 100×100 / center crop 84×84、3 帧栈、hidden 1024、eval episodes 10、Adam(.9,.999)、lr 1e-3（表中 otherwise；walker 非 cheetah）、conv 4 层 32 filter ReLU、γ=.99。
弃用：random crop（用户决策）、replay 100k / initial steps 1000 / batch 512 / Q-EMA τ=0.01（off-policy 机制项）、α 温度（lr 1e-4 / init 0.1）、latent 50 与 encoder EMA（CURL 专属）。

### 2.6 存档与"最新/最优"语义

每 `eval_interval`（默认 10K 环境步，100K 共约 10 节点）用 mean action 评估 `eval_episodes`（默认 10，Table 3）集：`latest.pt` 每节点覆写；`best.pt` 仅平均回报创新高时覆写；`eval_history.jsonl` 每节点追加一行 meta（seed/env_steps/mean_return/…）供画曲线。目录 `data/ac-YYYYmmdd-HHMMSS`，`--smoke` 加后缀；`data/` 已入 `.gitignore`。

### 2.7 先探后写、短跑即回归

写码前在真实 `.venv` 探测（临时脚本跑完即删）三事实：walker-walk time limit = **1000 环境步**（zero action 恰 1000 步触发 `last()`）；step+render ≈ **7.9 ms/步**（短跑定 600 环境步 ≈ 秒级）；action spec `(6,)` 界 [-1,1]、`step` 需 float64（tanh 输出转 float64 提交）。短跑自检同时作回归守卫。

## 三、实测结果（短跑自检）

### 3.1 训练链路

```
.venv\Scripts\python.exe src\ac\train.py --smoke
[ac-train] cfg seed=0 max_env_steps=600 action_repeat=2 lr=0.001 gamma=0.99 entropy_coef=0 epochs=1 ...
[ac-train] ep=1 env_steps=600 ep_return=39.947 ep_env=600 pg=0.5339 vf=0.04703 ent=8.514
[ac-train] EVAL env_steps=600 mean_return=44.978 best=44.978 saved=...
[ac-train] DONE env_steps=600 episodes=1 best_mean=44.978 out=...
```

600 环境步 → 1 次 update → 1 次 eval（完整 1000 环境步 episode）→ `latest.pt`/`best.pt`/`eval_history.jsonl` 落盘，退出码 0、无 warning；同 seed 重跑数值一致（39.947 / 44.978）。随机/初始策略回报 ≈ 40–50 量级，是判断"是否学会"（曲线突破随机基线并上升）的起点参照，非学习验收门限。

### 3.2 演示链路

```
.venv\Scripts\python.exe src\ac\eval_demo.py --dir data\ac-...-smoke --headless
[ac-demo] loaded ... best.pt meta=...
[ac-demo] DONE env_steps=1000 blocks=500 return=16.777
```

160 MB state_dict 保存/载入 roundtrip 正常，mean action 完整走完一个 episode。弹窗路径依赖显示设备，本轮未在无头环境验证（viewer 为冒烟阶段已验证实现的复刻）。

## 四、教训与经验

- **分歧先拍板再动手**：架构报告给 V 头、算法/环境表称"Pixel SAC 范畴"，critic 形态存在真实冲突；先用 3 个必答问题锁定机制/交付边界/增强，随后单路径实现，零返工。
- **"100K 步"必须拆成可执行口径**：环境步与决策块差 2 倍，不说清会导致训练时长与存档语义错位；本项目固定"环境步计数 + 块折扣"并写入文档。
- **认识 on-policy 的样本现实**：100K 环境步 ≈ 仅 200 次 episode 级更新，学习信号稀薄是 AC 基线的固有代价，也恰是 CURL（对比学习提样本效率）的对照点；不为此偷换 off-policy 机制。
- **长任务责任切分**：CPU 完整训练数小时级，以"短跑 = 确定性交付、长跑 = 用户执行 + 命令行说明"切分边界最干净。

## 五、后续衔接（用户本地运行）

```
.venv\Scripts\python.exe src\ac\train.py                                  # 完整 100K 环境步
.venv\Scripts\python.exe src\ac\train.py --entropy-coef 0.01 --epochs 4   # 探索不足时调参
.venv\Scripts\python.exe src\ac\eval_demo.py --dir data\ac-<时间戳>       # 弹窗演示（默认 best.pt）
```

判断学习效果：盯训练日志 `EVAL ... mean_return`（或 `eval_history.jsonl`），对比随机基线 ~40–50；长时间不涨优先调 `--entropy-coef`/`--epochs`/`--lr`。完整训练后的 `best.pt` 即演示载入目标。进入 CURL 阶段时，本基线（无 crop、无温度、on-policy）与 CURL+AC 的差异即对比学习引入的净收益，变量隔离干净。
