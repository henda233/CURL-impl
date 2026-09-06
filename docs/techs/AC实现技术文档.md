# AC 实现 —— 技术文档

> 记录 AC 基线（on-policy Advantage Actor-Critic，DMControl walker-walk 像素）的选型、设计、接口与运行方式。源码 `src/ac/`；复盘见 `docs/notes/AC算法实现开发复盘.md`；网络规格见 `docs/reports/模型架构报告.md` §2.1/§3.1；超参锚点见文献 Table 3（walker 行）。

## 一、需求复述

需求《实现Actor-Critic算法.md》：AC 用于连续环境 walker-walk，按文献 100K 步训练；训练中定时保存模型到 `data/ac-当前时间/`（最新 + 最优两档）；训练后提供载入参数、渲染演示的测试代码；代码保存至 `src/ac/`；不需考虑跨算法复用，可复用相同 CNN；训练/测试超参见 Table 3。已与用户对齐三约束：**on-policy Advantage AC（V 头即 critic）**、**交付代码 + 短跑自检**（完整训练用户本地执行）、**训练无 random crop**（统一 center crop）。

## 二、总体设计与数据流

### 2.1 观测管线

100×100×3 uint8（`env.physics.render(height=100,width=100,camera_id=0)`）→ center crop (84,84,3)（offset=8，训练/评估一致）→ 3 帧栈沿通道拼接 → (9,84,84) float32 ×(1/255)。episode reset 以首帧复制填满栈；每决策块结束推入一帧（新帧位于通道尾）。

### 2.2 网络规格

`forward: (B,9,84,84) → μ(B,6), log_std(B,6), V(B)`

```
C_dmc: Conv(9→32,k3,s2)+ReLU → Conv(32→32,k3,s1)×3+ReLU → flatten 39200
h:     Linear(39200→1024)+ReLU
actor: Linear(1024→256)+ReLU → μ: Linear(256→6) | log_std: Linear(256→6, 置零初始化)
value: Linear(1024→256)+ReLU → Linear(256→1) → V(s)
```

actor 与 V 共享 C_dmc + h；log_std 前向 clamp 至 [-5,2]。

### 2.3 决策语义与计数口径

- **环境步** = `env.step`（physics step）；**决策块** = 同一 action 连续作用 `action_repeat=2` 次。
- 训练主计数 = 环境步，默认 `max_env_steps=100_000`（文献 DMControl100k）。
- 块 reward = 块内各环境步奖励之和；γ=.99 按块折扣；episode return 以未折扣和报告。
- episode 上限 = walker time limit 1000 环境步（≈500 决策块）；rollout 可被 `env_budget` 整块截断，截断尾部按终止处理。

### 2.4 损失与更新

每 episode 结束后更新一次（on-policy，样本只用一次）：

```
y = r + γ·V(s′)·(1−done)         # 末位/终止补 0；target 全程 detach
A = y − V(s)                       # advantage detach
L_V = mean((V(s) − y)²)
L_π = −mean(A · log π(a|s))        # update 时重算，μ/σ 带梯度可回传
loss = L_π + L_V − entropy_coef·H  # H：高斯熵近似；entropy_coef 默认 0
Adam(lr=1e-3) 一次 step；epochs 默认 1
```

动作分布为 tanh-squashed Gaussian：训练采样 `a=tanh(μ+σε)`，评估取 `tanh(μ)`；`log π(a|s)` 为高斯 log-prob 减 Jacobian 项 `log(1−a²)`（`atanh` 前对 a 做 ±(1−1e-6) 数值保护）。

## 三、模块与接口

| 模块 | 关键接口 |
|---|---|
| `networks.py` | `ActorCritic(act_dim=6)`：`forward(obs)`→(μ, log_std, V)；`tanh_normal_log_prob(mu, log_std, action)`；`gaussian_entropy(log_std)` |
| `pipeline.py` | `FrameStack.reset(frame)/push(frame)/state()`→(9,84,84) float32；`center_crop(frame)` |
| `train.py` | `make_env(seed)`、`render_frame(env)`、`select_action(model, state, explore)`→(6,) float64、`run_block(env, stack, action, repeat)`→(r, done, used)、`collect_episode(model, env, stack, repeat, budget)`、`update(model, opt, buffer, γ, entropy_coef)`、`evaluate(...)` |
| `eval_demo.py` | `--dir data/ac-*`（默认载 `best.pt`，`--latest` 切换），`--headless` 不弹窗 |
| `viewer.py` | `PyGameViewer(size_hw, title).show(frame_hwc_uint8)/close()`，pygame 延迟 import |

`train.py` / `eval_demo.py` 以脚本直跑，sys.path 含自身目录、同级模块直接 import；入口逻辑仅位于 `if __name__ == "__main__"`，模块可被安全 import（`eval_demo.py` 复用 `train.py` 的交互函数）。

## 四、训练循环与存档

主循环（`train.py main`）：

```
while env_steps < max_env_steps:
    buffer, ep_return, ep_env = collect_episode(..., budget=max_env_steps − env_steps)
    env_steps += ep_env；空 buffer 则 break
    for _ in range(epochs): update(...)
    if env_steps >= next_eval:
        evaluate(eval_episodes 局，mean action) → mean_return
        latest.pt 覆写；mean_return 创新高则 best.pt 覆写
        eval_history.jsonl 追加一行；next_eval += eval_interval
```

存档目录 `data/ac-YYYYmmdd-HHMMSS[-smoke]` 内三件：`latest.pt`、`best.pt`、`eval_history.jsonl`。checkpoint 结构：`{"model": state_dict, "meta": {seed, env_steps, episodes, mean_return, eval_episodes, action_repeat, entropy_coef}}`。

CLI 参数（`train.py`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--seed` | 0 | torch/np/env 随机种子 |
| `--max-env-steps` | 100000 | 训练环境步上限 |
| `--action-repeat` | 2 | 决策块跨度（Table 3 walker 行） |
| `--eval-interval` | 10000 | 评估/存档间隔（环境步） |
| `--eval-episodes` | 10 | 每节点评估局数（Table 3） |
| `--lr` / `--gamma` | 1e-3 / 0.99 | 学习率 / 折扣 |
| `--entropy-coef` | 0.0 | 熵正则系数（无温度 AC 默认 0） |
| `--epochs` | 1 | 每 episode 重复更新次数 |
| `--smoke` | off | 短跑覆盖：max=600、interval=600、eval=1 |

## 五、运行与实测

```
.venv\Scripts\python.exe src\ac\train.py --smoke                          # 短跑自检 ≈15s
.venv\Scripts\python.exe src\ac\train.py                                  # 完整 100K 环境步
.venv\Scripts\python.exe src\ac\eval_demo.py --dir data\ac-<时间戳>       # 弹窗演示（best.pt）
.venv\Scripts\python.exe src\ac\eval_demo.py --dir data\ac-<时间戳> --headless
```

smoke 实测（seed=0，.venv）：

```
[ac-train] ep=1 env_steps=600 ep_return=39.947 ep_env=600 pg=0.5339 vf=0.04703 ent=8.514
[ac-train] EVAL env_steps=600 mean_return=44.978 best=44.978 saved=...
[ac-train] DONE env_steps=600 episodes=1 best_mean=44.978 out=...
[ac-demo]  DONE env_steps=1000 blocks=500 return=16.777
```

退出码 0、无 warning；同 seed 两次运行数值逐字节一致（链路可复现）。随机/初始策略回报 ≈ 40–50 量级，作为判断学习是否突破的起点参照，非验收门限。

## 六、边界与可扩展点

- **center crop 固定**（无 random crop）：offset=8 硬编码于 `pipeline.py`，训练与评估一致；随机 crop 留待 CURL+X。
- **TD(0) 单步 advantage**，无 GAE/多步；截断 episode 尾部按终止处理（V′ 补 0）。
- **γ 按块折扣**：块内两次物理步不逐物理步打折，与 DMControl 像素基线 wrapper 惯例一致，非物理精确折扣。
- **无熵正则默认**：log_std 可学习；若长跑出现 σ 坍缩/探索不足，调 `--entropy-coef`（≈0.01）+ `--epochs`。
- **on-policy**：每 episode 一次更新、样本只用一次，无 replay buffer。
- **弹窗依赖显示设备**；`--headless` 供无头/CI。终端输出全 ASCII（规避中文 Windows 乱码）。
- C_dmc 卷积规格与后续 CURL 同构，但本目录不自建跨算法抽象；CURL+AC 将在 z(50) 表征之上另建网络与对比/EMA 机制。
