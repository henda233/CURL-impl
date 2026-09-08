# CURL 实现 —— 技术文档

> 记录 CURL（CURL + SAC，方案 A）的选型、设计、接口、运行方式与 GPU 数值失稳的诊断/修复。源码 `src/curl-sac/`；复盘见 `docs/notes/CURL算法实现开发复盘.md`；网络规格锚定 `docs/reports/模型架构报告.md` §4/§5.2；机制与超参以 CURL 论文为准（本地全文 `docs/paper/paper.md`）。

## 一、需求复述

需求《实现CURL + SAC算法.md》：实现 CURL+SAC 作为**独立算法**（代码 `src/curl-sac/`，平行复制 SAC、不做代码共享）；重点为 CURL 表征编码器；架构、对比学习设计、超参以论文为主（含 4.7 伪代码）；不考虑 AC（on-policy）。已对齐决策：**方案 A（全链路共享单一 f_q）**、**联合步（Bellman+InfoNCE 等权一次 backward/step）**、**actor 不回传 f_q**、**100×100 原帧入 buffer**、**训练 random crop / 评估 center crop**、EMA **target critic τ=0.01 与 key f_k τ=0.05 两套**、存档 `{f_q, actor_head}`、CLI 全套 + `--diag`（见第六节）。

## 二、总体数据流与网络归属（方案 A）

```
渲染 100×100 → 3 帧栈 uint8 (9,100,100) 入 buffer（不预裁）
决策：random_crop(100→84) → Enc → actor_head → 动作；评估路径 center_crop
batch 采样 → random_crop 三份视图（obs_a / obs_k / next_a，栈内同一窗口坐标）
```

共享编码器 `f_q = Enc(θ_enc)`：C_dmc(9,84,84)→39200 → Linear(39200→1024)+ReLU → Linear(1024→50) → LayerNorm(50) → tanh ⇒ **z(50)**。actor 头与 critic 双 Q 头建于 z 之上；f_q 同时服务 RL 与 InfoNCE。每收集一个 action-repeat 块做一次更新：

```
① 联合步：z_a=Enc(obs_a)；z_k=f_k(obs_k).detach()            # f_k 为 key 编码器(EMA)
   y = r + γ(1−done)(min(Q′_t1,Q′_t2) − α′·logπ′)            # Q′ 用 target_encoder 的 next 表征
   ℓ_Q = Σ mean((Q_i(z_a,a)−y)²)                            # critic 头建于 z_a
   logits = z_a·(W·z_kᵀ)（W 为 (50,50) bilinear，行 max 稳定化）
   ℓ_cpc = CrossEntropy(logits, labels=arange(B))            # 对角为正样本对，batch 内其余为负
   ℓ = ℓ_Q + ℓ_cpc 一次 backward，step（θ_enc + Q 头 + 投影头 + W 同优化器，等权无系数）
   soft_update(target_encoder, Enc, τ=0.01)；soft_update(target_critic, critic, τ=0.01)
   soft_update(f_k, Enc, τ=0.05)
② actor 步：z_a.detach()；ã,logπ = reparam(actor_head(z_a))；ℓ_π = mean(α′·logπ − min(Q1,Q2))
   仅 step actor_head（不回传 f_q）
③ 温度步：ℓ_α = mean(−log α·(logπ+H₀))，H₀=−6，Adam(lr 1e-4, betas=(.5,.999))
```

next 动作 a2/logπ′ 由 online 组件（Enc + actor_head 对 next_a 采样）生成，target Q 值由 target_critic + target_encoder 计算（next 表征编码两遍，语义同 SAC 基线 target 结构）。

## 三、模块与接口

| 模块 | 说明 |
|---|---|
| `networks.py` | `Encoder`(f_q)、`ActorHead`(z→μ/log_std，log_std clamp[-5,2] 零初始)、`Critic`(concat(z,a) 双 Q 头，不含 Enc)、`BilinearSimilarity`(W=rand(50,50))、`_Temperature`、`soft_update`、`reparam_sample`（噪声形式 logp，见 §6.4） |
| `pipeline.py` | `FrameStack`（保留 100×100 原帧，`state()`→(9,100,100) uint8）；`random_crop`（(C,100,100) 单窗口 / (B,C,100,100) 每样本一窗口）；`center_crop`；`obs_to_tensor`（uint8 直传 device ÷255） |
| `buffer.py` | `ReplayBuffer(cap,act_dim)`：obs 帧栈 (9,100,100) uint8 + next 新帧 (3,100,100)，sample 拼 `concat(obs[:,3:], next_f)` |
| `train.py` | `train_step(...)`（联合步→双 EMA→actor→α；`diag` 可选旁路参数）、`select_action(encoder,actor,state,sample,crop)`、`evaluate`、`save_checkpoint`；CLI 见第五节 |
| `eval_demo.py` | 载 checkpoint 的 `{encoder, actor}`，center crop 演示 |
| `log_tee.py`/`viewer.py`/`profiler.py` | src/sac 原样副本（tee 全量日志 / pygame 弹窗 / 分段计时） |

## 四、训练循环、存档与 CLI

循环/存档结构同 SAC 基线：`data/curl-sac-<时间戳>[-smoke]/`，`latest.pt`/`best.pt`/`eval_history.jsonl`/`train_log.txt`；每 eval 节点独立 env + mean action（center crop）评估；walker time-limit 截断一律按 truncated（done 恒 False，Q 继续 bootstrap）。checkpoint：`{"model": {"encoder": state_dict, "actor": state_dict}, "meta": {...}}`（推理最小集合，demo 载入即走）。

CLI（默认值 = 论文 Table 3 walker 行 + 诊断开关）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--seed/--action-repeat/--max-env-steps/--eval-interval/--eval-episodes` | 0 / 2 / 100000 / 10000 / 10 | 同 SAC |
| `--lr/--alpha-lr/--gamma/--tau/--init-temperature/--initial-steps/--batch-size/--replay-size` | 1e-3 / 1e-4 / .99 / .01 / 0.1 / 1000 / 512 / 100000 | 同 SAC；key EMA 固定 `KEY_TAU=0.05`（硬编码常量） |
| `--gpu` | off | 显式启用 CUDA，不可用则 parser.error |
| `--smoke` | off | 覆盖 max=60, initial=20, replay=256, batch=8, eval_interval=60, eval_episodes=1 |
| `--profile` | off | 覆盖 max=2000、跳过 eval、写 profile.json（分段计时插桩沿用） |
| `--diag` | off | 每更新步旁路记录 TD/actor/表征/InfoNCE 数值 → `update_diag.jsonl`（见第六节） |

## 五、本机自检（seed=0，CPU smoke）

```
ep=1 env_steps=60 ep_return=6.417 q=0.3163 curl=2.081 pi=-1.443 temp=0.09989
EVAL mean_return=23.847
```

同 seed 两次逐字一致（确定性，random crop 用 `np.random`，与 `torch.manual_seed` 同步固定）；curl≈ln8 与 batch=8 的 InfoNCE 初始期望相符。修复 logp 求值（§6.4）后 q/curl/temp 与修复前逐位一致、pi 与 EVAL 仅尾数级微差（正常区数值等价佐证）。`--gpu` 无 CUDA 报 parser.error；`--profile` 通路 profile.json 正常。完整训练交给 GPU（见第六节，曾发散）。

## 六、GPU 数值失稳：现象、诊断与修复（2026-09）

### 6.1 现象（两次 GPU 全量训练）

首跑（seed=10086，batch 512，默认 cfg）：前 36k env steps 健康（q 缓升 ~1、curl 0.2–0.4、EVAL 曾到 52.021@30k）；ep37（env 36k–37k）内数值灾难：q→5.3e20 量级并继续放大、`curl` 跳高后**恒等于 6.238 = ln(512)**（InfoNCE 完全坍缩判据）、ep_return 回落到 ~13，env ~42022 由用户手动中断（产物 `data/curl-sac-20260908-145018/`）。

复跑（seed=0，带 `--grad-clip 10 --diag`，验证 clip 能否阻止崩溃）：崩溃推迟到 env ~45400（更新步 22675 起 logp tail 爆发），ep 均值在崩溃段内仍被掩盖（ep46 汇总正常），ep47 起 q→1e19 级并继续放大、`curl`→6.238、μ 指数爬升 2201→23759、`log_std` 钉在 clamp 下界 −5，env ~50k 手动中断（产物 `data/curl-sac-20260908-160421/`）。**clip 只推迟不消除崩溃**；且健康期联合步梯度范数 p50=18.3 > 10、`clip_main` 触发率 75.6%——clip=10 全程压制约 3/4 的更新，已实质扭曲训练。

### 6.2 诊断工具 `--diag`

`train_step` 旁路记录每更新步的量到 `out_dir/update_diag.jsonl`（不改变数值语义，默认关闭）。字段分组：TD/Q（`y_mean/y_absmax/qmin_absmax/tqmin_absmax`）、actor（`logp2_min/logp_min/mu_absmax/logstd_absmax`）、表征/InfoNCE（`z_std_mean`=batch 内每维 std 均值、`logits_spread`=行内 max−min 均值、`logits_absmax`）；`z_std_mean→0` 或 `logits_spread→0` 即坍缩判据，`curl` 恒 ln(B) 即 InfoNCE 退化。

### 6.3 归因（两次 diag 逐点数据，已证实）

两次崩溃同构（145018 共 20500 步 / 160421 共 24000 步）：
- **首个异常量是 actor 采样 logp**：先出现间歇 tail（145018 step 17223 `logp2_min≈−300`；160421 step 22675 起 −50→−1e3→−1e13 级持续加深），当时 Q/y/表征均健康；
- 崩溃前 `mu_absmax` 已越出 tanh 线性区（160421 step 21760 首破 9 = float32 下 tanh 饱和点 u≈9.01）、`log_std` 收缩至 clamp 下界 −5；μ 随后指数爬升（2201→23759）与 logp 无界加深同步；
- `tqmin_absmax` 全程未爆（~80–90），`qmin_absmax` 恶化与表征坍缩（`z_std_mean` 0.49→~0、`logits_spread`→0）均在 logp 爆发约百步之后 → **Q/target 未自发散、InfoNCE 坍缩是崩溃的结果而非起因**；
- **根因（本机 float32 数值实验 + 官方源码对比证实）**：本地原 logp 实现以 `u=atanh(a)` 反推再算 `((u−μ)/σ)²`，float32 下 |u|≳5 起反推失真、`atanh` 值被压在 ≈7.25，二次项除以 σ² 后无界负向（μ=41、σ=e⁻⁵ 时单维约 −1.3e7）；官方 `curl/curl_sac.py` 的 `gaussian_logprob` 用采样噪声 ε 的噪声形式（`−0.5ε² − log_std`，无该项），饱和校正受 `relu(1−a²)+1e-6` 保底 → logp 恒有界。float32 网格对比：正常区两式差 ~1e-6（数学等价），饱和区官方 ~12~17/维、本地旧式 −3.4e4~−1.6e7（μ≤45）。错误 logp 经 `−α·logp2` 污染 TD target（`y_absmax` 爆炸量与 `−α·logp2` 项吻合）→ MSE 大梯度经联合步反噬共享 encoder → z 漂移/坍缩 → 自激。

### 6.4 修复：logp 改噪声形式（对齐官方），移除 `--grad-clip`

- `networks.py`：删除 `tanh_normal_log_prob`（atanh 反推式）；`reparam_sample` 保存采样噪声 ε，`logp = Σ(−0.5ε² − log_std − 0.5·ln(2π) − log1p(−a²))`（a=clamp(tanh(u))），保留 `log_std` 的 clamp[-5,2] 与 `log1p(−a²)` 校正。正常区与原式数学等价（本机实测差 ~1e-6），饱和区恒有界（与官方同量级，512×6 样本下 logp ∈ ~13~103）。
- `train.py`：移除 `--grad-clip` CLI 与全部实现。弃用理由（实证）：clip=10 触发率 75.6%、健康训练被压制，且未阻止崩溃（仅推迟约 8k env 步）——clip 只裁更新步梯度，不裁 target 前向里每步重新采样的 `−α·logp2` 污染，属治标无效防护。
- **验证状态**：本机 smoke（seed=0）确定性通过（两次逐字一致）、float32 网格数值回归通过（饱和区有界断言）；GPU 全量复跑（batch 512、seed 10086 对照首跑）待用户回执。

## 七、正确性审查要点（实现期自查）

1. actor 与 critic 共享 f_q，但 actor 只更新头：actor 步对 z detach；θ_enc 只由联合步优化器 step，杜绝"同一参数多优化器双重更新"（SAC 基线方案 B 教训的结构化落地）。
2. target critic 与 f_k 是两份独立 EMA 副本（0.01 / 0.05），各含一份 θ_enc——对应论文表 3 "Q function EMA τ=0.01" 与 "Encoder EMA τ=0.05" 两行；next 动作由 online 组件采样、target Q 用 target_encoder 的 next 表征（同 SAC 基线 target 惯例）。
3. random crop 对 (B,9,100,100) 整体取每样本同一窗口坐标 → 帧栈内时间一致性自动满足；评估恒 center crop。
4. InfoNCE logits 先减行 max 再 softmax，labels=arange(B)；`z_k` detach（key 无梯度）；W 仅由联合步更新。
5. `--diag` 为默认关闭的旁路记录参数：不传时数值语义与基线逐位一致（smoke 已证），可随时移除；`reparam_sample` 的 logp 噪声形式在正常区与原 atanh 反推式逐位等价（float32 差 ~1e-6）、饱和区有界（防复发根因，见 §6.4）。
