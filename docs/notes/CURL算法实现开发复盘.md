# CURL 算法实现 —— 开发复盘

## 一、开发内容概述

### 1.1 任务背景与目标

按需求《实现CURL + SAC算法.md》交付 **CURL**（CURL + SAC，DMControl walker-walk 像素）：对比学习表征 + off-policy 控制，重点在共享编码器；代码 `src/curl-sac/` 平行复制 SAC（不做代码共享）；架构与对比机制以 CURL 论文为准（本地 `docs/paper/paper.md`）。技术细节见 [CURL实现技术文档.md](docs/techs/CURL实现技术文档.md)。

### 1.2 与用户对齐的决策（文字问答，未用 ask 工具）

1. **方案 A（全链路共享单一 f_q）**：actor/critic/InfoNCE query 共用同一 Encoder（C_dmc→1024→50→LN→tanh ⇒ z(50)），另建 target critic（τ=0.01）与 f_k（τ=0.05）两份 EMA 副本。否决方案 B（actor/critic/对比各自同构实例、梯度互不相通）——B 下 InfoNCE 梯度到不了 RL 所用表征，算法退化为"Pixel SAC + 旁路对比模块"，CURL 收益无从体现。
2. **actor 不回传 f_q**（detach z）：把 SAC 基线"共享参数双重更新"教训落实为梯度归属纪律——θ_enc 只由联合步优化器 step。
3. **联合步（路径甲）**：Bellman 与 InfoNCE 同 loss 一次 backward/step（论文 E.4 伪代码 `loss = curl + sac` 结构），等权无平衡系数；分步会致同一 θ_enc 被两优化器先后 step（双重更新），需手动梯度累积、易错。
4. **观测存储 100×100 原帧**：训练 random crop（100→84，栈内坐标一致）、评估 center crop；buffer 形状从 SAC 的 84 改 100（replay 100K 内存约 +43%，uint8 约 12 GB）。
5. **EMA 双套速率**：target critic 0.01 / key f_k 0.05（论文表 3 两行记法冲突——表 3 "Encoder EMA τ=0.05" 与 4.7 伪代码 m=0.95 实为同一公式 θk←0.95θk+0.05θq，统一为 τ 记法）。
6. **训练期收集也用 random crop**（与 batch 更新一致），评估/eval_demo 用 center crop。
7. **存档 {f_q, actor_head} + meta**（actor 不再是自包含网络）；日志增 curl loss。
8. **smoke 参数较 SAC 再降一档**（CPU 更新侧含 Enc 三次前向 + InfoNCE）：max=60 / initial=20 / replay=256 / batch=8 / eval_interval=60 / eval_episodes=1，保留 1 局 eval 覆盖评估通路。
9. **复制 profiler/log_tee/viewer**（5B 决议），`--gpu/--smoke/--profile` CLI 沿用。

### 1.3 交付物与验收口径

`src/curl-sac/` 八文件（networks/pipeline/buffer/train/eval_demo 新写；log_tee/viewer/profiler 原样复制）。验收（数据导向）：smoke 输出确定数值、退出码 0、存档齐、demo 载入一致、同 seed 两次逐字一致、`--gpu` 显式语义。完整 100K 训练由用户在 GPU 执行（首跑发散，见 §三）。

## 二、开发决策

### 2.1 方案 A vs B（共享语义是 CURL 的生命线）

核心不在"actor 与 critic 是否互享参数"，而在 **RL 前向读取的表征是否接收 InfoNCE 梯度**。SAC 基线选"actor/critic 各自同构 body"防的是 actor↔critic 策略梯度污染（off-policy 特有）；CURL 的共享是论文机制（policy/value 建于 query encoder、联合训练）。两决策正交、不矛盾：A 方案落地时把 SAC 的教训转成纪律（共享参数单一归属 + actor 对 Enc stop-grad），而非放弃共享。

### 2.2 联合步合体 vs 分步

θ_enc 需同时收 Bellman 与 InfoNCE 梯度。"分步"会让 θ_enc 每 block 被 opt_critic 与 opt_curl 先后 step 两次（非标准双重更新）——即 SAC 复盘里记录的坑在共享编码器上的重现；"合体一次 backward"等权相加即论文 E.4 字面结构，工程唯一干净解。代价：联合步计算图更大（Bellman+InfoNCE 同图），CPU batch8 单次 update 实测 ~244ms（量级参考）。

### 2.3 观测 100 原帧入 buffer 的连带

SAC 渲染 100 后立即 center crop 84 存 buffer；random crop 需要 84 之外仍有裁切余量 → buffer/FrameStack 全部改存 100 原帧，crop 外置到决策与 batch 采样路径。uint8 直传 device 再归一化的既有优化保留（crop 在 numpy uint8 侧做，避免 float 4× 搬运）。

### 2.4 GPU 数值失稳的处置路线（2026-09 新增，详见技术文档 §六）

GPU 首跑（seed=10086）约 env 37k 数值崩溃。处置不直接改算法而是按"归因→最小干预→验证"推进：

1. 拉取远程分析文件（整目录含 160MB×2 checkpoint，按用户决议只拉最小集：update_diag.jsonl 20500 行 + train_log + eval_history，落 `data/curl-sac-20260908-145018/`，经 scp 校验大小一致）。
2. 给 `train_step` 加 `--diag` 旁路记录（不改变语义，默认关闭），本机 smoke 自证"带 diag 与不带逐字一致 + 两次一致"。
3. 逐 1 步分析 20500 行 diag：首个异常是 actor 采样 logp（step 17223 间歇 tail −337 → 17232 起连续 −1e5 以下），Q/target Q 全程未爆（tqmin ~80–90），表征坍缩（z_std 0.49→0.05、curl→ln512）在崩溃后段——**归因：actor tanh-squash tail → 污染 TD target y → 大梯度经联合步反噬共享 encoder → 表征坍缩自激**。Q 高估循环与表征坍缩两个候选被数据排除为起点。
4. 修复初选 **约束 A（仅 CURL 侧）+ d 路径（`--grad-clip` 总范数裁剪）**：语义干扰最小、可逆，默认关闭不影响基线；阈值 10 取论文 Table 4 Atari 先例。启用 clip 时 diag 增记范数与触发标志，供 GPU 复跑校准阈值并观察 tail 频率（判断是否需叠加源头修复）。
5. GPU 复跑回执证伪 clip（seed=0，`--grad-clip 10`）：崩溃推迟到 env ~45400 仍复发，且健康期 `clip_main` 触发率 75.6%（阈值 10 低于健康梯度 p50=18.3，全程压制约 3/4 更新）。对照官方源码 + 本机 float32 数值实验，根因确认为本地 logp 的 atanh 反推求值在 tanh 饱和区失真无界（官方噪声形式无此问题）；终局改为 logp 噪声形式（networks.py `reparam_sample`），并**移除 `--grad-clip`**（详见技术文档 §6.3/§6.4）。

### 2.5 本机验证边界

CPU 只能承担 smoke 级（batch 8）与确定性自检；GPU 大 batch 的真实行为（数值、clip 触发率、完整 100K）以用户实机回执为准。

## 三、实测结果

### 3.1 本机 smoke（seed=0）

```
ep=1 env_steps=60 ep_return=6.417 q=0.3163 curl=2.081 pi=-1.443 temp=0.09989
EVAL mean_return=23.847
```

curl≈ln8 与 batch8 InfoNCE 初始期望一致，确认联合步学习通路正确。同 seed 两次逐字一致（logp 噪声形式修复后复验；q/curl/temp 与修复前逐位一致、pi/EVAL 尾数级微差，佐证正常区数值等价）。

### 3.2 GPU 两轮全量训练与崩溃回执

seed=10086 首跑：前 36k env steps 健康（EVAL 30k=52.021），ep37 起 q→1e20 级、curl 恒 ln(512)、ep_return 落至 ~13，env ~42022 手动中断（diag 归因见 §2.4/技术文档 §6.3）。`--grad-clip 10` 复跑（seed=0）：崩溃推迟到 env ~45400 仍复发（q→1e19 级、μ 指数爬至 23759），clip 健康期触发率 75.6%。终局修复 = logp 噪声形式（networks.py），`--grad-clip` 已移除；本机 smoke 与数值回归自证通过，GPU seed=10086 全量复跑待用户回执（归因与修复见技术文档 §6.3/§6.4）。

## 四、迭代中修正

1. `torch.optim.Optimizer` 无 `parameters()`（实现梯度裁剪需经 `param_groups` 取参，`_opt_params` 辅助）——`--grad-clip` 期间踩坑、GPU 前本机 smoke 即暴露 AttributeError；该参数随终局修复移除，坑随代码删除而消失。
2. diag 字段命名 `alpha→alpha_loss`（避免与温度混淆）。
3. smoke 档位依 CPU 实测下调：SAC 档（120/50/1024/16）对 CURL（3× Enc 前向 + InfoNCE）偏重，改 60/20/256/8。

## 五、经验与衔接

- **"共享与防污染是两件事"**：方案 A 的共享是论文机制（InfoNCE 必须塑形 RL 表征），SAC 基线"各自 body"是防 actor↔critic 梯度污染——把后者纪律（单一归属、stop-grad）结构化为前者的实现约束，两不冲突。
- **崩溃现场只留均值会误导**：逐 episode 平均掩盖了"Q 其实没炸、是 logp tail 先行"的真相；数值诊断要逐步记录 TD/actor/表征三组量，坍缩判据（curl=ln(B)、z batch 内 std→0）可离线直接验证。
- **实机发现的问题先归因、修复须能经实机证伪**：先旁路记录定位首变量（logp tail 先行、Q 与坍缩在后）；grad clip 作为最小干预的验证性修复，被 GPU 复跑证伪（只推迟不消除、阈值 10 干扰健康训练）后弃用，再以官方源码对照 + 数值实验定位到 logp 求值路径这一真正根因。SAC 基线对照（用户已证完整训练无问题）把根因域锁到 CURL 特有机制。
- **待办衔接**：GPU seed=10086 全量复跑回执（去掉 clip、对照首跑口径）确认不复发 → 归档本次两轮发散训练产物（145018 与 160421 的 diag/日志已拉取至本地 data/）→ 工作区改动 git 提交并入历史任务。
