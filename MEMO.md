# MEMO

## 项目现状

按文献 CURL 自零实现对比学习表征框架，将高维像素观测编码为低维表征，作为即插即用编码器接入 RL。推进顺序为环境适配 → AC → DQN → SAC → CURL，其中 AC 与 DQN 的先后以实际为准，最终对比。详情见 [README.md](README.md)、[模型架构报告.md](docs/reports/模型架构报告.md)、[算法与环境表.md](docs/reports/算法与环境表.md)。

已交付并通过 git 提交（2026-09，commit 见历史任务）：三框架环境冒烟、AC、SAC、GPU 训练开关与 SAC GPU 修复、无头 EGL 与耗时统计、训练日志记录、性能瓶颈诊断、CURL（实现 + 数值诊断 + 数值爆炸修复）。代码在 [src/ac/](src/ac/)、[src/sac/](src/sac/)、[src/curl-sac/](src/curl-sac/)，复盘与技术文档在 docs/notes、docs/techs。

CURL（CURL+SAC，方案 A）已实现并提交：全链路共享单一 f_q(z50)、Bellman+InfoNCE 联合步、双 EMA、random crop 增强。GPU 数值崩溃根因已闭环定位（本地 logp 的 atanh 反推求值在 float32 tanh 饱和区失真无界，官方噪声形式无此问题），修复为 logp 噪声形式（对齐官方）、`--grad-clip` 已移除（复跑实测证伪：仍崩且健康期触发率 75.6%）；改动待 git 提交 + GPU seed=10086 全量复跑回执。技术 [CURL实现技术文档.md](docs/techs/CURL实现技术文档.md)、复盘 [CURL算法实现开发复盘.md](docs/notes/CURL算法实现开发复盘.md)。

依赖由 uv 管理（[pyproject.toml](pyproject.toml)、[uv.lock](uv.lock)，清华镜像）。约定 torch 系游离清单外，GPU 主机独立维护 CUDA 版，同步带 --inexact。GPU 主机已用 uv.lock 复原依赖并跑通冒烟。数据落在 data/（已入 .gitignore）。

## 当前任务

2026-09 未闭环、待处理：
- CURL 数值崩溃修复（logp 噪声形式，`--grad-clip` 已移除）工作区改动待 git 提交；随后 GPU 全量复跑 seed=10086（对照首跑口径 batch 512/100k、不带 clip）回执：确认不复发，防复发判据 = curl 不触 ln(B)、z_std_mean 不塌、logp2_min 全程有界。细节见 [CURL实现技术文档.md](docs/techs/CURL实现技术文档.md) §6。
- uint8 直传优化 GPU 实机复验（见历史任务 `51f870e`）回执后做最终闭环。

## 历史任务（复盘与技术文档见各 docs 文件）

- RL 环境冒烟验证 `b019f41`：Gymnasium、DMControl、ALE 接入与渲染链路验证通过，产物 [smoke_environments.py](src/smoke_environments.py)。
- AC 算法 `f6ed057`：on-policy AC 基线（actor 与 V 头共享卷积编码器、tanh-squashed Gaussian、块折扣 TD(0)），作用于 DMControl walker/walk，交付 src/ac/。短跑自检确定性通过，完整 100K 步由用户本机执行。
- SAC 算法 `5c787d3`：在 AC 基础上实现，保留训练/测试/存档设计，交付 src/sac/。
- GPU 训练开关 `88346af`：AC/SAC 的 train 与 eval_demo 增 `--gpu`（不传则 CUDA 可用时自动启用，显式传但不可用则 parser.error），模型/批量张量统一搬运，meta 记录 device。
- SAC GPU 修复与 uint8 传输优化 `51f870e`：GPU 实机首训报 `log_alpha` 非叶子，改 nn.Parameter 封装进 `_Temperature`(nn.Module)、由 Adam 接管修复；复核后把 AC/SAC 观测管线统一为 uint8 直传 device 再归一化。GPU 主机 torch 2.14.0+cu132；uint8 直传的 GPU 实机复验待闭环（见当前任务）。
- 无头渲染与耗时统计一批 `784e376`：linux 无 DISPLAY 且未设 MUJOCO_GL 时于 dm_control import 前补 `MUJOCO_GL=egl`（无头 GPU 直跑不再报 GLFW/DISPLAY 错误）；SAC `--profile`（src/sac/profiler.py + train.py 插桩，分段计时 policy/phys/render/obs/sample/update 与 reset，覆盖 max_env_steps=2000、跳过 eval、写 profile.json）；AC/SAC eval_demo 与 `--gpu` 显式语义（device = cuda iff args.gpu，不指定恒 CPU）落地于此批，注意与 `88346af` 提交时"不传自动启用"语义的差异（见笔记）。
- 训练日志记录 `2946124`：ac/sac 训练控制台全量输出 tee 落盘 data/<算法>-<时间戳>[-smoke]/train_log.txt（src/{ac,sac}/log_tee.py + train.py 接入），平台无关已自证。
- SAC 性能瓶颈诊断 `92a5065`：src/sac/diag_update.py（GPU 复现 update mean=94ms/p50=88.6ms；device_ratio 行不可采信，见笔记），需求 [性能瓶颈分析.md](docs/functions/性能瓶颈分析.md)、[GPU服务器信息.md](docs/reports/GPU服务器信息.md)。
- CURL 算法 `3a65df6`/`0487fb8`/`800c67b`：方案 A（CURL+SAC，共享单一 f_q + Bellman/InfoNCE 联合步 + 双 EMA + random crop）交付 src/curl-sac/；`0487fb8` 增 `--diag` 数值旁路记录；`800c67b` 增 `--grad-clip`（经 GPU 复跑证伪后已随 logp 源头修复移除，见当前任务）。本机 smoke 自证通过。需求 [实现CURL + SAC算法.md](docs/functions/实现CURL + SAC算法.md)，技术 [CURL实现技术文档.md](docs/techs/CURL实现技术文档.md)、复盘 [CURL算法实现开发复盘.md](docs/notes/CURL算法实现开发复盘.md)。

## 笔记

- 任务粒度：按算法/特性为任务单位，一个算法的需求 → 技术文档 → 实现 → 验证记为一次历史任务。历史以 git commit 与 docs 为准，现状只写结论并链接、不重复细节；每次任务结束更新本文件。
- 进入 CURL 阶段时，AC 作为 on-policy 基线（无 random crop、无温度），其与 CURL+AC 的差异即引入对比学习的净收益。
- GPU 代码审查旁路发现、已落地或待定夺项：a) `--gpu` 显式语义已随 `784e376` 提交（2026-09），注意与 `88346af` 提交时"不传则自动启用"的差异；b) AC `evaluate` 复用 explore=True 致 eval 采样动作、与 SAC 确定性评估不一致，是否改确定性 tanh(μ) 待定；c) 多卡选卡依赖 CUDA_VISIBLE_DEVICES；d) 未设 cudnn.deterministic，GPU 训练不可逐位复现。
- 无头渲染约束：dm_control 的 MUJOCO_GL 在其 `_render` 模块 import 瞬间冻结（engine.py 顶层 `from dm_control import _render`），设置必须发生在任何 dm_control import 之前（放 main() 无效）。
- 模块命名避开标准库：profile.py 遮蔽 stdlib profile 致 cProfile/torch 导入失败，已改名 profiler.py；新模块名需核对无 stdlib 同名。
- 程序内 tee 全量日志要点（2026-09 训练日志记录）：委托 isatty/fileno/encoding 等流属性、flush 转发原流、文件句柄模块级持有防提前回收、逐行 flush 使 Ctrl+C 与异常 traceback 不丢；管道重定向下终端跨流乱序不影响文件按 write 调用序记录。
- 本机 Windows（torch CPU 版）跑同口径 batch 512 训练过慢：SAC 2K profile 运行超 15 分钟未完成已终止，此类全量剖析交给 GPU 服务器或用户执行。
- torch.profiler（cu132 + Blackwell）下 key_averages() 的 self_device_time_total 直接求和的 device 累计时间会重复/重叠计数，实测超区间墙钟（225%）而失效；判别 GPU 忙闲宜改用变量对照实验或逐算子表人工核对。
- CURL InfoNCE 表征坍缩的离线判据（2026-09）：curl_loss 恒等于 ln(batch)（softmax 均匀分布）、diag 的 z_std_mean（batch 内每维 std 均值）→0 与 logits_spread→0 均为坍缩信号；训练崩溃归因需逐更新步记录 TD/actor/表征三组量并区分先后，逐 episode 均值会掩盖首变量（实测 Q 未自爆、logp tail 先行）。
- CURL 崩溃根因（2026-09 闭环）：本地 logp 的 `atanh(a)` 反推 + `((u−μ)/σ)²` 求值在 float32 tanh 饱和区（|u|≳5 起失真、atanh 被压 ~7.25）无界负向（μ=41、σ=e⁻⁵ 单维 −1.3e7），官方 `curl/curl_sac.py` 噪声形式（`−0.5ε²−log_std`，无该项）恒有界；正常区两式逐位等价。同类 bug 判别：Q/表征健康而 logp 现 −1e3 以下异常负值。grad clip 治标无效已被实机证伪（复跑仍崩、健康期触发率 75.6%）。
- tanh-squash 策略数值自激机制（2026-09）：μ 越出 tanh 线性区 + log_std 收缩至 clamp 下界 → 采样饱和 a≈±1 → logp 巨大负 → 目标 y 的 −α·logπ 项污染 TD target → MSE 大梯度；CURL 因共享 encoder 每步被 Bellman 大梯度反噬（z 漂移→actor 失配）而放大，SAC 基线 actor/critic 各自 body 无此环节（推断，SAC 对照未做）。
- 崩溃型训练 run 的中断产物：update_diag.jsonl 每 500 步 flush，Ctrl+C 于 backward 中时已 flush 行完整（20500=500×41 边界）；远程 GPU 产物经 `ssh gpu-server` + scp 拉取，checkpoint 160MB 级按需取舍。

## 其他重要信息

- 环境 Windows；解释器 .venv\Scripts\python.exe；uv 0.12.9（C:\Users\yuanlinHex\.local\bin\uv.exe），全局默认索引清华镜像（%APPDATA%\uv\uv.toml），GPU 主机须能访问否则换源重跑 uv lock。
- uv.lock 相对当前 .venv 缺 torch、torchvision 及其专属传递依赖（jinja2、sympy、networkx 等），源于 torch 游离清单外。
- 仓库内 Gymnasium、dm_control、Arcade-Learning-Environment、curl（CURL 官方团队实现源码，本地新增参考，已入 .gitignore）为第三方源码，仅本地参考，非本项目交付物。
