# MEMO

## 项目现状

按文献 CURL 自零实现对比学习表征框架，将高维像素观测编码为低维表征，作为即插即用编码器接入 RL。推进顺序为环境适配 → AC → DQN → SAC → CURL，其中 AC 与 DQN 的先后以实际为准，最终对比。详情见 [README.md](README.md)、[模型架构报告.md](docs/reports/模型架构报告.md)、[算法与环境表.md](docs/reports/算法与环境表.md)。

已交付并通过 git 提交（2026-09，commit 见历史任务）：三框架环境冒烟、AC 基线、SAC、AC/SAC `--gpu` 训练开关与 SAC GPU 修复。代码在 [src/ac/](src/ac/)、[src/sac/](src/sac/)，复盘与技术文档在 docs/notes、docs/techs。

依赖由 uv 管理（[pyproject.toml](pyproject.toml)、[uv.lock](uv.lock)，清华镜像）。约定 torch 系游离清单外，GPU 主机独立维护 CUDA 版，同步带 --inexact。GPU 主机已用 uv.lock 复原依赖并跑通冒烟。数据落在 data/（已入 .gitignore）。

## 当前任务

2026-09 以下改动均未 git 提交：前三批经本机自证、待 GPU 服务器实机回执后闭环；末项训练日志记录平台无关、已本机自证，git 提交后即闭环：
- 无头默认 EGL（src/{ac,sac}/train.py）：linux 无 DISPLAY 且未设 MUJOCO_GL 时于 dm_control import 前补 `MUJOCO_GL=egl`；实机验证为不带任何 export 直跑不再报 GLFW/DISPLAY 错误。
- SAC `--profile`（新增 src/sac/profiler.py，train.py 插桩）：分段计时 policy/phys/render/obs/sample/update 与 reset，warmup/train 两阶段统计，覆盖 max_env_steps=2000、跳过 eval、写 profile.json；CPU 端与 GPU 端 2K 同口径（batch 512）剖析由用户自跑，回执后做收集侧/更新侧瓶颈分析。
- `--gpu` 语义改为显式（src/{ac,sac}/{train,eval_demo}.py 四文件）：`device = cuda iff args.gpu`，不指定恒 CPU，显式传但 CUDA 不可用仍 parser.error；mock CUDA 可用时 smoke 仍 device=cpu 已自证。实机验证：不带 `--gpu` cfg 应 device=cpu、带 `--gpu` 应 device=cuda。
- 旧收尾项：GPU 实机对 uint8 直传优化（见历史任务末条）复验并回执后做最终闭环。
- 训练日志记录：ac/sac 训练新增控制台全量输出落盘 data/<算法>-<时间戳>[-smoke]/train_log.txt——新增 src/{ac,sac}/log_tee.py 同构模块、train.py 各两处接入（mkdir 后 start、DONE 后 stop），tee 双写 stdout/stderr、原样照录、追加写入、逐条 flush、中断不丢；smoke 与终端逐字一致已自证，平台无关不需 GPU 实机。需求 [训练日志记录.md](docs/functions/训练日志记录.md)，技术 [训练日志记录技术文档.md](docs/techs/训练日志记录技术文档.md)、复盘 [训练日志记录开发复盘.md](docs/notes/训练日志记录开发复盘.md)；git 提交后本条移入历史任务。
- SAC update 段瓶颈诊断工具（新增 src/sac/diag_update.py，未提交）：GPU 实机 --gpu 跑多次 update，逐次墙钟统计之外由 torch.profiler 抓一次稳定段，输出 CUDA device 时间及其占区间墙钟比例，用于判别单次 update 87ms 属结构性算量还是同步/搬运等待；不改 train.py 常备路径，--cpu-smoke 本机通路已自证，实机回执后做性能瓶颈分析闭环。需求 [性能瓶颈分析.md](docs/functions/性能瓶颈分析.md)，GPU 服务器信息见 [GPU服务器信息.md](docs/reports/GPU服务器信息.md)。

## 历史任务（复盘与技术文档见各 docs 文件）

- RL 环境冒烟验证 `b019f41`：Gymnasium、DMControl、ALE 接入与渲染链路验证通过，产物 [smoke_environments.py](src/smoke_environments.py)。
- AC 算法 `f6ed057`：on-policy AC 基线（actor 与 V 头共享卷积编码器、tanh-squashed Gaussian、块折扣 TD(0)），作用于 DMControl walker/walk，交付 src/ac/。短跑自检确定性通过，完整 100K 步由用户本机执行。
- SAC 算法 `5c787d3`：在 AC 基础上实现，保留训练/测试/存档设计，交付 src/sac/。
- GPU 训练开关 `88346af`：AC/SAC 的 train 与 eval_demo 增 `--gpu`（不传则 CUDA 可用时自动启用，显式传但不可用则 parser.error），模型/批量张量统一搬运，meta 记录 device。
- SAC GPU 修复与 uint8 传输优化 `51f870e`：GPU 实机首训报 `log_alpha` 非叶子（.to() 使 requires_grad 张量变非叶子），改 nn.Parameter 封装进 `_Temperature`(nn.Module)、由 Adam 接管修复；复核后把 AC/SAC 观测管线统一为 uint8 直传 device 再归一化（消除 CPU 侧 float32 4× 搬运），GPU 端传输量预期降约 4×。GPU 主机 torch 2.14.0+cu132（CUDA 13.2）；本机 CPU smoke 已自证，实机复验待闭环。

## 笔记

- 任务粒度：按算法/特性为任务单位，一个算法的需求 → 技术文档 → 实现 → 验证记为一次历史任务。历史以 git commit 与 docs 为准，现状只写结论并链接、不重复细节；每次任务结束更新本文件。
- 进入 CURL 阶段时，AC 作为 on-policy 基线（无 random crop、无温度），其与 CURL+AC 的差异即引入对比学习的净收益。
- GPU 代码审查旁路发现、未改动的待定夺项：a) 已落地：`--gpu` 改为显式语义（2026-09，见当前任务，未提交），注意与历史行 88346af 记录（提交时为"不传则自动启用"）的差异；b) AC `evaluate` 复用 explore=True 致 eval 采样动作、与 SAC 确定性评估不一致，是否改确定性 tanh(μ) 待定；c) 多卡选卡依赖 CUDA_VISIBLE_DEVICES；d) 未设 cudnn.deterministic，GPU 训练不可逐位复现。
- 无头渲染约束：dm_control 的 MUJOCO_GL 在其 `_render` 模块 import 瞬间冻结（engine.py 顶层 `from dm_control import _render`），设置必须发生在任何 dm_control import 之前（放 main() 无效）。
- 模块命名避开标准库：profile.py 遮蔽 stdlib profile 致 cProfile/torch 导入失败，已改名 profiler.py；新模块名需核对无 stdlib 同名。
- 程序内 tee 全量日志要点（2026-09 训练日志记录）：委托 isatty/fileno/encoding 等流属性、flush 转发原流、文件句柄模块级持有防提前回收、逐行 flush 使 Ctrl+C 与异常 traceback 不丢；管道重定向下终端跨流乱序不影响文件按 write 调用序记录。
- 本机 Windows（torch CPU 版）跑同口径 batch 512 训练过慢：SAC 2K profile 运行超 15 分钟未完成已终止，此类全量剖析交给 GPU 服务器或用户执行。

## 其他重要信息

- 环境 Windows；解释器 .venv\Scripts\python.exe；uv 0.12.9（C:\Users\yuanlinHex\.local\bin\uv.exe），全局默认索引清华镜像（%APPDATA%\uv\uv.toml），GPU 主机须能访问否则换源重跑 uv lock。
- uv.lock 相对当前 .venv 缺 torch、torchvision 及其专属传递依赖（jinja2、sympy、networkx 等），源于 torch 游离清单外。
- 仓库内 Gymnasium、dm_control、Arcade-Learning-Environment 是第三方框架源码，仅本地参考，非本项目交付物。
