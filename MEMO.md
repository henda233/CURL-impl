# MEMO

## 项目现状

本项目按文献 CURL: Contrastive Unsupervised Representations for Reinforcement Learning 从零实现 CURL 表征学习框架，目标为基于对比学习建立状态表征模型，将高维图像转换为低维表征，并作为"即插即用"编码器接入 RL 算法。实现顺序为环境适配 → AC → DQN → SAC → CURL，其中 SAC 与 DQN 的先后以实际推进为准，最终进行对比。项目背景详见 [README.md](README.md)。

环境复现现状：已引入 uv 依赖管理，新增 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock)（清单锁定 33 个包，来源为清华镜像），当前 .venv 未动、仍含 torch 2.14.0+cpu 等 41 个包。约定 torch 不纳入 uv 清单，GPU 主机独立维护 CUDA 版 torch。GPU 主机即 AI 训练主力机，主 CPU 主机已含 AC/SAC GPU 开关代码但无 CUDA 可自证。

已完成的内容与佐证材料：

- 环境验证：已完成 Gymnasium、DMControl、ALE 三个强化学习框架的接入冒烟验证。复盘见 [RL环境冒烟验证开发复盘.md](docs/notes/RL环境冒烟验证开发复盘.md)，技术文档见 [RL环境冒烟验证技术文档.md](docs/techs/RL环境冒烟验证技术文档.md)。
- AC 算法：已完成 on-policy Advantage Actor-Critic 基线，作用于 DMControl walker/walk 像素观测，实现位于 [src/ac/](src/ac/)。复盘见 [AC算法实现开发复盘.md](docs/notes/AC算法实现开发复盘.md)，技术文档见 [AC实现技术文档.md](docs/techs/AC实现技术文档.md)，需求文档见 [实现Actor-Critic算法.md](docs/functions/实现Actor-Critic算法.md)。
- SAC 算法：已完成 soft actor-critic 算法代码与三份文档，尚未 git 提交（工作区状态）。实现位于 [src/sac/](src/sac/)，复盘见 [SAC算法实现开发复盘.md](docs/notes/SAC算法实现开发复盘.md)，技术文档见 [SAC实现技术文档.md](docs/techs/SAC实现技术文档.md)，需求文档见 [实现Soft Actor-Critic算法.md](docs/functions/实现Soft Actor-Critic算法.md)。

架构设计参考 [模型架构报告.md](docs/reports/模型架构报告.md)，训练数据目录映射见 [算法与环境表.md](docs/reports/算法与环境表.md)，`data/` 目录已入 .gitignore。

## 当前任务

暂无。待用户补充下一步任务（如 DQN 或 CURL 阶段）后再行记录。

2026-09 环境复现任务（GPU 主机部署）进行中：用户已在 GPU 主机复制本项目并用 uv.lock 复原依赖、跑通冒烟。

GPU 训练开关任务（AC/SAC `--gpu`）进行中：改造已完成并通过本机 CPU 验证（smoke 无回归、`--gpu` 于 CUDA 不可用时 parser.error 退出、eval_demo headless CPU 跑通）。GPU 实机 SAC 首训误触 `log_alpha` 非叶子回归，已按 A_1（`_Temperature` 模块封装，Adam 接管 module.parameters）修复，本机 CPU 已自证。GPU 代码审查（用户复核后通过）后新增"uint8 直传 device 再归一化"传输优化（AC/SAC 观测管线，消除 CPU 侧 float32 4×搬运冗余），本机 CPU smoke 已自证、布局/数值对照验证无误，待 GPU 实机复验回执。改动覆盖 src/{ac,sac}/{pipeline,train}.py，连同 GPU 开关本体均未 git 提交。

## 历史任务

- 任务：GPU 训练开关（AC/SAC --gpu）。内容：为 AC、SAC 的 train.py 与 eval_demo.py 增加 `--gpu` 布尔开关（不传则 cuda 可用时自动启用、显式传却在 CUDA 不可用时 parser.error），device 解析后模型、log_alpha(仅 SAC)、批量张量统一搬运，观测自建张量均以模型参数所在 device 推断并 `.to/`、返回 numpy 前 `.cpu()`；meta 与 cfg 日志记录 device。结果：本机 CPU 全路径验证通过、无明显回归。GPU 实机首训 SAC 报 `ValueError: can't optimize a non-leaf Tensor`：`.to(device)` 使 requires_grad 张量变非叶子；改以 `nn.Parameter` 封装进 `_Temperature`(nn.Module)、随模块统一 `.to(device)`、Adam 接管 `module.parameters()` 修复。GPU 主机 torch 2.14.0+cu132(CUDA 13.2)。后续 GPU 代码审查把 AC/SAC 观测管线统一为 uint8 直传 device 再归一化（见下方 GPU 传输优化复核），相应 `_Temperature` 修改与 CPU smoke 自证为合并记录。实机复验回执待闭环；尚未 git 提交。
- 任务：GPU 传输优化复核（uint8 直传 device）。内容：GPU 代码审查（P2，复核后通过）指出 SAC 的 obs/next_obs 原在 CPU 侧转 float32 造成 4× 冗余拷贝；将 AC/SAC 观测管线统一为归一化延迟到目标 device：SAC `obs_to_tensor` 改为接收 device、uint8 先 `.to(device)` 再 `.float().div_(255)`；AC `FrameStack` 改存 uint8（不再预转 float32），`select_action`/`update` 内 `.to(device).float().div_(255)`。结果：本机 CPU smoke（AC 600 步、SAC 120 步）跑通，AC 帧栈 dtype/HWC 布局与 crop 后数值对照验证正确。GPU 实机复验传输量预期降约 4×；待实机口径。
- 任务：uv 依赖管理方案。内容：应"在 GPU 主机无损复现 CPU 主机环境"需求，引入 uv 生成 [pyproject.toml](pyproject.toml)（5 个直接依赖按现状 pin，requires-python 3.12）与 [uv.lock](uv.lock)（33 个包，清华镜像），约定 torch 系游离清单外由 GPU 主机独立装 CUDA 版，同步须带 --inexact 防误删。结果：清单与两端操作指引已交付，未动 .venv，未提交 git；GPU 主机执行已通过。
- 任务：RL 环境冒烟验证。内容：验证 Gymnasium、DMControl、ALE 三个框架的接入与渲染链路。结果：通过，产出 [smoke_environments.py](src/smoke_environments.py)。提交 b019f41，时间为 2026-09-06。
- 任务：AC 算法实现。内容：按需求文档实现 on-policy AC 基线（actor 与 V 头共享卷积编码器，tanh-squashed Gaussian，块折扣 TD(0) 更新），交付 src/ac/ 五个文件与短跑自检。结果：短跑自检确定性通过，完整 100K 步训练留给用户本地执行。提交 f6ed057，时间为 2026-09-06。
- 任务：SAC 算法实现。内容：在 AC 基础上实现 SAC，保持训练、测试与存档设计，代码置于 src/sac/。结果：已完成，尚未 git 提交。

## 笔记

- 任务粒度约定：按算法/特性为任务单位，一个算法从需求、技术文档到实现、验证的完整过程记为一次历史任务。
- 记忆维护约定：每次任务结束后更新本文件；历史任务以 git 提交与 docs 文档为准，现状只写结论并链接到具体文档，不重复细节。
- 进入 CURL 阶段时，AC 基线（无 random crop、无温度、on-policy）与 CURL+AC 的差异即对比学习引入的净收益。
- 待用户定夺项（GPU 代码审查旁路发现，未在任务内改动）：a) `--gpu` 开关语义为"CUDA 可用即自动启用、显式传且不可用才 parser.error"，无强制 CPU 途径，如需 CPU 对照须新增 `--cpu`（用户已选暂不改，记备忘）；b) AC `evaluate` 复用 `collect_episode`（固定 explore=True）导致 eval 用采样动作、与 SAC 的 deterministic 评估不一致并抬高噪声，是否改为确定性 `tanh(μ)` 待定；c) 多卡主机无选卡参数，依赖 `CUDA_VISIBLE_DEVICES`（单卡可接受）；d) CUDA 下未设 `cudnn.deterministic`，GPU 训练不可逐位复现（RL 通常可接受）。

## 其他重要信息

- 运行环境为 Windows，项目使用 .venv 虚拟环境，python 解释器位于 .venv\Scripts\python.exe。
- 已装 uv 0.12.9（C:\Users\yuanlinHex\.local\bin\uv.exe）。uv 全局配置位于 %APPDATA%\uv\uv.toml，其中把 https://pypi.tuna.tsinghua.edu.cn/simple 设为默认索引，故 uv.lock 全部包来源为清华镜像；GPU 主机需能访问该镜像，否则须换源后重跑 uv lock。
- uv.lock 相对当前 .venv 缺的包为 torch、torchvision 及其专属传递依赖（jinja2、sympy、networkx、mpmath、functorch 等），源于 torch 游离于清单外。
- 工作区包含 Gymnasium、dm_control、Arcade-Learning-Environment 三个第三方框架的源码仓库，仅作本地参考，不属于本项目交付物。
