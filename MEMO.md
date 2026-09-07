# MEMO

## 项目现状

本项目按文献 CURL: Contrastive Unsupervised Representations for Reinforcement Learning 从零实现 CURL 表征学习框架，目标为基于对比学习建立状态表征模型，将高维图像转换为低维表征，并作为"即插即用"编码器接入 RL 算法。实现顺序为环境适配 → AC → DQN → SAC → CURL，其中 SAC 与 DQN 的先后以实际推进为准，最终进行对比。项目背景详见 [README.md](README.md)。

环境复现现状：已引入 uv 依赖管理，新增 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock)（清单锁定 33 个包，来源为清华镜像），当前 .venv 未动、仍含 torch 2.14.0+cpu 等 41 个包。约定 torch 不纳入 uv 清单，GPU 主机独立维护 CUDA 版 torch。

已完成的内容与佐证材料：

- 环境验证：已完成 Gymnasium、DMControl、ALE 三个强化学习框架的接入冒烟验证。复盘见 [RL环境冒烟验证开发复盘.md](docs/notes/RL环境冒烟验证开发复盘.md)，技术文档见 [RL环境冒烟验证技术文档.md](docs/techs/RL环境冒烟验证技术文档.md)。
- AC 算法：已完成 on-policy Advantage Actor-Critic 基线，作用于 DMControl walker/walk 像素观测，实现位于 [src/ac/](src/ac/)。复盘见 [AC算法实现开发复盘.md](docs/notes/AC算法实现开发复盘.md)，技术文档见 [AC实现技术文档.md](docs/techs/AC实现技术文档.md)，需求文档见 [实现Actor-Critic算法.md](docs/functions/实现Actor-Critic算法.md)。
- SAC 算法：已完成 soft actor-critic 算法代码与三份文档，尚未 git 提交（工作区状态）。实现位于 [src/sac/](src/sac/)，复盘见 [SAC算法实现开发复盘.md](docs/notes/SAC算法实现开发复盘.md)，技术文档见 [SAC实现技术文档.md](docs/techs/SAC实现技术文档.md)，需求文档见 [实现Soft Actor-Critic算法.md](docs/functions/实现Soft Actor-Critic算法.md)。

架构设计参考 [模型架构报告.md](docs/reports/模型架构报告.md)，训练数据目录映射见 [算法与环境表.md](docs/reports/算法与环境表.md)，`data/` 目录已入 .gitignore。

## 当前任务

暂无。待用户补充下一步任务（如 DQN 或 CURL 阶段）后再行记录。

2026-09 环境复现任务（GPU 主机部署）进行中：用户将在 GPU 主机（Windows + NVIDIA，CUDA 与 PyTorch 已配好）复制本项目，用 uv.lock 快速复原依赖；torch 由用户既有方案独立维护。清单与操作指引已交付，GPU 主机同步结果待用户回执。

## 历史任务

- 任务：uv 依赖管理方案。内容：应"在 GPU 主机无损复现 CPU 主机环境"需求，引入 uv 生成 [pyproject.toml](pyproject.toml)（5 个直接依赖按现状 pin，requires-python 3.12）与 [uv.lock](uv.lock)（33 个包，清华镜像），约定 torch 系游离清单外由 GPU 主机独立装 CUDA 版，同步须带 --inexact 防误删。结果：清单与两端操作指引已交付，未动 .venv，未提交 git；GPU 主机执行结果待回执。
- 任务：RL 环境冒烟验证。内容：验证 Gymnasium、DMControl、ALE 三个框架的接入与渲染链路。结果：通过，产出 [smoke_environments.py](src/smoke_environments.py)。提交 b019f41，时间为 2026-09-06。
- 任务：AC 算法实现。内容：按需求文档实现 on-policy AC 基线（actor 与 V 头共享卷积编码器，tanh-squashed Gaussian，块折扣 TD(0) 更新），交付 src/ac/ 五个文件与短跑自检。结果：短跑自检确定性通过，完整 100K 步训练留给用户本地执行。提交 f6ed057，时间为 2026-09-06。
- 任务：SAC 算法实现。内容：在 AC 基础上实现 SAC，保持训练、测试与存档设计，代码置于 src/sac/。结果：已完成，尚未 git 提交。

## 笔记

- 任务粒度约定：按算法/特性为任务单位，一个算法从需求、技术文档到实现、验证的完整过程记为一次历史任务。
- 记忆维护约定：每次任务结束后更新本文件；历史任务以 git 提交与 docs 文档为准，现状只写结论并链接到具体文档，不重复细节。
- 进入 CURL 阶段时，AC 基线（无 random crop、无温度、on-policy）与 CURL+AC 的差异即对比学习引入的净收益。

## 其他重要信息

- 运行环境为 Windows，项目使用 .venv 虚拟环境，python 解释器位于 .venv\Scripts\python.exe。
- 已装 uv 0.12.9（C:\Users\yuanlinHex\.local\bin\uv.exe）。uv 全局配置位于 %APPDATA%\uv\uv.toml，其中把 https://pypi.tuna.tsinghua.edu.cn/simple 设为默认索引，故 uv.lock 全部包来源为清华镜像；GPU 主机需能访问该镜像，否则须换源后重跑 uv lock。
- uv.lock 相对当前 .venv 缺的包为 torch、torchvision 及其专属传递依赖（jinja2、sympy、networkx、mpmath、functorch 等），源于 torch 游离于清单外。
- 工作区包含 Gymnasium、dm_control、Arcade-Learning-Environment 三个第三方框架的源码仓库，仅作本地参考，不属于本项目交付物。
