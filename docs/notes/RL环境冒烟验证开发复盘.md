# RL 交互环境冒烟验证 —— 开发复盘

## 一、开发内容概述

### 1.1 任务背景与目标

本仓库以"从零复现 CURL（Contrastive Unsupervised Representations for Reinforcement Learning）"为总目标。CURL 是融合对比学习表征与无模型 RL 的框架，其文献存于 `docs/paper/paper.md`。在实现底层算法（Actor-Critic、DQN、SAC）与 CURL 之前，需按需求文档《项目环境搭建与强化学习环境搭建》完成两类准备：一是"项目技术栈环境"（见 `docs/techs/项目技术栈.md`），二是本阶段的交付对象——"强化学习智能体交互环境"。

需求文档规定：项目仅采用两个交互环境，且需先编写测试代码、用随机策略确认环境"没有问题"。两环境为：

- DMControl 连续控制域：`walker` / `walk`（walker-walk），像素渲染；
- Atari 离散控制：`BattleZone`（gymnasium 注册名 `ALE/BattleZone-v5`）。

文献在评测层各用了数十个任务/游戏；本项目刻意收敛到上述最小可验证子集，待算法阶段再扩展，避免过早扩大验证面。

### 1.2 交付物与验收口径

产出纯 Python 冒烟脚本 `src/smoke_environments.py`，覆盖两条真实交互链路：DMControl（像素驱动器随机策略）与 ALE（gymnasium 原生像素随机策略）。验收贴合 AGENTS.md"客观条件 + 数据导向、以真实验证为主"：

- reset/step 返回结构与各框架协议一致；
- 观测 shape/dtype 合法（DMControl (100,100,3) uint8、ALE (210,160,3) uint8）；
- 随机 action 数值有限且在动作空间内；reward 数值有限、结束标志为布尔；
- 任何断言失败以抛错 + 非零退出码暴露。

随后按讨论为脚本追加可选开关 `show`（布尔），控制是否实时弹窗显示画面；该增补被严格做成"显示门控"层，不改动既有统计与断言逻辑。

### 1.3 实测结果（默认/不弹窗）

用 `.venv` Python 直接运行 `src\smoke_environments.py`，示例输出：

```
[dmcontrol] walker/walk  | obs=(100, 100, 3) uint8 | act=(6,) float64[-1,1] | episodes=3 steps=3000 total=85.03 mean=0.028
[ale] ALE/BattleZone-v5  | obs=(210, 160, 3) uint8 | act=Discrete(18)     | episodes=2 steps=2293 total=2000  mean=0.872
```

两链路均能走通且回报量级符合随机策略预期，确认环境本体无异常。

### 1.4 运行方式与 show 参数

脚本提供命令行与模块两级入口（详见技术文档），最常用为直接运行与可选弹窗：

```
.venv\Scripts\python.exe src\smoke_environments.py         # 不弹窗，输出统计
.venv\Scripts\python.exe src\smoke_environments.py --show  # 实时弹窗看画面
```

`--show` 仅在可视化叠加层生效：DMControl 由 `_PyGameViewer`（pygame）自建窗逐帧渲染，ALE 交给 env 原生 `render_mode="human"`；由于 `show` 与数值链路解耦，两种模式行为一致、统计完全可比，这也是后续作为训练守卫的基础。

## 二、开发决策

### 2.1 不新增任何第三方依赖

首步不是写码而是"读文献 + 核对技术栈"。通读 CURL 原文并对照 `uv pip list` 实测，得两条结论：① 随机策略冒烟所需能力（DMControl 仿真、ALE/gymnasium 协议、numpy）均已被 `项目技术栈.md` 规划的 `.venv` 覆盖，**本轮零新增**；② 文献附录 A 的向量化 random crop 用 `scikit-image.view_as_windows`，属后续训练增强细节、与本阶段目标无关且可用 torch 等价替代，故不提前装。

教训点：难点在辨识"文献技术要求"与"当下任务目标"的分界线，不照单全装，避免无谓环境膨胀。

### 2.2 纯 Python 冒烟 + 像素观测 + 固定种子

三点经与用户确认：测试形态纯 Python 冒烟、观测取像素、验收＝数值合法性 + 回报统计。像素观测是刻意之举——冒烟期就带出从像素渲染/读取到协议的整条链路，避免算法一接入就集中暴露底层渲染问题；固定种子用 `np.random.default_rng`，让转 CI/回归可复现。

### 2.3 先探测、后编码，避免把"想当然的接口"写进交付

编码前用一行式探针在真实 `.venv` 验证三件易翻车事：DMControl `suite.load('walker','walk')` 能否 reset/step、action_spec 维度界；其像素怎么拿；ALE `gym.make` 是否可行、观测与动作 shape。探针最有价值的事实是：**DMControl suite 默认不提供像素观测**——reset 观测 spec 是 `orientations/height/velocity` 状态量，需主动 `env.physics.render(height=100,width=100,camera_id=0)` 才得 (100,100,3) uint8。这与 gym/ALE 把像素当默认观测不同。冒烟最终揉合这两种"观测来源"成统一随机交互环，未因差异留任何"solver 分支"，正是先探后写的收益。ALE 侧亦先探明须 `gym.register_envs(ale_py)` 注册。

### 2.4 追加 show 时，把"显示"与"数值链路解耦"

用户增量需求为布尔 `show`，且 DMControl 也要真实弹窗。关键纪律：`show` 只决定"是否把已有帧送进窗口"，绝不参与观测采样、action、断言或回报统计，因此 `show=False` 与改动前一致，可作无头/CI 回归。现实取舍：ALE 按用户选择走 env 原生 `render_mode="human"`；DMControl 因 MJ 渲染无自带窗口，用已装 pygame 自建窗逐帧 flip；两链路互不共用窗口。pygame import 放显示分支内，避免无头副作用。

### 2.5 CLI 参数与模块复用并存

`main()` 用 argparse 暴露 `--show`（store_true）；同时保留 `run_dmcontrol_walker`/`run_ale_battlezone` 的模块复用面，均带 `show: bool = False`，便于训练/评估脚本直接调用，无全局状态泄漏。

### 2.6 顺带厘清统计口径

初版 DMControl 侧把 `steps` 用最后一 episode 覆盖、`mean_step_reward` 界定不严谨。在加 show 的同一次改动中统一：全程累加 `total_steps`，`mean_step_reward = total_reward / total_steps`，`episodes` 为真实回合数；字段名不变以兼容打印。属口径澄清，非功能变更。

## 三、教训与经验

### 3.1 接口想当然与实测的鸿沟，务必先探再写

DMControl 不给现成像素观测，若按"Mujoco/gymnasium 观测应含画面"惯性写，会写出拿不到图像的假设。对第三方封装环境永远先当黑盒试探 entry/exit，再依赖其约定；一行式/临时脚本探针在真实 `.venv` 执行是最低成本确定性来源。

### 3.2 gymnasium/ALE 的注册是独立步骤

必须 `gym.register_envs(ale_py)` 才能 `gym.make("ALE/...")`，否则抛 `NamespaceNotFound: ALE`。本阶段真实触发过一次并修复。此步手册有写、首次必踩，务必在脚本里显式注册（幂等）。

### 3.3 Windows 下别用超长 `python -c` 内联验证

PowerShell 内联引号嵌套 `$`/反引号近乎必死，多次产出晦涩解析错。规避：把探针写进临时脚本文件跑完即删（本次 `_pygame_probe.py`、误建的 `_draft.py` 均已清理）。另：脚本打印给终端的中文在中文 Windows 易乱码，标题用 ASCII 最稳妥。

### 3.4 把"需求扩散"用显式决策分治

两次"加功能/开始开发"都被拆成净边界：首次通过讨论对齐三问；`--show` 先探 pygame 再定解耦。通用手法：任何增强先用 2–3 个"必答问题"把范围、链接机制、验收一次性问完，同意后小步完成并以"旧功能回归"为硬验收（实现 show 后立即重跑 `show=False` 核对无回归）。

### 3.5 show 默认关闭的意义

不止少打扰，更保证脚本在无显示设备（CI/容器）仍是完整回归工具；有显示时 `--show` 把同一批帧投递到窗口。让"冒烟"兼做肉眼 play 与环境守卫。

### 3.6 不为演示硬造数字

脚本不校验"回报落在某启发式区间"——随机策略期望回报随环境与 episode 波动，无稳定可写死门限。验收锚点＝可复现的"无异常、结构合法、数值有限、跑够步数"，外加打印统计供人工判量级；数据导向 ≠ 硬造浮点门限之后再声称达标。

## 四、后续衔接

本交付为算法接入提供可自校验的"环境基准"：记录了像素观测真实来源形态（DMControl 走 physics.render 100×100×3；ALE 走 v5 原生 210×160×3）、action 协议（连续6维 / 离散18）如何收敛到统一"随机步进 + step 校验"环。进入 AC / DQN / SAC 基础实现或 CURL 本身时，可复用本脚本作"环境冒烟守门"，训练循环以 `env.step` 为协议抽取标准，避免重复踩"环境联调"坑。
