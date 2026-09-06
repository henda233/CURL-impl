# RL 交互环境冒烟验证 —— 技术文档

> 本项目以在真实 PyTorch/Gymnasium 等技术栈上复现 CURL 为目标；本文记录强化学习交互环境"随机策略冒烟验证"阶段的需求、选型与实现细节。源码：`src/smoke_environments.py`；复盘见 `docs/notes/RL环境冒烟验证开发复盘.md`。

## 一、需求复述

CURL 用从原始像素提取的高层特征做无模型 RL，算法实现在前必须先把"像素 → 智能体交互协议"的真实链路跑通。需求文档《项目环境搭建与强化学习环境搭建》把准备阶段切为"项目技术栈环境"（见 `docs/techs/项目技术栈.md`）与"强化学习智能体交互环境"两块，本文记录后者，其硬性要求为：

- 项目仅采用两个交互环境：DMControl 连续控制 `walker` 域 / `walk` 任务（walker-walk，像素渲染）；Atari 离散控制 `BattleZone`（gymnasium 注册名 `ALE/BattleZone-v5`）。
- 用**随机策略**编写测试代码，确认以上环境"没有问题"。
- 框架只允许以包管理器方式安装（`uv pip install` 进 `.venv`）；仓库内 `Arcade-Learning-Environment/`、`Gymnasium/`、`dm_control/` 等源码树仅作阅读参考，不得复制或在更不旁安装文件。

文献在评测层用了 DMControl 6 个任务与全部 26 个 Atari 游戏以支撑对比，本阶段缩到各取其一，已能覆盖文献依赖的两类协议能力：连续动作 + 从像素渲染（DMControl/SAC）、离散动作 + gym 像素帧（Atari/Efficient-Rainbow）。文献附录 A（Table 3）标注 DMControl 评测渲染为 100×100（下采样到 84×84），Atari 侧（Table 4）输入约 84×84；本冒烟按其上游渲染读取，把 84×84 裁剪/帧栈留给训练管线阶段处理，不在冒烟层越界。

## 二、技术选型

### 2.1 依赖与"本轮新增为零"结论

用 `uv pip list` 实测现有关键包：python 3.12（uv 管理）、numpy 2.5.2、gymnasium 1.3.0、ale-py 0.12.1、dm-control 1.0.45、mujoco 3.12.0、pygame 2.6.1；torch/torchvision 2.14/0.29 已在但本脚本不 import。

结论：**就随机策略冒烟验证，本轮无需新增任何第三方依赖**。文献附录 A 的向量化 random crop 用 `scikit-image.view_as_windows`，但它只属后续 CURL 训练增强的实现细节，与"环境可用性"无关，且可用 torch 等价实现，故不提前安装，避免无谓的环境膨胀。

### 2.2 冒烟脚本形态

- 纯 Python 可执行脚本：顶层 `main()` + `if __name__=="__main__"`，无测试框架运行时依赖，直接 `python src\smoke_environments.py`；内部断言失败抛异常、退出码非零，符合"真实环境集成验证"取向。
- 像素观测为主路径：不取 DMControl 低维状态作降级，带出"像素读取 → 协议 →(可选)渲染"整条链路。
- 固定种子复现：`np.random.default_rng(seed)`，action 由种子化 rng 采样。

### 2.3 显示能力选型（追加需求）

用布尔 `show` 控制是否显示画面，且要求 DMControl 也真实弹窗。取舍：

- DMControl：Mujoco 渲染无自带窗口，选用已装的 pygame（SDL2）自行建窗逐帧 `flip`，少装、且与后续帧栈可视化方向一致；
- ALE：直接走环境原生 `render_mode="human"`，交 gymnasium/ale 处理，不自建窗口；
- 两条链路互不共用窗口；`show` 只用其"是否把已采样帧送入窗口"，不参与观测/action/统计/断言，故 `show=False` 与改动前逐字节一致，可安全用于无头/CI。

## 三、技术实现细节

### 3.1 DMControl（walker-walk）链路

`suite.load("walker","walk")` 返回 `dm_control.rl.control.Environment`（本质 `dm_env.Environment` 物理封装）。`action_spec()` 为连续 BoundedArray：`shape=(6,), float64`, 界 `[-1,1]`；观测以 `TimeStep.observation` 给出低维项（`orientations/height/velocity` 等），**像素不在其中**。

像素需另经物理对象取：`env.physics.render(height=100,width=100,camera_id=0)` 返回 numpy，实测 `(100,100,3)` uint8，与本链路每次 reset/step 后取帧对应其 100×100 原渲染。随机 action 因 spec 无 `sample()`，用 `rng.uniform(spec.minimum,spec.maximum,size=spec.shape)` 生成；以 `TimeStep.last()` 判回合并叠加 `max_steps` 兜底。每帧做结构断言（shape/dtype/action 全有限/last 为布尔）。

单 episode 主循环（伪码）：

```
ts = env.reset()
img = env.physics.render(100,100,camera_id=0)
while not ts.last() and step < max_steps:
    action = rng.uniform(spec.minimum,spec.maximum,spec.shape)
    ts = env.step(action)
    img = env.physics.render(100,100,camera_id=0)
```

实时弹窗（show=True）用 `_PyGameViewer`：`set_mode((w*SCALE,h*SCALE))`（`RENDER_SCALE=4` 便于观察）；每帧 `pygame.surfarray.make_surface(frame.swapaxes(0,1))` 把 HWC 帧转 surface（surfarray 要求宽×高×通道，故先 swapaxes），`transform.smoothscale` 放大后 `blit` + `display.flip()`；事件侧只非阻塞 `pygame.event.get()`，遇 `QUIT` 抛 `SystemExit`；`try/finally` 中 `pygame.display.quit(); pygame.quit()` 回收。pygame import 延迟到需要显示的分支内，避免无头副作用。

### 3.2 ALE（BattleZone）链路

ALE 与 gymnasium 集成必须先注册：

```
import ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/BattleZone-v5")
```

漏 `register_envs` 直接 make 会抛 `NamespaceNotFound: ALE`。v5 变体 `reset` 返回 `(obs,info)`，`obs` 实测 `(210,160,3)` uint8，`action_space` 为 `Discrete(18)`。`step` 遵循 gymnasium 五元组 `(obs,reward,terminated,truncated,info)`；以 `rng.integers(0, env.action_space.n)` 代 `action_space.sample()` 求可复现，回合终止用 `terminated or truncated`，叠加 `max_steps` 兜底。show=True 时以 `render_mode="human"` 载入，交给 env 原生弹窗渲染，无需自行调 `render()`。

### 3.3 归一化、统一接口与入口

两协议差异（连续6维 / 离散18）各自封装在各 runner 内，不过度抽象成共享接口。统计统一：每 episode 累加奖励全程总步数，返回固定字段 `episodes / steps(total) / total_reward / mean_step_reward(=total/total_steps)`，使打印行同构可读。

两入口：CLI 用 `argparse` 布尔 `--show`（`store_true`，缺省 False）传给两个 runner；模块级 `run_dmcontrol_walker(seed=0, episodes=3, max_steps=1000, show=False)`、`run_ale_battlezone(seed=0, episodes=2, max_steps=2000, show=False)` 可被外 import，便于接测试/训练。各 helper 单一职责：`_sample_control_action`(DM 采样)、`_PyGameViewer`(dm 弹窗)、`_check_dm_frame`(帧断言)、`_format_row`(排印)。代码遵循高内聚低耦合、无死代码；类层级保持扁平，匹配冒烟脚本轻量定位。

## 四、运行与实测

```
# 默认（不弹窗）跑完 DMControl + ALE
.venv\Scripts\python.exe src\smoke_environments.py
# 实时弹窗
.venv\Scripts\python.exe src\smoke_environments.py --show
```

实测（本机、.venv）：

| 链路 | 观测 | 动作 | 结果 |
|---|---|---|---|
| walker/walk | 100×100×3 uint8 | 连续6维 | 像素链路正常（数千步秒级完成） |
| BattleZone-v5 | 210×160×3 uint8 | 离散18 | 协议正常 |

`--show` 实测 DMControl 用 pygame 建窗逐帧渲染、ALE 用 human 渲染均通过，`finally`/`close()` 处正常回收，无崩溃。

## 五、可扩展点与边界

- 使用固定种子可复现并便于对后续测试/训练作"环境守门"：默认（不带 `--show`）即可作可复现的环境状态回归守卫。
- 限制：DMControl 观测为 100×100 RGB 原渲染，3 帧堆叠与 84×84 crop 属训练管线；ALE v5 含 `repeat_action_probability` 等终端抖动，min/max-skip 帧栈留到实现 Efficient-Rainbow/CURL 时用 wrapper 配置；`show`/弹窗依赖有显示设备。
- 与约束兼容：源码树仅阅读未复制/import；实际依赖全部来自 `.venv` 的 wheel，本文所述 DMControl 摄像机参数、ALE 装载即据源码树与实测核验。
