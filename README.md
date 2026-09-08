# CURL框架实现 项目

CURL框架源于文献CURL: Contrastive Unsupervised Representations for Reinforcement Learning（该文献的md版本存储在docs\paper\paper.md下）。

## 1. CURL的基本定义

CURL作为一种框架或者说模型，其根本目的是：基于对比学习建立一个状态表征模型，实现将高维复杂的图像转换为低维简单的表征向量。

CURL本身不是RL模型，它作为一种“即插即用”表征编码器，可以灵活地接入以AC、DQN框架为主的强化学习算法中。

## 2. 项目目标

本项目的最终目标是基于文献内容从零实现CURL模型。但是在实现前，还需要实现如下模型：

- Actor-Critic模型：最基础的AC模型；
- DQN；
- SAC模型：Soft Actor-Critic模型。

然后将其进行对比。

## 3. 项目目录

```markdown
docs/
    paper/                          # 存储与本项目相关的文献，md格式
    notes/                          # 开发复盘文档
    techs/                          # 技术文档
    functions/                      # 需求文档
    gymnasium/                      # gymnasium框架的相关文档
    curl/                           # CURL文献官方团队实现源码
dm_control/                         # DMControl框架的源码仓库
Gymnasium/                          # Gymnasium框架的源码仓库
Arcade-Learning-Environment/        # ALE框架的源码仓库
```