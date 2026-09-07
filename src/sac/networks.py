"""SAC 网络（方案 B：actor 与 critic 同构 C_dmc→h(1024)，但各自持有独立参数）。

结构对应 docs/reports/模型架构报告.md §3.2：actor = body + μ/log_std 头；
critic = 自建 body，其上挂 Q1/Q2 双头（输入 concat(h, a)）；target critic 由
critic 的同构实例经 EMA(τ) 更新获得。actor 与 critic 之间不共享任何参数。
"""

import math

import torch
from torch import nn

ACT_DIM = 6
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
HIDDEN = 1024
HEAD = 256


def make_body():
    """(B,9,84,84) → h(B,1024)，每次调用自建参数。"""
    return nn.Sequential(
        nn.Conv2d(9, 32, kernel_size=3, stride=2), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(39200, HIDDEN), nn.ReLU(),
    )


def tanh_normal_log_prob(mu, log_std, action):
    """tanh-squashed Gaussian 的 log π(a|s)；action ∈ (-1,1) 开区间。"""
    a = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    u = torch.atanh(a)
    std = torch.exp(log_std)
    logp = -0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
    return (logp - torch.log1p(-a * a)).sum(-1)


def reparam_sample(mu, log_std):
    """SAC 重参数采样：u = μ + σ·ε，a = tanh(u)，返回 (a, log π(a|s))，梯度可回传。"""
    u = mu + torch.exp(log_std) * torch.randn_like(mu)
    action = torch.tanh(u)
    return action, tanh_normal_log_prob(mu, log_std, action)


class Actor(nn.Module):
    """观测 → (μ, log_std)（log_std 前向 clamp 至 [-5,2]，零初始化置 σ=1）。"""

    def __init__(self, act_dim=ACT_DIM):
        super().__init__()
        self.body = make_body()
        self.p_head = nn.Sequential(nn.Linear(HIDDEN, HEAD), nn.ReLU())
        self.mu = nn.Linear(HEAD, act_dim)
        self.log_std = nn.Linear(HEAD, act_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, obs):
        h = self.p_head(self.body(obs))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std


class _QHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HEAD), nn.ReLU(), nn.Linear(HEAD, 1)
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class Critic(nn.Module):
    """共享自建 body，双 Q 头 concat(h, a)：forward → (Q1, Q2)。"""

    def __init__(self, act_dim=ACT_DIM):
        super().__init__()
        self.body = make_body()
        self.q1 = _QHead(HIDDEN + act_dim)
        self.q2 = _QHead(HIDDEN + act_dim)

    def forward(self, obs, action):
        h = torch.cat([self.body(obs), action], dim=1)
        return self.q1(h), self.q2(h)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
