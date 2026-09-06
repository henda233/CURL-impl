"""AC 基线网络：共享卷积段 C_dmc + h(1024)，actor(μ, log_std) 与 V 头分叉。

架构对应 docs/reports/模型架构报告.md §2.1 / §3.1；训练目标在 train.py。
"""

import math

import torch
from torch import nn

ACT_DIM = 6
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def tanh_normal_log_prob(mu, log_std, action):
    """tanh-squashed Gaussian 的 log π(a|s)；action ∈ (-1,1) 开区间。"""
    a = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    u = torch.atanh(a)
    std = torch.exp(log_std)
    logp = -0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
    return (logp - torch.log1p(-a * a)).sum(-1)


def gaussian_entropy(log_std):
    """高斯熵近似（tanh 压缩前的分布熵），仅供可选熵正则使用。"""
    return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)


class ActorCritic(nn.Module):
    """输入 (B,9,84,84) float32 归一化帧栈；输出 μ(B,6)、log_std(B,6)、V(B)。"""

    def __init__(self, act_dim=ACT_DIM):
        super().__init__()
        self.act_dim = act_dim
        self.body = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(39200, 1024), nn.ReLU(),
        )
        self.p_head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.v_head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.reset_log_std()

    def reset_log_std(self):
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, obs):
        h = self.body(obs)
        hp = self.p_head(h)
        mu = self.mu(hp)
        log_std = torch.clamp(self.log_std(hp), LOG_STD_MIN, LOG_STD_MAX)
        v = self.v_head(h).squeeze(-1)
        return mu, log_std, v
