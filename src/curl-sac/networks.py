"""CURL 网络（方案 A：全链路共享单一 f_q 编码器，RL 头建于 z(50) 之上）。

结构对应 docs/reports/模型架构报告.md §4/§5.2：f_q = C_dmc + Linear(39200→1024)+ReLU
+ Linear(1024→50) + LayerNorm + tanh ⇒ z(50)；actor 头与 critic 双 Q 头均建于 z(50)
之上。actor 不回传 f_q（优化器只含头参数）；critic 的 Bellman 与 InfoNCE 梯度经
train.py 联合步汇入 f_q。f_k 与 target critic 在 train.py 内经 deepcopy 装配并各自
EMA（τ=0.05 / τ=0.01）。W 为 InfoNCE 的 bilinear 相似度矩阵。
"""

import math

import torch
from torch import nn

ACT_DIM = 6
Z_DIM = 50
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
HEAD = 256


def make_cnn():
    """(B,9,84,84) → (B,39200) 的 4 层卷积段（无 padding，规格同 C_dmc）。"""
    return nn.Sequential(
        nn.Conv2d(9, 32, kernel_size=3, stride=2), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
        nn.Flatten(),
    )


class Encoder(nn.Module):
    """f_q：观测 → z(50)。C_dmc → Linear(39200→1024)+ReLU → Linear(1024→50) → LN → tanh。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            make_cnn(),
            nn.Linear(39200, 1024), nn.ReLU(),
            nn.Linear(1024, Z_DIM),
            nn.LayerNorm(Z_DIM),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class ActorHead(nn.Module):
    """z(50) → (μ, log_std)；log_std 前向 clamp 至 [-5,2]，零初始化置 σ=1。"""

    def __init__(self, act_dim=ACT_DIM):
        super().__init__()
        self.p_head = nn.Sequential(nn.Linear(Z_DIM, HEAD), nn.ReLU())
        self.mu = nn.Linear(HEAD, act_dim)
        self.log_std = nn.Linear(HEAD, act_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, z):
        h = self.p_head(z)
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
    """建于 z 上的双 Q 头：concat(z, a) → (Q1, Q2)，不含 f_q 参数。"""

    def __init__(self, act_dim=ACT_DIM):
        super().__init__()
        self.q1 = _QHead(Z_DIM + act_dim)
        self.q2 = _QHead(Z_DIM + act_dim)

    def forward(self, z, action):
        h = torch.cat([z, action], dim=1)
        return self.q1(h), self.q2(h)


class BilinearSimilarity(nn.Module):
    """InfoNCE 的 bilinear 内积矩阵 W(z_dim, z_dim)，按论文以 rand 口径初始化。"""

    def __init__(self, dim=Z_DIM):
        super().__init__()
        self.W = nn.Parameter(torch.rand(dim, dim))

    def forward(self, z_q, z_k):
        return torch.matmul(z_q, torch.matmul(self.W, z_k.t()))


class _Temperature(nn.Module):
    """可训练 log 温度标量，随网络统一 .to(device)；forward 返回叶子张量。"""

    def __init__(self, init_log_alpha):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor([init_log_alpha], dtype=torch.float32))

    def forward(self):
        return self.log_alpha


def reparam_sample(mu, log_std):
    """重参数采样：u = μ + σ·ε，a = tanh(u)，返回 (a, log π(a|s))，梯度可回传。

    logp 用采样噪声 ε 的噪声形式（-0.5ε² - log_std - log(1-a²) 校正）计算，而不是
    由 a=atanh 反推 u 再除 σ²：float32 下 tanh 饱和区（μ 越界且 σ 收缩）反推失真，
    二次项无界负向爆炸并污染 TD target（对齐官方 curl/curl_sac.py 的 gaussian_logprob）。"""
    noise = torch.randn_like(mu)
    action = torch.tanh(mu + torch.exp(log_std) * noise)
    a = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    logp = (-0.5 * noise.pow(2) - log_std
            - 0.5 * math.log(2 * math.pi) - torch.log1p(-a * a)).sum(-1)
    return action, logp


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
