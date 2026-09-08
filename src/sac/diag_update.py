"""SAC update 段性能诊断：区分结构性算量与同步/搬运等待两类瓶颈画像。

以与 train.py 完全相同的网络结构、缓冲与入参构造在 GPU 上连续执行多次
update：逐次墙钟统计之外，用 torch.profiler 抓取一次稳定段 update，输出
CUDA device 累计时间及其占区间墙钟比例。device 占比高表明 GPU 计算主导
（结构性成本），占比低表明同步/搬运/主机端等待主导（工程性开销）。仅作
诊断工具，不改动 train.py 的常备 profile 路径。
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from train import ACT_DIM, Actor, Critic, _Temperature, train_step
from buffer import ReplayBuffer
from pipeline import CROP, FRAMES, STACK_C, obs_to_tensor

ROOT = Path(__file__).resolve().parents[2]
GAMMA = 0.99
TAU = 0.01
INIT_TEMP = 0.1
CAPACITY = 1024
FILL = 1000


def make_models(device):
    """按 train.py 同口径构建 actor/critic/target/温度与三个优化器。"""
    actor = Actor(ACT_DIM).to(device)
    critic = Critic(ACT_DIM).to(device)
    target = copy.deepcopy(critic)
    temp_module = _Temperature(math.log(INIT_TEMP)).to(device)
    log_alpha = temp_module()
    opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    opt_alpha = torch.optim.Adam(temp_module.parameters(), lr=1e-4, betas=(0.5, 0.999))
    return (actor, critic, target, log_alpha, opt_actor, opt_critic, opt_alpha)


def fill_buffer():
    """填充随机 uint8 帧栈数据至容量，供 sample 走真实 gather 路径。"""
    buf = ReplayBuffer(CAPACITY, ACT_DIM)
    obs = np.random.randint(0, 256, size=(FILL, STACK_C, CROP, CROP), dtype=np.uint8)
    frames = np.random.randint(0, 256, size=(FILL, FRAMES, CROP, CROP), dtype=np.uint8)
    act = np.random.uniform(-1.0, 1.0, size=(FILL, ACT_DIM)).astype(np.float32)
    for i in range(FILL):
        buf.add(obs[i], act[i], float(np.random.rand()), frames[i], False)
    return buf


def to_inputs(sample, device):
    """与 train.py update 段相同的 numpy→device 入参构造（含归一化）。"""
    obs, act, rew, next_obs, done = sample
    return (obs_to_tensor(obs, device), torch.from_numpy(act).to(device),
            torch.from_numpy(rew).to(device), obs_to_tensor(next_obs, device),
            torch.from_numpy(done).to(device))


def run_once(models, sample, device):
    """执行一次等价于 train.py update 段的完整区间，返回墙钟毫秒。"""
    actor, critic, target, log_alpha, opt_actor, opt_critic, opt_alpha = models
    t0 = time.perf_counter()
    train_step(actor, critic, target, log_alpha, opt_actor, opt_critic,
               opt_alpha, *to_inputs(sample, device), GAMMA, TAU, -ACT_DIM)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def summarize(ms_list):
    """输出逐次墙钟的分位数摘要。"""
    a = np.asarray(ms_list, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p10_ms": float(np.percentile(a, 10)),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="SAC update 段性能诊断（需 GPU，传 --gpu）")
    parser.add_argument("--gpu", action="store_true", help="使用 CUDA（需显式指定）")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8, help="预热 update 数，吸收首次编译，不计时")
    parser.add_argument("--steps", type=int, default=56, help="计时 update 数，中部一次走 profiler")
    parser.add_argument("--cpu-smoke", action="store_true",
                        help="无 CUDA 快速通路自检（覆盖 batch=16 warmup=1 steps=2）")
    args = parser.parse_args()
    if args.cpu_smoke:
        args.batch, args.warmup, args.steps = 16, 1, 2
    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requested but CUDA is not available")
    device = "cpu" if args.cpu_smoke else ("cuda" if args.gpu else None)
    if device is None:
        parser.error("诊断需测 CUDA device 时间，请传 --gpu；无 GPU 环境仅可用 --cpu-smoke 做通路自检")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    print("[diag] torch=%s cuda=%s cudnn=%s gpu=%s" % (
        torch.__version__, torch.version.cuda, cudnn_ver, gpu_name))
    print("[diag] cfg device=%s batch=%d warmup=%d steps=%d" % (
        device, args.batch, args.warmup, args.steps))

    models = make_models(device)
    buf = fill_buffer()
    for _ in range(args.warmup):
        run_once(models, buf.sample(args.batch), device)
    print("[diag] warmup done updates=%d" % args.warmup)

    prof_idx = args.steps // 2
    per_step = []
    prof = None
    for i in range(args.steps):
        sample = buf.sample(args.batch)
        if device == "cuda" and i == prof_idx:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                wall_ms = run_once(models, sample, device)
            device_ms = sum(e.self_device_time_total for e in p.key_averages()) / 1000.0
            prof = {"step": i, "wall_ms": wall_ms, "device_ms": device_ms,
                    "device_ratio": device_ms / wall_ms if wall_ms else 0.0}
        else:
            per_step.append(run_once(models, sample, device))

    stats = summarize(per_step)
    if prof is not None:
        print("[diag] profiler@step=%d wall_ms=%.2f device_ms=%.2f device_ratio=%.1f%%" % (
            prof["step"], prof["wall_ms"], prof["device_ms"], prof["device_ratio"] * 100.0))
    for i, ms in enumerate(per_step):
        print("[diag] step=%d wall_ms=%.2f" % (i, ms))
    print("[diag] stats n=%d mean_ms=%.2f p10=%.2f p50=%.2f p90=%.2f min=%.2f max=%.2f" % (
        stats["n"], stats["mean_ms"], stats["p10_ms"], stats["p50_ms"],
        stats["p90_ms"], stats["min_ms"], stats["max_ms"]))
    if prof is not None:
        ratio = prof["device_ratio"]
        verdict = "compute dominant" if ratio >= 0.6 else ("wait dominant" if ratio <= 0.3 else "mixed")
        print("[diag] readout device_ratio=%.0f%% %s" % (ratio * 100.0, verdict))

    if not args.cpu_smoke:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = ROOT / "data" / ("diag-update-%s" % stamp)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": cudnn_ver,
            "gpu": gpu_name, "cfg": vars(args), "profiler": prof,
            "per_step_ms": per_step, "stats": stats,
        }
        with open(out_dir / "diag.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("[diag] saved=%s" % (out_dir / "diag.json"))
    else:
        print("[diag] selfcheck ok")
    print("[diag] DONE")


if __name__ == "__main__":
    main()
