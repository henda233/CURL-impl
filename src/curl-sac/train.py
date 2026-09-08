"""CURL（方案 A：CURL + SAC，共享 query encoder）训练：DMControl walker-walk 像素观测。

机制与超参锚定 CURL 论文：f_q 共享编码器输出 z(50)（C_dmc→1024→50 LN+tanh），
actor/critic 头建于 z 之上；训练期 random crop（100→84，栈内坐标一致）作用于收集与
更新两侧，评估 center crop。每 block 联合步：Bellman 与 InfoNCE 同 loss 同 step 更新
(θ_enc + Q 头 + W)（等权，无平衡系数），随后 target critic（τ=0.01）与 key encoder
f_k（τ=0.05）各自 EMA；actor 与温度独立步，actor 不回传 f_q（z 前向 detach）。
InfoNCE 的负样本取同 batch 其余样本（无 memory bank），labels=arange(B)。

观测管线与存档设计沿用 SAC：环境渲染 100×100 原帧入 buffer（uint8），latest/best.pt
每 eval 节点存档 {f_q, actor_head} + meta 并写 eval_history.jsonl（checkpoint 供演示
载入）；控制台输出 tee 到 train_log.txt。walker 的 episode 只在 time-limit 截断，
一律按 truncated 处理，Q target 继续 bootstrap。

用法：
  python src/curl-sac/train.py --smoke              # 短跑自检（小 batch/短步数）
  python src/curl-sac/train.py                      # 完整训练（默认 100K env steps）
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if (sys.platform.startswith("linux") and "DISPLAY" not in os.environ
        and "MUJOCO_GL" not in os.environ):
    os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite

from networks import (
    ActorHead, BilinearSimilarity, Critic, Encoder, _Temperature,
    reparam_sample, soft_update)
from pipeline import FrameStack, center_crop, obs_to_tensor, random_crop
from buffer import ReplayBuffer
import log_tee
from profiler import Profile

ACT_DIM = 6
KEY_TAU = 0.05
RENDER_H = RENDER_W = 100
ROOT = Path(__file__).resolve().parents[2]


def make_env(seed):
    return suite.load("walker", "walk", task_kwargs={"random": seed})


def render_frame(env):
    return env.physics.render(height=RENDER_H, width=RENDER_W, camera_id=0)


def run_block(env, action, action_repeat):
    """执行一个决策块（重复 action_repeat 次），返回 (reward_sum, last, env_steps)。"""
    reward_sum = 0.0
    env_steps = 0
    ts = None
    for _ in range(action_repeat):
        ts = env.step(action)
        env_steps += 1
        reward_sum += float(ts.reward)
        if ts.last():
            break
    return reward_sum, ts.last(), env_steps


def random_action():
    return np.random.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float64)


def select_action(encoder, actor, state_u8, sample, crop):
    """state (9,100,100) uint8 → action (6,) float64。crop 为 random_crop 或 center_crop。
    sample 采样 tanh(μ+σε) 否则取 tanh(μ)。"""
    device = next(encoder.parameters()).device
    obs = obs_to_tensor(crop(state_u8)[np.newaxis], device)
    with torch.no_grad():
        mu, log_std = actor(encoder(obs))
        if sample:
            u = mu + torch.exp(log_std) * torch.randn_like(mu)
        else:
            u = mu
        action = torch.tanh(u)
    return action[0].cpu().numpy().astype(np.float64)


def train_step(encoder, critic, target_encoder, target_critic, key_encoder, bilinear,
               actor, log_alpha, opt_main, opt_actor, opt_alpha,
               obs_a, obs_k, next_a, act, rew, done, gamma, tau, target_entropy,
               diag=None):
    """一次更新循环：联合步（Bellman+InfoNCE）→ target EMA → f_k EMA → actor → 温度。

    返回 (q_loss, curl_loss, pi_loss, alpha_loss, temperature)。diag 非 None 时把
    TD/actor/表征/InfoNCE 各量写入该 dict（旁路收集，不改变数值语义），供 --diag 落盘。
    """
    alpha = log_alpha.exp().detach()
    z_a = encoder(obs_a)
    z_k = key_encoder(obs_k).detach()
    with torch.no_grad():
        z_n = encoder(next_a)
        a2, logp2 = reparam_sample(*actor(z_n))
        z_t = target_encoder(next_a)
        qt1, qt2 = target_critic(z_t, a2)
        y = rew + gamma * (~done).float() * (torch.min(qt1, qt2) - alpha * logp2)

    q1, q2 = critic(z_a, act)
    q_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
    logits = bilinear(z_a, z_k)
    logits = logits - logits.max(dim=1, keepdim=True).values
    labels = torch.arange(logits.shape[0], device=logits.device)
    curl_loss = F.cross_entropy(logits, labels)
    opt_main.zero_grad()
    (q_loss + curl_loss).backward()
    opt_main.step()
    soft_update(target_encoder, encoder, tau)
    soft_update(target_critic, critic, tau)
    soft_update(key_encoder, encoder, KEY_TAU)

    z_a = z_a.detach()
    mu, log_std = actor(z_a)
    a, logp = reparam_sample(mu, log_std)
    q1a, q2a = critic(z_a, a)
    pi_loss = (alpha * logp - torch.min(q1a, q2a)).mean()
    opt_actor.zero_grad()
    pi_loss.backward()
    opt_actor.step()

    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad()
    alpha_loss.backward()
    opt_alpha.step()
    temp = float(log_alpha.detach().exp())
    if diag is not None:
        qm = torch.min(q1, q2)
        qt = torch.min(qt1, qt2)
        with torch.no_grad():
            diag["y_mean"] = float(y.mean())
            diag["y_absmax"] = float(y.abs().max())
            diag["qmin_mean"] = float(qm.mean())
            diag["qmin_absmax"] = float(qm.abs().max())
            diag["tqmin_mean"] = float(qt.mean())
            diag["tqmin_absmax"] = float(qt.abs().max())
            diag["logp2_mean"] = float(logp2.mean())
            diag["logp2_min"] = float(logp2.min())
            diag["logp_mean"] = float(logp.mean())
            diag["logp_min"] = float(logp.min())
            diag["mu_absmax"] = float(mu.abs().max())
            diag["logstd_absmax"] = float(log_std.abs().max())
            diag["z_absmax"] = float(z_a.abs().max())
            diag["z_std_mean"] = float(z_a.std(dim=0).mean())
            diag["logits_absmax"] = float(logits.abs().max())
            diag["logits_spread"] = float(
                (logits.max(dim=1).values - logits.min(dim=1).values).mean())
    return (float(q_loss.detach()), float(curl_loss.detach()),
            float(pi_loss.detach()), float(alpha_loss.detach()), temp)


def evaluate(encoder, actor, action_repeat, episodes, seed):
    """用独立 env + center crop 评估 mean-action 累计回报（不触碰训练 env）。"""
    env = make_env(seed)
    returns = []
    for _ in range(episodes):
        ts = env.reset()
        stack = FrameStack()
        stack.reset(render_frame(env))
        ep_return = 0.0
        while not ts.last():
            action = select_action(encoder, actor, stack.state(), sample=False,
                                   crop=center_crop)
            reward_sum, last, _ = run_block(env, action, action_repeat)
            ep_return += reward_sum
            if last:
                break
            stack.push(render_frame(env))
        returns.append(ep_return)
    env.close()
    return returns


def save_checkpoint(path, encoder, actor, meta):
    torch.save({"model": {"encoder": encoder.state_dict(), "actor": actor.state_dict()},
                "meta": meta}, path)


def flush_diag(path, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rows.clear()


def main():
    parser = argparse.ArgumentParser(description="CURL train (walker-walk pixels)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-env-steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--initial-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--gpu", action="store_true", help="使用 CUDA 训练（需显式指定，否则用 CPU）")
    parser.add_argument("--smoke", action="store_true", help="短跑自检参数覆盖")
    parser.add_argument("--profile", action="store_true",
                        help="分段计时剖析：覆盖 max_env_steps=2000、跳过 eval、写 profile.json")
    parser.add_argument("--diag", action="store_true",
                        help="每更新步旁路记录 TD/actor/表征/InfoNCE 数值，写 update_diag.jsonl")
    args = parser.parse_args()
    if args.smoke:
        args.max_env_steps = 60
        args.initial_steps = 20
        args.replay_size = 256
        args.batch_size = 8
        args.eval_interval = 60
        args.eval_episodes = 1
    if args.profile and not args.smoke:
        args.max_env_steps = 2000

    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requested but CUDA is not available")
    device = "cuda" if args.gpu else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_env(args.seed)
    encoder = Encoder().to(device)
    actor = ActorHead(ACT_DIM).to(device)
    critic = Critic(ACT_DIM).to(device)
    target_encoder = copy.deepcopy(encoder)
    target_critic = copy.deepcopy(critic)
    key_encoder = copy.deepcopy(encoder)
    bilinear = BilinearSimilarity().to(device)
    opt_main = torch.optim.Adam(
        list(encoder.parameters()) + list(critic.parameters())
        + list(bilinear.parameters()), lr=args.lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    temp_module = _Temperature(math.log(args.init_temperature)).to(device)
    log_alpha = temp_module()
    opt_alpha = torch.optim.Adam(
        temp_module.parameters(), lr=args.alpha_lr, betas=(0.5, 0.999))
    buffer = ReplayBuffer(args.replay_size, ACT_DIM)
    target_entropy = -ACT_DIM

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "data" / (
        f"curl-sac-{stamp}-smoke" if args.smoke else f"curl-sac-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_tee.start(out_dir / "train_log.txt")

    print("[curl-sac-train] out=%s" % out_dir)
    print("[curl-sac-train] cfg device=%s profile=%s diag=%s seed=%d max_env_steps=%d action_repeat=%d "
          "lr=%g alpha_lr=%g gamma=%g tau=%g key_tau=%g init_temperature=%g "
          "initial_steps=%d batch_size=%d replay_size=%d eval_interval=%d "
          "eval_episodes=%d" % (
              device, args.profile, args.diag, args.seed,
              args.max_env_steps, args.action_repeat, args.lr, args.alpha_lr,
              args.gamma, args.tau, KEY_TAU, args.init_temperature,
              args.initial_steps, args.batch_size, args.replay_size,
              args.eval_interval, args.eval_episodes))

    history = []
    diag_rows = []
    n_diag = 0
    diag_path = out_dir / "update_diag.jsonl"
    best_mean = None
    episodes = 0
    env_steps = 0
    next_eval = min(args.eval_interval, args.max_env_steps)
    profile = Profile(args.profile)
    wall0 = time.perf_counter()

    while env_steps < args.max_env_steps:
        budget = (args.max_env_steps - env_steps) // args.action_repeat
        if budget <= 0:
            break
        r0 = time.perf_counter()
        ts = env.reset()
        stack = FrameStack()
        stack.reset(render_frame(env))
        profile.record_reset(time.perf_counter() - r0)
        ep_return = 0.0
        ep_env = 0
        nblocks = 0
        qsum = csum = psum = asum = tsum = nloss = 0
        while not ts.last() and nblocks < budget:
            profile.set_stage("warmup" if env_steps < args.initial_steps else "train")
            profile.start("policy")
            state = stack.state()
            if env_steps < args.initial_steps:
                action = random_action()
            else:
                action = select_action(encoder, actor, state, sample=True,
                                       crop=random_crop)
            profile.stop("policy")
            profile.start("phys")
            reward_sum, last, used = run_block(env, action, args.action_repeat)
            profile.stop("phys")
            profile.start("render")
            frame = render_frame(env)
            profile.stop("render")
            profile.start("obs")
            stack.push(frame)
            buffer.add(state, action.astype(np.float32), reward_sum,
                       frame.transpose(2, 0, 1), done=False)
            profile.stop("obs")
            env_steps += used
            ep_return += reward_sum
            ep_env += used
            nblocks += 1

            if env_steps >= args.initial_steps and len(buffer) >= args.batch_size:
                profile.start("sample")
                obs, act, rew, next_obs, done = buffer.sample(args.batch_size)
                obs_a = random_crop(obs)
                obs_k = random_crop(obs)
                next_a = random_crop(next_obs)
                profile.stop("sample")
                if args.profile and device == "cuda":
                    torch.cuda.synchronize()
                profile.start("update")
                diag = {} if args.diag else None
                ql, cl, pl, al, temp = train_step(
                    encoder, critic, target_encoder, target_critic, key_encoder,
                    bilinear, actor, log_alpha, opt_main, opt_actor, opt_alpha,
                    obs_to_tensor(obs_a, device), obs_to_tensor(obs_k, device),
                    obs_to_tensor(next_a, device), torch.from_numpy(act).to(device),
                    torch.from_numpy(rew).to(device), torch.from_numpy(done).to(device),
                    args.gamma, args.tau, target_entropy, diag=diag)
                if args.profile and device == "cuda":
                    torch.cuda.synchronize()
                profile.stop("update")
                if args.diag:
                    row = {"step": n_diag, "env_steps": env_steps}
                    row.update(diag)
                    row.update(q=ql, curl=cl, pi=pl, alpha_loss=al, temp=temp)
                    diag_rows.append(row)
                    n_diag += 1
                    if n_diag % 500 == 0:
                        flush_diag(diag_path, diag_rows)
                qsum += ql
                csum += cl
                psum += pl
                asum += al
                tsum += temp
                nloss += 1
            if last:
                break
        episodes += 1
        if nloss:
            print("[curl-sac-train] ep=%d env_steps=%d ep_return=%.3f ep_env=%d "
                  "q=%.4g curl=%.4g pi=%.4g temp=%.4g" % (
                      episodes, env_steps, ep_return, ep_env,
                      qsum / nloss, csum / nloss, psum / nloss, tsum / nloss))
        else:
            print("[curl-sac-train] ep=%d env_steps=%d ep_return=%.3f ep_env=%d (warmup)" % (
                episodes, env_steps, ep_return, ep_env))

        if env_steps >= args.initial_steps and env_steps >= next_eval and not args.profile:
            returns = evaluate(encoder, actor, args.action_repeat, args.eval_episodes,
                               args.seed)
            mean_return = float(np.mean(returns))
            meta = {
                "seed": args.seed, "device": device, "env_steps": env_steps,
                "episodes": episodes, "mean_return": mean_return,
                "eval_episodes": len(returns), "action_repeat": args.action_repeat,
                "temperature": tsum / nloss if nloss else float(log_alpha.detach().exp()),
                "curl_loss": csum / nloss if nloss else None,
                "lr": args.lr, "alpha_lr": args.alpha_lr, "gamma": args.gamma,
                "tau": args.tau, "key_tau": KEY_TAU, "batch_size": args.batch_size,
                "initial_steps": args.initial_steps,
            }
            save_checkpoint(out_dir / "latest.pt", encoder, actor, meta)
            if best_mean is None or mean_return > best_mean:
                best_mean = mean_return
                save_checkpoint(out_dir / "best.pt", encoder, actor, meta)
            history.append(meta)
            with open(out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(meta) + "\n")
            print("[curl-sac-train] EVAL env_steps=%d mean_return=%.3f best=%.3f "
                  "saved=%s" % (env_steps, mean_return, best_mean, out_dir))
            next_eval = min(env_steps + args.eval_interval, args.max_env_steps)
    env.close()
    if args.diag:
        flush_diag(diag_path, diag_rows)
        print("[curl-sac-train] DIAG rows=%d saved=%s" % (n_diag, diag_path))
    wall = time.perf_counter() - wall0
    if args.profile:
        for line in profile.report_lines(env_steps, episodes, wall):
            print("[curl-sac-train] %s" % line)
        with open(out_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(env_steps, episodes, wall), f, indent=2)
        print("[curl-sac-train] PROFILE saved=%s" % (out_dir / "profile.json"))

    print("[curl-sac-train] DONE env_steps=%d episodes=%d best_mean=%.3f out=%s" % (
        env_steps, episodes, best_mean if best_mean is not None else float("nan"),
        out_dir))
    log_tee.stop()


if __name__ == "__main__":
    main()
