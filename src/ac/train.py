"""on-policy Advantage Actor-Critic 训练：DMControl walker-walk 像素观测。

数据流：每个 action-repeat 块产生一条 (s, a, r, done)；每 episode 结束后用
TD(0) advantage 更新一次：L_v = (V - (r + γV'))²，L_π = -mean(A · log π(a|s))。
步数按环境步(physics step)计数，默认 100_000（文献 DMControl100k，Table 3：
walker 的 action repeat=2、lr 1e-3、γ .99、eval 10 episodes）。
模型每 eval 节点存档到 data/ac-<时间戳>/{latest,best}.pt 并写 eval_history.jsonl；
控制台全部输出逐条 tee 到同目录 train_log.txt（追加）。

用法：
  python src/ac/train.py --smoke                      # 短跑自检
  python src/ac/train.py                              # 完整训练（默认 100K env steps）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if (sys.platform.startswith("linux") and "DISPLAY" not in os.environ
        and "MUJOCO_GL" not in os.environ):
    os.environ["MUJOCO_GL"] = "egl"

from dm_control import suite

import log_tee
import networks
from networks import ActorCritic
from pipeline import FrameStack

RENDER_H = RENDER_W = 100
ROOT = Path(__file__).resolve().parents[2]


def make_env(seed):
    return suite.load("walker", "walk", task_kwargs={"random": seed})


def render_frame(env):
    return env.physics.render(height=RENDER_H, width=RENDER_W, camera_id=0)


def select_action(model, state, explore):
    """state (9,84,84) uint8 -> action (6,) float64；explore 采样 tanh(μ+σε) 否则取 tanh(μ)。"""
    device = next(model.parameters()).device
    obs = torch.from_numpy(state).to(device=device).float().div_(255.0).unsqueeze(0)
    with torch.no_grad():
        mu, log_std, _ = model(obs)
        if explore:
            u = mu + torch.exp(log_std) * torch.randn_like(mu)
        else:
            u = mu
        action = torch.tanh(u)
    return action[0].cpu().numpy().astype(np.float64)


def run_block(env, stack, action, action_repeat):
    """执行一个决策块（重复 action_repeat 次），返回 (reward_sum, done, env_steps)。"""
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


def collect_episode(model, env, stack, action_repeat, env_budget):
    """收集至多一个 episode 的 (state, action, reward, done) 序列，不超 env_budget。"""
    buffer = []
    ts = env.reset()
    stack.reset(render_frame(env))
    ep_return = 0.0
    ep_env_steps = 0
    while not ts.last() and ep_env_steps + action_repeat <= env_budget:
        state = stack.state()
        action = select_action(model, state, explore=True)
        reward, done, env_steps = run_block(env, stack, action, action_repeat)
        ep_return += reward
        ep_env_steps += env_steps
        if not done:
            stack.push(render_frame(env))
        buffer.append((state, action, reward, done))
        if done:
            break
    return buffer, ep_return, ep_env_steps


def update(model, optimizer, buffer, gamma, entropy_coef):
    device = next(model.parameters()).device
    states = torch.from_numpy(np.stack([t[0] for t in buffer])).to(
        device=device).float().div_(255.0)
    actions = torch.from_numpy(np.stack([t[1] for t in buffer]).astype(np.float32)).to(device)
    rewards = torch.from_numpy(np.array([t[2] for t in buffer], dtype=np.float32)).to(device)
    done = torch.from_numpy(np.array([t[3] for t in buffer], dtype=np.bool_)).to(device)

    mu, log_std, v = model(states)
    with torch.no_grad():
        v_next = torch.cat([v[:-1], v.new_zeros(1)])
        target = rewards + gamma * v_next * (~done).float()
        advantage = target - v
    log_prob = networks.tanh_normal_log_prob(mu, log_std, actions)
    pg_loss = -(advantage * log_prob).mean()
    v_loss = ((v - target) ** 2).mean()
    entropy = networks.gaussian_entropy(log_std).mean()
    total = pg_loss + v_loss - entropy_coef * entropy

    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return (float(pg_loss.detach()), float(v_loss.detach()),
            float(entropy.detach()))


def evaluate(model, env, action_repeat, episodes):
    returns = []
    for _ in range(episodes):
        _, ep_return, _ = collect_episode(
            model, env, FrameStack(), action_repeat, env_budget=10_000)
        returns.append(ep_return)
    return returns


def save_checkpoint(path, model, meta):
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def main():
    parser = argparse.ArgumentParser(description="on-policy Advantage AC train (walker-walk)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--max-env-steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gpu", action="store_true", help="使用 CUDA 训练（需显式指定，否则用 CPU）")
    parser.add_argument("--smoke", action="store_true", help="短跑自检参数覆盖")
    args = parser.parse_args()
    if args.smoke:
        args.max_env_steps = 600
        args.eval_interval = 600
        args.eval_episodes = 1

    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requested but CUDA is not available")
    device = "cuda" if args.gpu else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_env(args.seed)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "data" / (f"ac-{stamp}-smoke" if args.smoke else f"ac-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_tee.start(out_dir / "train_log.txt")

    print("[ac-train] out=%s" % out_dir)
    print("[ac-train] cfg device=%s seed=%d max_env_steps=%d action_repeat=%d lr=%g gamma=%g "
          "entropy_coef=%g epochs=%d eval_interval=%d eval_episodes=%d" % (
              device, args.seed, args.max_env_steps, args.action_repeat, args.lr, args.gamma,
              args.entropy_coef, args.epochs, args.eval_interval, args.eval_episodes))

    stack = FrameStack()
    history = []
    best_mean = None
    env_steps = 0
    episodes = 0
    next_eval = min(args.eval_interval, args.max_env_steps)

    while env_steps < args.max_env_steps:
        buffer, ep_return, ep_env_steps = collect_episode(
            model, env, stack, args.action_repeat,
            args.max_env_steps - env_steps)
        env_steps += ep_env_steps
        if not buffer:
            break
        episodes += 1
        for _ in range(args.epochs):
            pg_loss, v_loss, entropy = update(
                model, optimizer, buffer, args.gamma, args.entropy_coef)
        print("[ac-train] ep=%d env_steps=%d ep_return=%.3f ep_env=%d "
              "pg=%.4g vf=%.4g ent=%.4g" % (
                  episodes, env_steps, ep_return, ep_env_steps,
                  pg_loss, v_loss, entropy))

        if env_steps >= next_eval:
            returns = evaluate(model, env, args.action_repeat, args.eval_episodes)
            mean_return = float(np.mean(returns))
            meta = {
                "seed": args.seed, "device": device, "env_steps": env_steps,
                "episodes": episodes, "mean_return": mean_return,
                "eval_episodes": len(returns), "action_repeat": args.action_repeat,
                "entropy_coef": args.entropy_coef,
            }
            save_checkpoint(out_dir / "latest.pt", model, meta)
            if best_mean is None or mean_return > best_mean:
                best_mean = mean_return
                save_checkpoint(out_dir / "best.pt", model, meta)
            history.append(meta)
            with open(out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(meta) + "\n")
            print("[ac-train] EVAL env_steps=%d mean_return=%.3f best=%.3f "
                  "saved=%s" % (env_steps, mean_return, best_mean, out_dir))
            next_eval = min(env_steps + args.eval_interval, args.max_env_steps)

    print("[ac-train] DONE env_steps=%d episodes=%d best_mean=%.3f out=%s" % (
        env_steps, episodes, best_mean if best_mean is not None else float("nan"),
        out_dir))
    log_tee.stop()


if __name__ == "__main__":
    main()
