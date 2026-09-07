"""off-policy SAC（双 Q + target EMA + 自动温度）训练：DMControl walker-walk 像素观测。

机制与超参锚定 CURL 论文 Table 3（walker 行）：replay 100k、initial steps 1000、
batch 512、action repeat 2、lr 1e-3、α lr 1e-4 init 0.1、Q-EMA τ=0.01、γ .99、
eval 10 episodes。观测管线同 AC 基线：center crop 84×84 + 3 帧栈，训练无 random
crop（增强变量留待 CURL 阶段）。探针实测 walker 的 episode 只在 time-limit
(1000 env steps) 截断，无真终止 → 一律按 truncated 处理，Q target 继续 bootstrap。

训练环境收集与评估使用独立 env 实例；收集侧以 time-limit 为单位推进，评估不打断
训练 episode。每收集一个 action-repeat 块后做一次更新（sample batch → critic →
soft update target → actor → temperature）。模型每 eval 节点存档到
data/sac-<时间戳>/{latest,best}.pt 并写 eval_history.jsonl（checkpoint 只含 actor
与 meta，供演示载入）。

用法：
  python src/sac/train.py --smoke              # 短跑自检（小 batch/短步数）
  python src/sac/train.py                      # 完整训练（默认 100K env steps）
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from dm_control import suite

from networks import Actor, Critic, reparam_sample, soft_update
from pipeline import FrameStack, center_crop, obs_to_tensor
from buffer import ReplayBuffer

ACT_DIM = 6
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


def select_action(actor, state_u8, sample):
    """state (9,84,84) uint8 → action (6,) float64；sample 采样 tanh(μ+σε) 否则取 tanh(μ)。"""
    device = next(actor.parameters()).device
    obs = obs_to_tensor(state_u8).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, log_std = actor(obs)
        if sample:
            u = mu + torch.exp(log_std) * torch.randn_like(mu)
        else:
            u = mu
        action = torch.tanh(u)
    return action[0].cpu().numpy().astype(np.float64)


def train_step(actor, critic, target, log_alpha, opt_actor, opt_critic, opt_alpha,
               obs, act, rew, next_obs, done, gamma, tau, target_entropy):
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        a2, logp2 = reparam_sample(*actor(next_obs))
        q1n, q2n = target(next_obs, a2)
        y = rew + gamma * (~done).float() * (torch.min(q1n, q2n) - alpha * logp2)
    q1, q2 = critic(obs, act)
    q_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
    opt_critic.zero_grad()
    q_loss.backward()
    opt_critic.step()
    soft_update(target, critic, tau)

    a, logp = reparam_sample(*actor(obs))
    q1a, q2a = critic(obs, a)
    pi_loss = (alpha * logp - torch.min(q1a, q2a)).mean()
    opt_actor.zero_grad()
    pi_loss.backward()
    opt_actor.step()

    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    opt_alpha.zero_grad()
    alpha_loss.backward()
    opt_alpha.step()
    temp = float(log_alpha.detach().exp())
    return float(q_loss.detach()), float(pi_loss.detach()), \
        float(alpha_loss.detach()), temp


def evaluate(actor, action_repeat, episodes, seed):
    """用独立 env 评估 mean-action 累计回报（不触碰训练 env）。"""
    env = make_env(seed)
    returns = []
    for _ in range(episodes):
        ts = env.reset()
        stack = FrameStack()
        stack.reset(render_frame(env))
        ep_return = 0.0
        while not ts.last():
            action = select_action(actor, stack.state(), sample=False)
            reward_sum, last, _ = run_block(env, action, action_repeat)
            ep_return += reward_sum
            if last:
                break
            stack.push(render_frame(env))
        returns.append(ep_return)
    env.close()
    return returns


def save_checkpoint(path, model, meta):
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def main():
    parser = argparse.ArgumentParser(description="SAC train (walker-walk pixels)")
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
    parser.add_argument("--gpu", action="store_true", help="使用 CUDA 训练（不传则 cuda 可用时自动启用）")
    parser.add_argument("--smoke", action="store_true", help="短跑自检参数覆盖")
    args = parser.parse_args()
    if args.smoke:
        args.max_env_steps = 120
        args.initial_steps = 50
        args.replay_size = 1024
        args.batch_size = 16
        args.eval_interval = 120
        args.eval_episodes = 1

    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requested but CUDA is not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_env(args.seed)
    actor = Actor(ACT_DIM).to(device)
    critic = Critic(ACT_DIM).to(device)
    target = copy.deepcopy(critic)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)
    log_alpha = torch.tensor([math.log(args.init_temperature)], requires_grad=True).to(device)
    opt_alpha = torch.optim.Adam([log_alpha], lr=args.alpha_lr, betas=(0.5, 0.999))
    buffer = ReplayBuffer(args.replay_size, ACT_DIM)
    target_entropy = -ACT_DIM

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "data" / (f"sac-{stamp}-smoke" if args.smoke else f"sac-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[sac-train] out=%s" % out_dir)
    print("[sac-train] cfg device=%s seed=%d max_env_steps=%d action_repeat=%d lr=%g alpha_lr=%g "
          "gamma=%g tau=%g init_temperature=%g initial_steps=%d batch_size=%d "
          "replay_size=%d eval_interval=%d eval_episodes=%d" % (
              device, args.seed, args.max_env_steps, args.action_repeat, args.lr, args.alpha_lr,
              args.gamma, args.tau, args.init_temperature, args.initial_steps,
              args.batch_size, args.replay_size, args.eval_interval, args.eval_episodes))

    history = []
    best_mean = None
    episodes = 0
    env_steps = 0
    next_eval = min(args.eval_interval, args.max_env_steps)

    while env_steps < args.max_env_steps:
        budget = (args.max_env_steps - env_steps) // args.action_repeat
        if budget <= 0:
            break
        ts = env.reset()
        stack = FrameStack()
        stack.reset(render_frame(env))
        ep_return = 0.0
        ep_env = 0
        nblocks = 0
        qsum = psum = asum = tsum = nloss = 0
        while not ts.last() and nblocks < budget:
            state = stack.state()
            if env_steps < args.initial_steps:
                action = random_action()
            else:
                action = select_action(actor, state, sample=True)
            reward_sum, last, used = run_block(env, action, args.action_repeat)
            frame = render_frame(env)
            stack.push(frame)
            buffer.add(state, action.astype(np.float32), reward_sum,
                       center_crop(frame).transpose(2, 0, 1), done=False)
            env_steps += used
            ep_return += reward_sum
            ep_env += used
            nblocks += 1

            if env_steps >= args.initial_steps and len(buffer) >= args.batch_size:
                obs, act, rew, next_obs, done = buffer.sample(args.batch_size)
                ql, pl, al, temp = train_step(
                    actor, critic, target, log_alpha, opt_actor, opt_critic, opt_alpha,
                    obs_to_tensor(obs).to(device), torch.from_numpy(act).to(device),
                    torch.from_numpy(rew).to(device), obs_to_tensor(next_obs).to(device),
                    torch.from_numpy(done).to(device),
                    args.gamma, args.tau, target_entropy)
                qsum += ql
                psum += pl
                asum += al
                tsum += temp
                nloss += 1
            if last:
                break
        episodes += 1
        if nloss:
            print("[sac-train] ep=%d env_steps=%d ep_return=%.3f ep_env=%d "
                  "q=%.4g pi=%.4g temp=%.4g" % (
                      episodes, env_steps, ep_return, ep_env,
                      qsum / nloss, psum / nloss, tsum / nloss))
        else:
            print("[sac-train] ep=%d env_steps=%d ep_return=%.3f ep_env=%d (warmup)" % (
                episodes, env_steps, ep_return, ep_env))

        if env_steps >= args.initial_steps and env_steps >= next_eval:
            returns = evaluate(actor, args.action_repeat, args.eval_episodes, args.seed)
            mean_return = float(np.mean(returns))
            meta = {
                "seed": args.seed, "device": device, "env_steps": env_steps,
                "episodes": episodes, "mean_return": mean_return,
                "eval_episodes": len(returns), "action_repeat": args.action_repeat,
                "temperature": tsum / nloss if nloss else float(log_alpha.detach().exp()),
                "lr": args.lr, "alpha_lr": args.alpha_lr, "gamma": args.gamma,
                "tau": args.tau, "batch_size": args.batch_size,
                "initial_steps": args.initial_steps,
            }
            save_checkpoint(out_dir / "latest.pt", actor, meta)
            if best_mean is None or mean_return > best_mean:
                best_mean = mean_return
                save_checkpoint(out_dir / "best.pt", actor, meta)
            history.append(meta)
            with open(out_dir / "eval_history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(meta) + "\n")
            print("[sac-train] EVAL env_steps=%d mean_return=%.3f best=%.3f "
                  "saved=%s" % (env_steps, mean_return, best_mean, out_dir))
            next_eval = min(env_steps + args.eval_interval, args.max_env_steps)
    env.close()

    print("[sac-train] DONE env_steps=%d episodes=%d best_mean=%.3f out=%s" % (
        env_steps, episodes, best_mean if best_mean is not None else float("nan"),
        out_dir))


if __name__ == "__main__":
    main()
