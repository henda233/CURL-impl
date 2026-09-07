"""载入 data/sac-*/ checkpoint，渲染演示智能体走一遍 walker-walk。

默认载 best.pt（--latest 载最新）；--headless 不弹窗、只打印统计（供自检/CI）。
checkpoint 的 model 字段为 SAC actor 网络。
"""

import argparse
from pathlib import Path

import torch
from dm_control import suite

from networks import Actor, ACT_DIM
from pipeline import FrameStack
from train import render_frame, run_block, select_action
from viewer import PyGameViewer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="SAC 训练结果渲染演示")
    parser.add_argument("--dir", type=str, required=True, help="data/sac-<时间戳> 目录")
    parser.add_argument("--latest", action="store_true", help="载 latest.pt 而非 best.pt")
    parser.add_argument("--headless", action="store_true", help="不弹窗只打印")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_path = Path(args.dir) / ("latest.pt" if args.latest else "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = Actor(ACT_DIM)
    model.load_state_dict(ckpt["model"])
    print("[sac-demo] loaded %s meta=%s" % (ckpt_path, ckpt["meta"]))

    env = suite.load("walker", "walk", task_kwargs={"random": args.seed})
    stack = FrameStack()
    viewer = (
        None
        if args.headless
        else PyGameViewer((100, 100), "sac walker/walk %s" % args.dir)
    )

    try:
        ts = env.reset()
        stack.reset(render_frame(env))
        if viewer:
            viewer.show(render_frame(env))
        ep_return = 0.0
        env_steps = 0
        blocks = 0
        while not ts.last():
            action = select_action(model, stack.state(), sample=False)
            reward_sum, done, used = run_block(env, action, args.action_repeat)
            ep_return += reward_sum
            env_steps += used
            blocks += 1
            if done:
                break
            frame = render_frame(env)
            stack.push(frame)
            if viewer:
                viewer.show(frame)
        print("[sac-demo] DONE env_steps=%d blocks=%d return=%.3f" % (
            env_steps, blocks, ep_return))
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
