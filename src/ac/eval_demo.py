"""载入 data/ac-*/ checkpoint，渲染演示智能体走一遍 walker-walk。

默认载 best.pt（--latest 载最新）；--headless 不弹窗、只打印统计（供自检/CI）。
"""

import argparse
from pathlib import Path

import torch
from dm_control import suite

from networks import ActorCritic
from pipeline import FrameStack
from train import render_frame, run_block, select_action
from viewer import PyGameViewer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="AC 训练结果渲染演示")
    parser.add_argument("--dir", type=str, required=True, help="data/ac-<时间戳> 目录")
    parser.add_argument("--latest", action="store_true", help="载 latest.pt 而非 best.pt")
    parser.add_argument("--headless", action="store_true", help="不弹窗只打印")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", action="store_true", help="使用 CUDA 推理（不传则 cuda 可用时自动启用）")
    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requested but CUDA is not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.dir) / ("latest.pt" if args.latest else "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = ActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    print("[ac-demo] loaded device=%s %s meta=%s" % (device, ckpt_path, ckpt["meta"]))

    env = suite.load("walker", "walk", task_kwargs={"random": args.seed})
    stack = FrameStack()
    viewer = (
        None
        if args.headless
        else PyGameViewer((100, 100), "ac walker/walk %s" % args.dir)
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
            action = select_action(model, stack.state(), explore=False)
            reward, done, used = run_block(env, stack, action, args.action_repeat)
            ep_return += reward
            env_steps += used
            blocks += 1
            if not done:
                frame = render_frame(env)
                stack.push(frame)
                if viewer:
                    viewer.show(frame)
            if done:
                break
        print("[ac-demo] DONE env_steps=%d blocks=%d return=%.3f" % (
            env_steps, blocks, ep_return))
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
