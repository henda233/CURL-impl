"""冒烟：DMControl(walker-walk) 与 ALE(BattleZone) 随机策略像素交互验证。

show=True 时实时弹窗显示画面（DMControl 用自带 pygame viewer，ALE 交给 env 的
human 渲染）。show 只门控画面显示，数值统计与观测采样保持一致。
"""

import numpy as np

RENDER_SCALE = 4


def _sample_control_action(env, rng):
    spec = env.action_spec()
    return rng.uniform(spec.minimum, spec.maximum, size=spec.shape)


class _PyGameViewer:
    def __init__(self, size, title):
        import pygame

        self._dim = (int(size[0] * RENDER_SCALE), int(size[1] * RENDER_SCALE))
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode(self._dim)

    def show(self, frame_hwc_uint8):
        import pygame
        import pygame.surfarray

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit("窗口已关闭")
        surface = pygame.surfarray.make_surface(frame_hwc_uint8.swapaxes(0, 1))
        scaled = pygame.transform.smoothscale(surface, self._dim)
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def close(self):
        import pygame

        pygame.display.quit()
        pygame.quit()


def _check_dm_frame(img):
    if img.shape != (100, 100, 3):
        raise AssertionError(f"dm_control: obs shape 异常 {img.shape}")
    if img.dtype != np.uint8:
        raise AssertionError(f"dm_control: obs dtype 异常 {img.dtype}")


def run_dmcontrol_walker(seed=0, episodes=3, max_steps=1000, show=False):
    from dm_control import suite

    env = suite.load("walker", "walk")
    rng = np.random.default_rng(seed)

    episode_returns = []
    total_steps = 0
    obs_like = None
    viewer = (
        None
        if not show
        else _PyGameViewer((100, 100), "dm_control walker/walk")
    )

    try:
        for _ in range(episodes):
            ts = env.reset()
            step_count = 0
            episode_return = 0.0
            img = env.physics.render(height=100, width=100, camera_id=0)
            obs_like = img
            _check_dm_frame(img)
            if viewer:
                viewer.show(img)
            while not ts.last() and step_count < max_steps:
                action = _sample_control_action(env, rng)
                if not np.isfinite(action).all():
                    raise AssertionError("dm_control: action 含非有限数")
                ts = env.step(action)
                img = env.physics.render(height=100, width=100, camera_id=0)
                _check_dm_frame(img)
                if viewer:
                    viewer.show(img)
                if not isinstance(ts.last(), bool):
                    raise AssertionError("dm_control: last() 非 bool")
                episode_return += 0.0 if ts.reward is None else float(ts.reward)
                step_count += 1
            episode_returns.append(episode_return)
            total_steps += step_count
    finally:
        if viewer:
            viewer.close()

    assert obs_like is not None, "未获取到像素观测"
    return {
        "key": "dmcontrol",
        "id": "walker/walk",
        "action_space": f"{env.action_spec().shape} float64[-1,1]",
        "obs_shape": obs_like.shape,
        "obs_dtype": str(obs_like.dtype),
        "episodes": len(episode_returns),
        "steps": total_steps,
        "total_reward": float(sum(episode_returns)),
        "mean_step_reward": float(sum(episode_returns) / max(total_steps, 1)),
    }


def run_ale_battlezone(seed=0, episodes=2, max_steps=2000, show=False):
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    make_kwargs = {"render_mode": "human"} if show else {}
    env = gym.make("ALE/BattleZone-v5", **make_kwargs)
    rng = np.random.default_rng(seed)

    episode_returns = []
    total_steps = 0
    obs_like = None

    for _ in range(episodes):
        obs, _ = env.reset(seed=seed)
        obs_like = obs
        done = truncated = False
        step_count = 0
        episode_return = 0.0
        while not (done or truncated) and step_count < max_steps:
            action = rng.integers(0, env.action_space.n)
            obs, reward, done, truncated, _ = env.step(action)
            episode_return += float(reward)
            step_count += 1
        episode_returns.append(episode_return)
        total_steps += step_count

    env.close()
    return {
        "key": "ale",
        "id": env.spec.id,
        "action_space": str(env.action_space),
        "obs_shape": obs_like.shape,
        "obs_dtype": str(obs_like.dtype),
        "episodes": len(episode_returns),
        "steps": total_steps,
        "total_reward": float(sum(episode_returns)),
        "mean_step_reward": float(sum(episode_returns) / max(total_steps, 1)),
    }


def _format_row(s):
    numerics = " ".join(
        f"{k}="
        f"{s[k]:.4g}"
        if isinstance(s[k], (int, float))
        else f"{k}={s[k]}"
        for k in ("episodes", "steps", "total_reward", "mean_step_reward")
    )
    return (
        f"[{s['key']}] {s['id']} | "
        f"obs={s['obs_shape']} {s['obs_dtype']} | "
        f"act={s['action_space']} | {numerics}"
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description="随机策略环境交互冒烟")
    p.add_argument("--show", action="store_true", default=False,
                   help="实时弹窗显示环境画面")
    args = p.parse_args()

    print("== smoke: random-policy env interaction ==")
    for runner, name in (
        (lambda show: run_dmcontrol_walker(show=show), "dmcontrol"),
        (lambda show: run_ale_battlezone(show=show), "ale"),
    ):
        stats = runner(args.show)
        print(_format_row(stats))


if __name__ == "__main__":
    main()
