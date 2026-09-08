"""profile 分段墙钟统计：按 warmup/train 两阶段累计各段耗时，输出终端报表与 JSON。"""

import time

COLLECT_SECTIONS = ("phys", "render", "obs", "policy")
ALL_SECTIONS = COLLECT_SECTIONS + ("sample", "update")


class Profile:
    """收集 (stage, section) 累计耗时与次数；disabled 时各方法为 no-op。

    stage 由调用方按 env_steps 逐块 set_stage；reset 单独累计不归阶段。
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self._running = {}
        self._total = {}
        self._count = {}
        self._stage = "warmup"
        self.reset_seconds = 0.0
        self.reset_count = 0

    def set_stage(self, stage):
        self._stage = stage

    def start(self, section):
        if self.enabled:
            self._running[section] = time.perf_counter()

    def stop(self, section):
        if self.enabled:
            dt = time.perf_counter() - self._running.pop(section)
            key = "%s.%s" % (self._stage, section)
            self._total[key] = self._total.get(key, 0.0) + dt
            self._count[key] = self._count.get(key, 0) + 1

    def record_reset(self, seconds):
        if self.enabled:
            self.reset_seconds += seconds
            self.reset_count += 1

    def _mean_ms(self, stage, section):
        key = "%s.%s" % (stage, section)
        n = self._count.get(key, 0)
        return 0.0 if not n else self._total[key] / n * 1000.0

    def blocks(self, stage):
        return self._count.get("%s.render" % stage, 0)

    def collect_mean_ms(self, stage):
        return sum(self._mean_ms(stage, s) for s in COLLECT_SECTIONS)

    def report_lines(self, env_steps, episodes, wall_seconds):
        lines = [
            "PROFILE wall=%.1fs env_steps=%d episodes=%d env_steps/s=%.2f" % (
                wall_seconds, env_steps, episodes,
                env_steps / wall_seconds if wall_seconds else 0.0)]
        if self.reset_count:
            lines.append("PROFILE reset mean=%.2fms n=%d" % (
                self.reset_seconds / self.reset_count * 1000.0, self.reset_count))
        for stage in ("warmup", "train"):
            lines.append("PROFILE stage=%s blocks=%d collect_mean/block=%.2fms" % (
                stage, self.blocks(stage), self.collect_mean_ms(stage)))
            for s in ALL_SECTIONS:
                n = self._count.get("%s.%s" % (stage, s), 0)
                if n:
                    lines.append("PROFILE   %-7s mean=%.3fms n=%d" % (
                        s, self._mean_ms(stage, s), n))
            if stage == "train":
                n_upd = self._count.get("%s.update" % stage, 0)
                if n_upd:
                    lines.append("PROFILE   update_mean=%.3fms n=%d" % (
                        self._mean_ms(stage, "sample") + self._mean_ms(stage, "update"),
                        n_upd))
        return lines

    def to_dict(self, env_steps, episodes, wall_seconds):
        data = {
            "env_steps": env_steps,
            "episodes": episodes,
            "wall_seconds": wall_seconds,
            "env_steps_per_s": env_steps / wall_seconds if wall_seconds else 0.0,
            "reset": {
                "n": self.reset_count,
                "mean_ms": self.reset_seconds / self.reset_count * 1000.0
                if self.reset_count else 0.0},
            "stages": {},
        }
        for stage in ("warmup", "train"):
            stage_data = data["stages"][stage] = {
                "blocks": self.blocks(stage),
                "collect_mean_ms_per_block": self.collect_mean_ms(stage),
                "sections": {},
            }
            for s in ALL_SECTIONS:
                key = "%s.%s" % (stage, s)
                stage_data["sections"][s] = {
                    "n": self._count.get(key, 0),
                    "total_seconds": self._total.get(key, 0.0),
                    "mean_ms": self._mean_ms(stage, s),
                }
        return data
