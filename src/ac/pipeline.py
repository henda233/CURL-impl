"""像素观测预处理：center crop + 帧栈 + 归一化。

AC 基线不做 random crop（决策见 AC 实现复盘），训练与评估统一 center crop。
"""

import numpy as np

RENDER_H = RENDER_W = 100
CROP = 84
OFFSET = (RENDER_H - CROP) // 2
FRAMES = 3
STACK_C = FRAMES * 3


def center_crop(frame_hwc_uint8):
    return frame_hwc_uint8[OFFSET:OFFSET + CROP, OFFSET:OFFSET + CROP]


def _to_chw_float(crop):
    return crop.astype(np.float32).transpose(2, 0, 1) * (1.0 / 255.0)


class FrameStack:
    """维护最近 FRAMES 帧的 (9,84,84) float32 观测（旧帧在前）。"""

    def __init__(self):
        self._frames = []

    def reset(self, frame_hwc_uint8):
        first = _to_chw_float(center_crop(frame_hwc_uint8))
        self._frames = [first.copy() for _ in range(FRAMES)]

    def push(self, frame_hwc_uint8):
        self._frames.pop(0)
        self._frames.append(_to_chw_float(center_crop(frame_hwc_uint8)))

    def state(self):
        return np.concatenate(self._frames, axis=0)
