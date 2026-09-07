"""像素观测预处理：center crop + 3 帧栈 uint8。

归一化延迟至张量进网络时（搬运到目标 device 后）执行，帧以 uint8 保存，
避免 CPU 侧 float32 归一化造成 4 倍搬运与存储冗余。与 SAC 管线一致。
"""

import numpy as np

RENDER_H = RENDER_W = 100
CROP = 84
OFFSET = (RENDER_H - CROP) // 2
FRAMES = 3
STACK_C = FRAMES * 3


def center_crop(frame_hwc_uint8):
    return frame_hwc_uint8[OFFSET:OFFSET + CROP, OFFSET:OFFSET + CROP]


class FrameStack:
    """维护最近 FRAMES 帧 center-crop 后的 uint8 帧（HWC），state() 拼成 (9,84,84) uint8。"""

    def __init__(self):
        self._frames = []

    def reset(self, frame_hwc_uint8):
        first = center_crop(frame_hwc_uint8)
        self._frames = [first.copy() for _ in range(FRAMES)]

    def push(self, frame_hwc_uint8):
        self._frames.pop(0)
        self._frames.append(center_crop(frame_hwc_uint8))

    def state(self):
        return np.concatenate(self._frames, axis=2).transpose(2, 0, 1)
