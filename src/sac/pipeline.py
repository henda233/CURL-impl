"""像素观测管线：center crop + 3 帧栈 uint8。

SAC 训练不做 random crop（与 AC 基线一致，增强留待 CURL 阶段）；帧以 uint8 保存
供 replay buffer 存储，采样时由 obs_to_tensor 归一化。
"""

import numpy as np
import torch

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


def obs_to_tensor(obs, device):
    """(...,9,84,84) uint8 帧栈 → device 上的 float32 张量（归一化在 device 执行）。"""
    return torch.from_numpy(np.asarray(obs)).to(device=device).float().div_(255.0)
