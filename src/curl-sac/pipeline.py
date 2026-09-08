"""像素观测管线：100×100 原帧 3 帧栈 uint8 + random/center crop。

CURL 训练期 random crop（100→84，同一窗口施加整栈），评估期 center crop 84。
帧以 uint8 保存供 replay buffer 存储，采样/决策时先 crop 再由 obs_to_tensor 归一化。
"""

import numpy as np
import torch

RENDER_H = RENDER_W = 100
CROP = 84
OFFSET = (RENDER_H - CROP) // 2
FRAMES = 3
STACK_C = FRAMES * 3


def center_crop(x):
    """末两维 100×100 中心裁到 84×84；输入可为 (C,100,100) 或帧 (100,100,3) HWC。"""
    return x[..., OFFSET:OFFSET + CROP, OFFSET:OFFSET + CROP]


def random_crop(x):
    """末两维随机裁到 84×84。x 形状为 (C,100,100)（单窗口）或 (B,C,100,100)（每样本一窗口）。"""
    size = CROP
    if x.ndim == 3:
        c, h, w = x.shape
        hs = np.random.randint(0, h - size + 1, size=1)
        ws = np.random.randint(0, w - size + 1, size=1)
        out = x[:, hs[0]:hs[0] + size, ws[0]:ws[0] + size]
    elif x.ndim == 4:
        b, c, h, w = x.shape
        hs = np.random.randint(0, h - size + 1, size=b)
        ws = np.random.randint(0, w - size + 1, size=b)
        out = np.empty((b, c, size, size), dtype=x.dtype)
        for i in range(b):
            out[i] = x[i, :, hs[i]:hs[i] + size, ws[i]:ws[i] + size]
    else:
        raise ValueError("random_crop expects (C,H,W) or (B,C,H,W), got %s" % (x.shape,))
    return out


class FrameStack:
    """维护最近 FRAMES 帧 100×100 原帧（HWC uint8），state() 拼成 (9,100,100) uint8。"""

    def __init__(self):
        self._frames = []

    def reset(self, frame_hwc_uint8):
        self._frames = [frame_hwc_uint8.copy() for _ in range(FRAMES)]

    def push(self, frame_hwc_uint8):
        self._frames.pop(0)
        self._frames.append(frame_hwc_uint8)

    def state(self):
        return np.concatenate(self._frames, axis=2).transpose(2, 0, 1)


def obs_to_tensor(obs, device):
    """(...,C,84,84) 或 (...,C,100,100) uint8 → device float32（归一化在 device 执行）。"""
    return torch.from_numpy(np.asarray(obs)).to(device=device).float().div_(255.0)
