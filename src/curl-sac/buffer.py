"""off-policy replay buffer（环形覆盖，像素帧以 100×100 uint8 存储省内存）。

一条 transition 存：obs 帧栈 (9,100,100) uint8、action float32、reward、
next 新帧 (3,100,100) uint8。next 观测在 sample 时由 obs[3:] 与新帧拼接得到
（帧栈丢最旧帧、压入新帧）。100 原帧保留供训练期 random crop（100→84）。
"""

import numpy as np

STACK_C = 9
FRAME_C = 3
RENDER = 100


class ReplayBuffer:
    def __init__(self, capacity, act_dim=6):
        self.capacity = int(capacity)
        self.obs = np.empty((self.capacity, STACK_C, RENDER, RENDER), dtype=np.uint8)
        self.next_f = np.empty((self.capacity, FRAME_C, RENDER, RENDER), dtype=np.uint8)
        self.act = np.empty((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.empty((self.capacity,), dtype=np.float32)
        self.done = np.empty((self.capacity,), dtype=np.bool_)
        self._pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_frame, done):
        i = self._pos
        self.obs[i] = obs
        self.next_f[i] = next_frame
        self.act[i] = action
        self.rew[i] = reward
        self.done[i] = done
        self._pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        obs = self.obs[idx]
        act = self.act[idx].copy()
        rew = self.rew[idx].copy()
        done = self.done[idx].copy()
        next_obs = np.concatenate([obs[:, FRAME_C:], self.next_f[idx]], axis=1)
        return obs, act, rew, next_obs, done

    def __len__(self):
        return self.size
