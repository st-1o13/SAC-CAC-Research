# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import torch
import random
import numpy as np


class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        state      = torch.from_numpy(state).float()
        action     = torch.from_numpy(action).float()
        reward     = torch.from_numpy(reward).float()
        next_state = torch.from_numpy(next_state).float()
        done       = torch.from_numpy(done).float()
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        # ここを追加
        self.buffer.clear()
        self.position = 0