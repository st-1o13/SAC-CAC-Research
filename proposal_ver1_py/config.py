# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import torch
import numpy as np


# --- from the 'gym_name / device / seed' setup cell ---
gym_name = 'Pendulum-v1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


# --- from the top of the args/run_experiment cell (matplotlib setup,
#     args dict, BASE_SEED, total_episodes) ---
import os
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42


args = {
    'gym_name' : 'Pendulum-v1',
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.9,
    'seed': 123456,
    'batch_size': 256,
    'hidden_size': 256,
    'start_steps': 1000,
    'updates_per_step': 1,
    'target_update_interval': 1,
    'memory_size': 100000,
    'epochs': 100,
    'eval_interval': 10,
    'stage1_episodes': 0,
    'stage2_episodes': 250,
    'log_interval':  10,
}

BASE_SEED = 123456  # run_seed = BASE_SEED + run_id (per-run reproducible seeding)

total_episodes = args['stage1_episodes'] + args['stage2_episodes']

# ============================================================
# 1. 1回分のデータ保存関数
# ============================================================
