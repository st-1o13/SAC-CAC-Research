# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

# save_single_run(), save_figures_for_run() and run_experiment() are kept
# in this single file because run_experiment() declares module-level
# `global episode_rewards_nav, ...` variables that save_single_run() and
# save_figures_for_run() read via bare names (not function arguments).
# Splitting them into separate files would break that sharing under
# Python's per-module `global` semantics. See the design note reported
# alongside this migration.

import os
import time
import random
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from config import args, BASE_SEED, total_episodes, device
from agent import SoftActorCriticModel
from replay_memory import ReplayMemory


def save_single_run(run_id, save_dir="runs"):
    os.makedirs(save_dir, exist_ok=True)

    data = {
        "episode_rewards_nav": np.array(episode_rewards_nav),
        "episode_rewards_safe": np.array(episode_rewards_safe),
        "episode_h": np.array(episode_h),
        "episode_h_value": np.array(episode_h_value),
        "episode_m_dir": np.array(episode_m_dir),
        "episode_violations": np.array(episode_violations),
        "episode_safe_grad_stage2": np.array(episode_safe_grad_stage2),
        "episode_stability_grad": np.array(episode_stability_grad),
        "episode_grad_dot": np.array(episode_grad_dot),
        "episode_theta_dot_mean": np.array(episode_theta_dot_mean),
        "episode_theta_dot_max": np.array(episode_theta_dot_max),
        "episode_theta_dot_min": np.array(episode_theta_dot_min),
        "runtime_sec": time.time() - start_time,
    }

    np.savez_compressed(f"{save_dir}/run_{run_id}.npz", **data)
    print(f"[SAVE] Saved run data to {save_dir}/run_{run_id}.npz")


# ============================================================
# 2. 図を保存する関数
# ============================================================

def save_figures_for_run(run_id, save_dir="runs"):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 安全報酬
    plt.figure(figsize=(10,4))
    plt.plot(episode_rewards_safe)
    plt.ylim(-200, 0)
    plt.title(f"Safe Reward (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_safe_reward.png")
    plt.close()

    # 2. 性能報酬
    plt.figure(figsize=(10,4))
    plt.plot(episode_rewards_nav)
    plt.ylim(-2000, 0)
    plt.title(f"Navigation Reward (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_nav_reward.png")
    plt.close()


    # 3. m_dir
    plt.figure(figsize=(10,4))
    plt.plot(episode_m_dir)
    plt.title(f"m_dir (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_m_dir.png")
    plt.close()

    # 4. h_value
    plt.figure(figsize=(10,4))
    plt.plot(episode_h_value)
    plt.title(f"h_value (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_h_value.png")
    plt.close()

    # 5. h(s)
    plt.figure(figsize=(10,4))
    plt.plot(episode_h)
    plt.title(f"h(s) (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_h_s.png")
    plt.close()

    # 6. 安全違反
    plt.figure(figsize=(10,4))
    plt.plot(episode_violations)
    plt.title(f"Violations (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_violations.png")
    plt.close()

    # 7. grad_dot
    plt.figure(figsize=(10,4))
    plt.plot(episode_grad_dot)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"grad_st · grad_sa (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_grad_dot.png")
    plt.close()

    # 8. stability_grad
    plt.figure(figsize=(10,4))
    plt.plot(episode_stability_grad)
    plt.title(f"Stability Gradient Norm (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_stability_grad.png")
    plt.close()

    # 9. safe_grad
    plt.figure(figsize=(10,4))
    plt.plot(episode_safe_grad_stage2)
    plt.title(f"Safe Gradient Norm (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_safe_grad.png")
    plt.close()

    # 10. theta_dot
    plt.figure(figsize=(10,4))
    plt.plot(episode_theta_dot_mean, label="mean")
    plt.plot(episode_theta_dot_max, label="max")
    plt.plot(episode_theta_dot_min, label="min")
    plt.legend()
    plt.title(f"Angular Velocity Stats (Run {run_id})")
    plt.grid()
    plt.savefig(f"{save_dir}/run_{run_id}_theta_dot.png")
    plt.close()

    print(f"[SAVE] Saved ALL figures for run {run_id}")
    

def run_experiment(run_id):

    print(f"\n===== RUN {run_id} START =====")
    # -----------------------------
    # 1. すべての episode_xxx を初期化
    # -----------------------------
    global episode_rewards_nav, episode_rewards_safe
    global episode_h, episode_h_value, episode_m_dir
    global episode_violations, episode_safe_grad_stage2
    global episode_stability_grad, episode_grad_dot
    global episode_theta_dot_mean, episode_theta_dot_max, episode_theta_dot_min
    global n_steps, n_update, printed_stage1, start_time

    episode_rewards_nav = []
    episode_rewards_safe = []
    episode_h = []
    episode_h_value = []
    episode_m_dir = []
    episode_violations = []
    episode_safe_grad_stage2 = []
    episode_stability_grad = []
    episode_grad_dot = []
    episode_theta_dot_mean = []
    episode_theta_dot_max = []
    episode_theta_dot_min = []


    # -----------------------------
    # 2. 環境・エージェント・メモリを毎回作り直す（重要）
    # -----------------------------
    run_seed = BASE_SEED + run_id
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    env = gym.make(args['gym_name'])
    env.action_space.seed(run_seed)

    agent = SoftActorCriticModel(
        state_num=env.observation_space.shape[0],
        action_num=env.action_space.shape[0],
        action_scale=env.action_space.high[0],
        args=args,
        device=device
    )

    for opt in (agent.actor_optim, agent.critic_optim, agent.critic_safe_optim, agent.alpha_optim):#下３行追加したよー
        for g in opt.param_groups:
            g['lr'] = g.get('lr', 3e-4) * 0.3#上３行はつかう 1e-3

    """for g in agent.alpha_optim.param_groups:
           g['lr'] *= 0.5 """

    # Inserted: lower critic_safe lr to base_lr * 0.125 (absolute overwrite)
    base_lr = 2e-4
    for g in agent.critic_safe_optim.param_groups:
        g['lr'] = base_lr * 0.75#0.50 #0.25 #0.125

    memory = ReplayMemory(args['memory_size'])

    # -----------------------------
    # 3. カウンタ初期化
    # -----------------------------
    n_steps = 0
    n_update = 0
    printed_stage1 = False
    start_time = time.time()


    # ★★★ ここにあなたの「学習ループ（for ep in ...）」を丸ごと入れる ★★★
    
    for ep in range(1, total_episodes + 1):
    # === α を小 → 大にスケジューリング ===
        alpha_start = 0.1
        alpha_end   = 0.9
        total_eps   = total_episodes

        new_alpha = alpha_start + (alpha_end - alpha_start) * (ep / total_eps)
        args['alpha'] = new_alpha
        agent.alpha = torch.tensor(new_alpha).to(device)

        if ep % 5 == 0:
            print(f"[Alpha Schedule] ep={ep}, alpha={new_alpha:.4f}")

        # -----------------------------
        # エピソード初期化
        # -----------------------------

        ep_ret_safe = 0.0
        ep_ret_nav = 0.0

        perupdate_st_norms = []
        perupdate_sa_norms = []
        perupdate_dots = []
        perupdate_safe_grad_norms = []


        perupdate_ratio_m_list = [] 
        loss_nav_list = []
        loss_safe_list = []

        violations    = 0
        done = False
        # episode 1でrun_seedを指定して環境の乱数Generatorを初期化し、
        # episode 2以降はそのGeneratorの状態を継続して使用するため、
        # 毎episodeでseedを再設定しない。
        if ep == 1:
            state, _ = env.reset(seed=run_seed)
        else:
            state, _ = env.reset()

        # [ADDED START] per-episode theta_dot collection
        theta_dot_vals = []
        theta_dot_signed = []       # stores signed theta_dot for this episode
        # [ADDED END]

        perupdate_m_dir = []
        perstep_h = []   # ← h(x) を保存するバッフ

        while not done:

            if args['start_steps'] > n_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            # === h(x) を保存 ===
            s_t = torch.from_numpy(state).float().to(device).unsqueeze(0)
            h_val = agent.h(s_t).item()
            perstep_h.append(h_val)


            if len(memory) > args['batch_size']:
                for _ in range(args['updates_per_step']):
                    batch = memory.sample(args['batch_size'])

                    res = agent.update_critics_and_actor(batch, episode_index=ep)


                    if res is None:
                        loss_nav, loss_safe, safe_grad_norm, st_norm_raw, sa_norm_raw, dot_raw = (None, None, None, None, None, None)
                    else:
                        loss_nav, loss_safe, safe_grad_norm, st_norm_raw, sa_norm_raw, dot_raw = res
                
                    if hasattr(agent, "last_m_dir"):
                        perupdate_m_dir.append(agent.last_m_dir)



                    # original loss list append
                    if loss_safe is not None:########
                        loss_safe_list.append(loss_safe)##########
                    if loss_nav is not None:
                        loss_nav_list.append(loss_nav)

                    # append per-update metrics into the episode-local buffers
                    if safe_grad_norm is not None:#safe_grad_normから変更追加した
                        perupdate_safe_grad_norms.append(float(safe_grad_norm))#ここも同じ変更
                    if st_norm_raw is not None:
                        perupdate_st_norms.append(float(st_norm_raw))
                    if sa_norm_raw is not None:
                        perupdate_sa_norms.append(float(sa_norm_raw))
                    if dot_raw is not None:
                        perupdate_dots.append(float(dot_raw))
                        
                    n_update += 1

            # 環境ステップ
            next_state, r_env, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state_t      = torch.from_numpy(state).float().to(device)
            next_state_t = torch.from_numpy(next_state).float().to(device)
            

            #安全報酬
            action_t = torch.from_numpy(action).float().to(device).unsqueeze(0)
            r_safe = agent.compute_safety_reward(
                 state_t.unsqueeze(0),
                 action_t,
                 next_state_t.unsqueeze(0),
                 args['alpha']
             ).item()
             # --- Barrier 関数違反カウント ---
            # h(s_next)<0 を「違反」としてカウント
            if agent.h(next_state_t.unsqueeze(0)).item() < 0.0:
                violations += 1

             # 両者をタプルで保存
            memory.push(state=state, action=action,reward=(r_env, r_safe), next_state=next_state,mask = float(not done))

            # [ADDED START] collect theta_dot (Pendulum obs: [cos, sin, theta_dot])
            theta_dot = float(next_state[2])
            theta_dot_signed.append(theta_dot)           # record signed value
            theta_dot_vals.append(abs(theta_dot))  # store absolute angular velocity
            # [ADDED END]


            #else:
            #  done=False
            n_steps += 1
            ep_ret_nav += r_env
            ep_ret_safe+= r_safe

            state = next_state
        # -----------------------------
        # エピソード終了処理
        # -----------------------------
        episode_rewards_nav.append(ep_ret_nav)
        episode_rewards_safe.append(ep_ret_safe)
        episode_violations.append(violations)

        episode_h.append(float(np.mean(perstep_h)) if perstep_h else 0.0)
        episode_h_value.append(float(np.mean(agent.h_buffer)) if hasattr(agent,"h_buffer") and agent.h_buffer else 0.0)
        episode_m_dir.append(float(np.mean(perupdate_m_dir)) if perupdate_m_dir else 0.0)
        


        if theta_dot_vals:
            episode_theta_dot_mean.append(float(np.mean(theta_dot_vals)))
            episode_theta_dot_max.append(float(np.max(theta_dot_vals)))
            episode_theta_dot_min.append(float(np.min(theta_dot_vals)))
        else:
            episode_theta_dot_mean.append(0.0)
            episode_theta_dot_max.append(0.0)
            episode_theta_dot_min.append(0.0)

        # 勾配統計
        if perupdate_safe_grad_norms:
            episode_safe_grad_stage2.append(float(np.mean(perupdate_safe_grad_norms)))
        else:
            episode_safe_grad_stage2.append(0.0)

        if perupdate_st_norms:
            episode_stability_grad.append(float(np.mean(perupdate_st_norms)))
        else:
            episode_stability_grad.append(0.0)

        if perupdate_dots:
            episode_grad_dot.append(float(np.mean(perupdate_dots)))
        else:
            episode_grad_dot.append(0.0)
            
        
    # つまり、今あなたが貼った巨大な for ep in ... の部分を
    # そのまま run_experiment() の中に移動させる

    # --- 3. 1回分のデータ保存 ---
    save_single_run(run_id)
    save_figures_for_run(run_id)

    print(f"===== RUN {run_id} END =====\n")
