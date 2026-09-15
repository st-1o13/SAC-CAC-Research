# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

# load_all_runs(), the stats print, the mean/std plot, the scatter plot
# and the runtime stats/histogram are kept in this single file because,
# in the notebook, they share module-level globals (nav_all, safe_all,
# h_all, hval_all, mdir_all, dot_all, viol_all, runtimes, n_runs) across
# what were separate cells, via bare-name references rather than
# function arguments.

import matplotlib
# NOTE: in the notebook, the Agg (non-interactive) backend is set once in the
# args/run_experiment cell (config.py in this migration) and stays in effect
# for the rest of the kernel session, so this analysis section never opens a
# GUI window even though it calls plt.show(). Since this file is meant to be
# runnable on its own (e.g. after training has already finished, possibly in
# a separate process), the backend must be set here too -- otherwise plt.show()
# can block waiting on an interactive backend with no display attached.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import glob

def load_all_runs(folder="runs"):
    files = sorted(glob.glob(f"{folder}/run_*.npz"))
    nav_list = []
    safe_list = []
    h_list = []
    hval_list = []
    mdir_list = []
    dot_list = []
    viol_list = []
    runtime_list = []

    for f in files:
        data = np.load(f)
        nav_list.append(data["episode_rewards_nav"])
        safe_list.append(data["episode_rewards_safe"])
        h_list.append(data["episode_h"])
        hval_list.append(data["episode_h_value"])
        mdir_list.append(data["episode_m_dir"])
        dot_list.append(data["episode_grad_dot"])
        viol_list.append(data["episode_violations"])
        runtime_list.append(float(data["runtime_sec"]))

    return (np.array(nav_list), np.array(safe_list),
            np.array(h_list), np.array(hval_list),
            np.array(mdir_list), np.array(dot_list),
            np.array(viol_list), np.array(runtime_list),
            )


(nav_all, safe_all, h_all, hval_all, mdir_all, dot_all, viol_all, runtimes) = load_all_runs()

n_runs = nav_all.shape[0]  # 実際にロードされたrun数（ハードコードせず動的に取得）

nav_mean = nav_all.mean(axis=0)
nav_std  = nav_all.std(axis=0)

safe_mean = safe_all.mean(axis=0)
safe_std  = safe_all.std(axis=0)

# --- 性能報酬 ---
plt.figure(figsize=(10,4))
plt.plot(nav_mean, label="mean nav reward")
plt.fill_between(range(len(nav_mean)),
                 nav_mean - nav_std,
                 nav_mean + nav_std,
                 alpha=0.3)
plt.ylim(-2000, 0)
plt.title(f"Navigation Reward (mean ± std over {n_runs} runs)")
plt.grid()
plt.legend()
plt.savefig("nav_mean_std.png", dpi=200)
plt.show()

# --- 安全報酬 ---
plt.figure(figsize=(10,4))
plt.plot(safe_mean, label="mean safe reward")
plt.fill_between(range(len(safe_mean)),
                 safe_mean - safe_std,
                 safe_mean + safe_std,
                 alpha=0.3)
plt.ylim(-200, 0)
plt.title(f"Safety Reward (mean ± std over {n_runs} runs)")
plt.grid()
plt.legend()
plt.savefig("safe_mean_std.png", dpi=200)
plt.show()


print(f"=== Safety vs Performance Statistics ({nav_all.shape[0]} runs) ===")

print(f"Nav Reward: mean={nav_all.mean():.2f}, std={nav_all.std():.2f}")
print(f"Safe Reward: mean={safe_all.mean():.2f}, std={safe_all.std():.2f}")

print(f"h(x): mean={h_all.mean():.4f}, std={h_all.std():.4f}")
print(f"h_value: mean={hval_all.mean():.4f}, std={hval_all.std():.4f}")

print(f"m_dir: mean={mdir_all.mean():.4f}, std={mdir_all.std():.4f}")
print(f"grad_dot: mean={dot_all.mean():.4f}, std={dot_all.std():.4f}")

print(f"Violations: mean={viol_all.mean():.2f}, std={viol_all.std():.2f}")

print(f"Runtime: mean={runtimes.mean():.2f} sec, std={runtimes.std():.2f} sec")


plt.figure(figsize=(6,6))
plt.scatter(hval_all.flatten(), mdir_all.flatten(), alpha=0.2, s=5)
plt.xlabel("h_value (Safety Level)")
plt.ylabel("m_dir (Safety vs Performance Mix)")
plt.title(f"m_dir vs h_value ({hval_all.shape[0]} runs × episodes)")
plt.grid()
plt.savefig("scatter_mdir_hvalue.png", dpi=200)   # ★ これを追加
plt.close() 


print(f"{len(runtimes)}回分の実行時間（秒）:")
print(runtimes)

print(f"\n平均実行時間: {runtimes.mean():.2f} 秒")
print(f"最短実行時間: {runtimes.min():.2f} 秒")
print(f"最長実行時間: {runtimes.max():.2f} 秒")
print(f"標準偏差: {runtimes.std():.2f} 秒")

plt.figure(figsize=(8,4))
plt.hist(runtimes, bins=20, color='skyblue', edgecolor='black')
plt.title(f"Runtime Distribution ({len(runtimes)} runs)")
plt.xlabel("seconds")
plt.ylabel("count")
plt.grid()
plt.savefig("runtime_hist.png", dpi=200)
plt.close()
