"""
Stage A, n=1: run Proposal Ver.1's run_experiment(1) with diagnostic
instrumentation (Methods A/B/C from diag_hooks.py / diag_shadow.py)
installed, WITHOUT modifying any file under proposal_ver1_py/.

Saves to runs_proposal_ver1_diag/ only. Never touches runs_proposal_ver1/
or runs/. After running, verifies the diagnostic run's own run_1.npz
(the same 12 arrays saved by the unmodified save_single_run/save_figures_for_run)
is bit-identical to the existing baseline runs_proposal_ver1/run_1.npz.
If any array differs, this script reports it as a STOP condition and does
NOT claim the diagnostic logs are trustworthy.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proposal_ver1_py"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import diag_hooks

import config    # noqa: E402  (unmodified proposal_ver1_py module)
import agent     # noqa: E402
import train     # noqa: E402

BASELINE_PATH = "runs_proposal_ver1/run_1.npz"
DIAG_SAVE_DIR = "runs_proposal_ver1_diag"

TARGET_KEYS = [
    "episode_rewards_nav", "episode_rewards_safe", "episode_h", "episode_h_value",
    "episode_m_dir", "episode_violations", "episode_safe_grad_stage2",
    "episode_stability_grad", "episode_grad_dot",
    "episode_theta_dot_mean", "episode_theta_dot_max", "episode_theta_dot_min",
]


def main():
    print(">>> Stage A, n=1: installing diagnostic patches (agent.py / train.py unmodified) <<<")
    diag_hooks.install_patches(agent, train)

    print(">>> Running run_experiment(1, save_dir='runs_proposal_ver1_diag') <<<")
    train.run_experiment(1, save_dir=DIAG_SAVE_DIR)

    episode_violations = list(train.episode_violations)  # module-global, set by run_experiment
    final_env_steps = diag_hooks.get_final_env_step_count()

    diag_hooks.uninstall_patches(agent, train)
    print(">>> Patches uninstalled <<<")

    update_csv_path = f"{DIAG_SAVE_DIR}/run_1_update_log.csv"
    episode_csv_path = f"{DIAG_SAVE_DIR}/run_1_episode_log.csv"
    n_update_rows, n_clip_rows = diag_hooks.write_update_log_csv(update_csv_path)
    n_episode_rows = diag_hooks.write_episode_log_csv(episode_csv_path, episode_violations, final_env_steps)

    mismatch_count, total_shadow_calls = diag_hooks.get_mismatch_count()

    print(f"\n=== Diagnostic log summary ===")
    print(f"update_log rows: {n_update_rows} (written to {update_csv_path})")
    print(f"clip_log rows:   {n_clip_rows} (written to {DIAG_SAVE_DIR}/run_1_clip_log.csv)")
    print(f"episode_log rows: {n_episode_rows} (written to {episode_csv_path})")
    print(f"shadow self-consistency: {mismatch_count} mismatches / {total_shadow_calls} calls")

    # ------------------------------------------------------------------
    # Critical regression check: diagnostic-run npz must equal baseline
    # ------------------------------------------------------------------
    diag_npz_path = f"{DIAG_SAVE_DIR}/run_1.npz"
    print(f"\n=== Regression check: {diag_npz_path} vs {BASELINE_PATH} ===")

    d_baseline = np.load(BASELINE_PATH)
    d_diag = np.load(diag_npz_path)

    all_match = True
    for k in TARGET_KEYS:
        a = d_baseline[k]
        b = d_diag[k]
        eq = bool(np.array_equal(a, b))
        if a.shape == b.shape:
            max_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        else:
            max_diff = float("nan")
        status = "OK" if eq else "MISMATCH"
        print(f"  {k:28s} array_equal={eq!s:5s} max|diff|={max_diff:<12.6g} [{status}]")
        if not eq:
            all_match = False

    print()
    if all_match:
        print(">>> RESULT: all 12 arrays match baseline exactly. Diagnostic patches did NOT alter training. <<<")
        print(">>> Diagnostic logs (update/episode CSVs) can be trusted as observing the real training run. <<<")
    else:
        print(">>> STOP: at least one array MISMATCHES the baseline. <<<")
        print(">>> Diagnostic patches may have altered training behavior. Do NOT proceed to Stage B/C. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
