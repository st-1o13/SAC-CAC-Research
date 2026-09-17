"""
Stage B-extended, A1 (n=1): validate the EXTENDED diagnostic hooks
(post-m_dir pipeline: m0/m_soft/rel_cap/m_scale, chain-level e allclose
check, and post-clip gradient norm) on run_id=1, before reusing them on
the Stage B anomaly run_ids (54, 60, 83, 3, 7).

Does NOT modify proposal_ver1_py/. Does NOT touch runs_proposal_ver1/,
runs_proposal_ver1_diag/, or runs_proposal_ver1_diag_stageB/ (all
pre-existing diagnostic results are left untouched). Saves to a NEW
directory: runs_proposal_ver1_diag_stageB_ext_n1/.

The actor_optim.step() instance-level patch (parameter-delta observation)
is explicitly OUT OF SCOPE for this script, per instructions -- it is a
separate, later stage.

Two independent pass/fail gates, both required before the extended logs
can be trusted:
  1. The 12 saved arrays in this run's own run_1.npz must exactly match
     the pre-existing baseline runs_proposal_ver1/run_1.npz (proves the
     extended patches did not alter training).
  2. The shadow-computed final vector (shadow_e) must match the real
     restricted_direction()'s returned e via torch.allclose(), for every
     single update call in this run (proves the extended shadow chain is
     not silently wrong on real training data, not just synthetic tests).
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
DIAG_SAVE_DIR = "runs_proposal_ver1_diag_stageB_ext_n1"

TARGET_KEYS = [
    "episode_rewards_nav", "episode_rewards_safe", "episode_h", "episode_h_value",
    "episode_m_dir", "episode_violations", "episode_safe_grad_stage2",
    "episode_stability_grad", "episode_grad_dot",
    "episode_theta_dot_mean", "episode_theta_dot_max", "episode_theta_dot_min",
]


def main():
    print(">>> Stage B-extended A1, n=1: installing EXTENDED diagnostic patches "
          "(agent.py / train.py unmodified) <<<")
    diag_hooks.install_patches(agent, train)

    print(f">>> Running run_experiment(1, save_dir='{DIAG_SAVE_DIR}') <<<")
    train.run_experiment(1, save_dir=DIAG_SAVE_DIR)

    episode_violations = list(train.episode_violations)
    final_env_steps = diag_hooks.get_final_env_step_count()

    diag_hooks.uninstall_patches(agent, train)
    print(">>> Patches uninstalled <<<")

    update_csv_path = f"{DIAG_SAVE_DIR}/run_1_update_log.csv"
    episode_csv_path = f"{DIAG_SAVE_DIR}/run_1_episode_log.csv"
    n_update_rows, n_clip_rows = diag_hooks.write_update_log_csv(update_csv_path)
    n_episode_rows = diag_hooks.write_episode_log_csv(episode_csv_path, episode_violations, final_env_steps)

    mismatch_count, total_shadow_calls = diag_hooks.get_mismatch_count()
    chain_mismatch_count, _ = diag_hooks.get_chain_mismatch_count()

    print(f"\n=== Diagnostic log summary ===")
    print(f"update_log rows: {n_update_rows} (written to {update_csv_path})")
    print(f"clip_log rows:   {n_clip_rows} (written to {DIAG_SAVE_DIR}/run_1_clip_log.csv)")
    print(f"episode_log rows: {n_episode_rows} (written to {episode_csv_path})")
    print(f"m_dir scalar shadow self-consistency: {mismatch_count} mismatches / {total_shadow_calls} calls")
    print(f"FULL-CHAIN (shadow_e vs real e, torch.allclose) self-consistency: "
          f"{chain_mismatch_count} mismatches / {total_shadow_calls} calls")

    # ------------------------------------------------------------------
    # Gate 1: diagnostic-run npz must equal baseline
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
        max_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape and a.size else (0.0 if eq else float("nan"))
        status = "OK" if eq else "MISMATCH"
        print(f"  {k:28s} array_equal={eq!s:5s} max|diff|={max_diff:<12.6g} [{status}]")
        if not eq:
            all_match = False

    print()
    if not all_match:
        print(">>> STOP: at least one array MISMATCHES the baseline. <<<")
        print(">>> Extended patches may have altered training behavior. Do NOT trust these logs. <<<")
        sys.exit(1)

    print(">>> Gate 1 PASSED: all 12 arrays match baseline exactly. Extended patches did NOT alter training. <<<")

    # ------------------------------------------------------------------
    # Gate 2: full-chain shadow self-consistency must be perfect
    # ------------------------------------------------------------------
    if chain_mismatch_count > 0 or mismatch_count > 0:
        print(f"\n>>> STOP: shadow self-consistency check FAILED "
              f"(m_dir mismatches={mismatch_count}, chain mismatches={chain_mismatch_count}). <<<")
        print(">>> The extended shadow pipeline (m0/m_soft/rel_cap/m_scale) may have a transcription "
              "error. Do NOT trust m0/m_soft/rel_cap/m_scale columns until this is resolved. <<<")
        sys.exit(1)

    print(">>> Gate 2 PASSED: shadow_e matches real e (torch.allclose) for every update call in this run. <<<")
    print("\n>>> ALL GATES PASSED. Extended diagnostic logs can be trusted. "
          "Safe to proceed to Stage B-extended re-runs of run_id 54, 60, 83, 3, 7. <<<")


if __name__ == "__main__":
    main()
