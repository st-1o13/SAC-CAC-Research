"""
Stage B, n=5: re-run the specific run_ids that showed m_dir/grad_dot
anomalies in the original 100-run production results (54, 60, 83 for
m_dir; 3, 7 for grad_dot), with the same diagnostic instrumentation as
Stage A, to try to observe the anomaly mechanism directly.

Does NOT modify proposal_ver1_py/. Does NOT touch runs_proposal_ver1/
(read-only baseline comparison only). Saves to a directory separate from
Stage A: runs_proposal_ver1_diag_stageB/.

Per run_id: same 12-array regression check against the ORIGINAL 100-run
baseline (runs_proposal_ver1/run_{run_id}.npz) as Stage A used, plus the
same per-update shadow self-consistency check. A run that does not
reproduce the original anomaly is NOT treated as a failure (seeds are
deterministic and identical, so if it does not reproduce, that is itself
a fact to report, not an error).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proposal_ver1_py"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np

import diag_hooks

import config    # noqa: E402  (unmodified proposal_ver1_py module)
import agent     # noqa: E402
import train     # noqa: E402

DIAG_SAVE_DIR = "runs_proposal_ver1_diag_stageB"
BASELINE_DIR = "runs_proposal_ver1"

TARGET_RUN_IDS = [54, 60, 83, 3, 7]

# Known anomalies from the original 100-run analysis, for reference in the report
KNOWN_ANOMALIES = {
    54: ("m_dir", 182, 283804.77),
    60: ("m_dir", 246, 1468.87),
    83: ("m_dir", 65, -8230.14),
    3:  ("grad_dot", 235, -1834.90),
    7:  ("grad_dot", 218, -948.62),
}

TARGET_KEYS = [
    "episode_rewards_nav", "episode_rewards_safe", "episode_h", "episode_h_value",
    "episode_m_dir", "episode_violations", "episode_safe_grad_stage2",
    "episode_stability_grad", "episode_grad_dot",
    "episode_theta_dot_mean", "episode_theta_dot_max", "episode_theta_dot_min",
]


def regression_check(run_id):
    baseline_path = f"{BASELINE_DIR}/run_{run_id}.npz"
    diag_path = f"{DIAG_SAVE_DIR}/run_{run_id}.npz"
    d_baseline = np.load(baseline_path)
    d_diag = np.load(diag_path)
    all_match = True
    details = []
    for k in TARGET_KEYS:
        a = d_baseline[k]
        b = d_diag[k]
        eq = bool(np.array_equal(a, b))
        max_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape and a.size else (0.0 if eq else float("nan"))
        details.append((k, eq, max_diff))
        if not eq:
            all_match = False
    return all_match, details


def analyze_run(run_id, episode_violations, final_env_steps):
    """Post-hoc scan of this run's update_log / episode_log for extreme
    values and their relationship to h_value / violation rate."""
    update_csv = f"{DIAG_SAVE_DIR}/run_{run_id}_update_log.csv"
    episode_csv = f"{DIAG_SAVE_DIR}/run_{run_id}_episode_log.csv"

    ep_length = {}
    ep_viol = {}
    ep_rate = {}
    with open(episode_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode"])
            ep_length[ep] = int(row["episode_length"]) if row["episode_length"] else None
            ep_viol[ep] = int(row["episode_violations"]) if row["episode_violations"] else None
            ep_rate[ep] = float(row["violation_rate"]) if row["violation_rate"] not in ("", "None") else None

    max_abs_mdir = (None, None)   # (value, episode)
    max_abs_gd = (None, None)
    min_abs_denom1 = (None, None)
    min_abs_denom2 = (None, None)
    n_rows = 0

    with open(update_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            ep = int(row["episode"]) if row["episode"] else None
            try:
                mdir = float(row["m_dir_real"])
                if max_abs_mdir[0] is None or abs(mdir) > abs(max_abs_mdir[0]):
                    max_abs_mdir = (mdir, ep)
            except (ValueError, TypeError):
                pass
            try:
                gd = float(row["grad_dot_raw"])
                if max_abs_gd[0] is None or abs(gd) > abs(max_abs_gd[0]):
                    max_abs_gd = (gd, ep)
            except (ValueError, TypeError):
                pass
            for key, tracker in (("denom1", "min_abs_denom1"), ("denom2", "min_abs_denom2")):
                val = row.get(key)
                if val not in (None, "", "None"):
                    try:
                        v = float(val)
                        if tracker == "min_abs_denom1":
                            if min_abs_denom1[0] is None or abs(v) < abs(min_abs_denom1[0]):
                                min_abs_denom1 = (v, ep)
                        else:
                            if min_abs_denom2[0] is None or abs(v) < abs(min_abs_denom2[0]):
                                min_abs_denom2 = (v, ep)
                    except ValueError:
                        pass

    clip_counts = {"critic_net": 0, "critic_safe": 0, "actor_net": 0, "unknown": 0}
    clip_triggered = {"critic_net": 0, "critic_safe": 0, "actor_net": 0, "unknown": 0}
    with open(f"{DIAG_SAVE_DIR}/run_{run_id}_clip_log.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = row["clip_role"]
            clip_counts[role] = clip_counts.get(role, 0) + 1
            if row["clip_triggered"] == "True":
                clip_triggered[role] = clip_triggered.get(role, 0) + 1

    return {
        "n_update_rows": n_rows,
        "max_abs_mdir": max_abs_mdir,
        "max_abs_grad_dot": max_abs_gd,
        "min_abs_denom1": min_abs_denom1,
        "min_abs_denom2": min_abs_denom2,
        "clip_counts": clip_counts,
        "clip_triggered": clip_triggered,
        "ep_viol_at_mdir_ep": ep_viol.get(max_abs_mdir[1]),
        "ep_rate_at_mdir_ep": ep_rate.get(max_abs_mdir[1]),
        "ep_viol_at_gd_ep": ep_viol.get(max_abs_gd[1]),
        "ep_rate_at_gd_ep": ep_rate.get(max_abs_gd[1]),
    }


def main():
    print(f">>> Stage B, n={len(TARGET_RUN_IDS)}: installing diagnostic patches (agent.py / train.py unmodified) <<<")
    diag_hooks.install_patches(agent, train)

    summary = {}

    for run_id in TARGET_RUN_IDS:
        print(f"\n{'='*70}\n>>> run_id={run_id} (known anomaly: {KNOWN_ANOMALIES.get(run_id)}) <<<\n{'='*70}")
        diag_hooks.reset_logs()

        try:
            train.run_experiment(run_id, save_dir=DIAG_SAVE_DIR)
            completed = True
            error = None
        except Exception as e:
            completed = False
            error = repr(e)
            print(f"!!! run_id={run_id} raised an exception (recorded, not suppressed as success): {error}")

        if not completed:
            summary[run_id] = {"completed": False, "error": error}
            continue

        episode_violations = list(train.episode_violations)
        final_env_steps = diag_hooks.get_final_env_step_count()

        update_csv_path = f"{DIAG_SAVE_DIR}/run_{run_id}_update_log.csv"
        episode_csv_path = f"{DIAG_SAVE_DIR}/run_{run_id}_episode_log.csv"
        n_update_rows, n_clip_rows = diag_hooks.write_update_log_csv(update_csv_path)
        n_episode_rows = diag_hooks.write_episode_log_csv(episode_csv_path, episode_violations, final_env_steps)
        mismatch_count, total_shadow_calls = diag_hooks.get_mismatch_count()

        all_match, details = regression_check(run_id)

        analysis = analyze_run(run_id, episode_violations, final_env_steps)

        summary[run_id] = {
            "completed": True,
            "error": None,
            "n_update_rows": n_update_rows,
            "n_clip_rows": n_clip_rows,
            "n_episode_rows": n_episode_rows,
            "mismatch_count": mismatch_count,
            "total_shadow_calls": total_shadow_calls,
            "regression_all_match": all_match,
            "regression_details": details,
            "analysis": analysis,
        }

        print(f"[run_id={run_id}] completed. update_rows={n_update_rows} shadow_mismatches={mismatch_count}/{total_shadow_calls} "
              f"regression_match={all_match}")
        if not all_match:
            print(f"!!! run_id={run_id}: REGRESSION MISMATCH DETECTED !!!")
            for k, eq, diff in details:
                if not eq:
                    print(f"    {k}: array_equal=False max|diff|={diff}")

    diag_hooks.uninstall_patches(agent, train)
    print("\n>>> Patches uninstalled <<<")

    print(f"\n\n{'#'*70}\n# STAGE B FINAL SUMMARY\n{'#'*70}")
    for run_id in TARGET_RUN_IDS:
        s = summary.get(run_id, {})
        print(f"\n--- run_id={run_id} ---")
        if not s.get("completed"):
            print(f"  NOT COMPLETED. error={s.get('error')}")
            continue
        print(f"  regression_all_match: {s['regression_all_match']}")
        print(f"  shadow mismatches: {s['mismatch_count']} / {s['total_shadow_calls']}")
        a = s["analysis"]
        print(f"  max|m_dir| = {a['max_abs_mdir'][0]!r} at episode {a['max_abs_mdir'][1]} "
              f"(violations={a['ep_viol_at_mdir_ep']}, rate={a['ep_rate_at_mdir_ep']})")
        print(f"  max|grad_dot_raw| = {a['max_abs_grad_dot'][0]!r} at episode {a['max_abs_grad_dot'][1]} "
              f"(violations={a['ep_viol_at_gd_ep']}, rate={a['ep_rate_at_gd_ep']})")
        print(f"  min|denom1| = {a['min_abs_denom1'][0]!r} at episode {a['min_abs_denom1'][1]}")
        print(f"  min|denom2| = {a['min_abs_denom2'][0]!r} at episode {a['min_abs_denom2'][1]}")
        print(f"  clip counts: {a['clip_counts']}")
        print(f"  clip triggered: {a['clip_triggered']}")

    n_completed = sum(1 for s in summary.values() if s.get("completed"))
    n_match = sum(1 for s in summary.values() if s.get("completed") and s.get("regression_all_match"))
    n_mismatch_total = sum(s.get("mismatch_count", 0) for s in summary.values() if s.get("completed"))
    print(f"\n>>> {n_completed}/{len(TARGET_RUN_IDS)} runs completed. "
          f"{n_match}/{n_completed} regression-matched. total shadow mismatches: {n_mismatch_total} <<<")


if __name__ == "__main__":
    main()
