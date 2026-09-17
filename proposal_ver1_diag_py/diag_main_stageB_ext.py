"""
Stage B-extended, n=5: re-run the same anomaly run_ids as Stage B
(54, 60, 83, 3, 7) with the EXTENDED diagnostic hooks (validated on n=1 by
diag_main_stageB_ext_n1.py, which passed both gates with 0 mismatches),
to trace how the m1/m2/m_dir divergence propagates through
restricted_direction()'s downstream pipeline (m0, m_soft, rel_cap,
m_scale) to the final vector e, and into the actor's clipped gradient.

Does NOT modify proposal_ver1_py/. Does NOT touch runs_proposal_ver1/,
runs_proposal_ver1_diag/, runs_proposal_ver1_diag_stageB/, or
runs_proposal_ver1_diag_stageB_ext_n1/ (all pre-existing results are left
untouched). Saves to a NEW directory: runs_proposal_ver1_diag_stageB_ext/.

Per run_id, two independent gates (same as Stage A / Stage B-extended A1):
  1. 12-array regression check against runs_proposal_ver1/run_{run_id}.npz
  2. shadow_e vs real e torch.allclose check, for every update call

If EITHER gate fails for a given run_id, that run's diagnostic data is
flagged as untrustworthy and NOT used in the anomaly analysis below (its
raw files are still kept on disk for inspection), but the script continues
to the remaining run_ids rather than aborting the whole batch, since each
run_id is an independent, already-completed training run by the time the
checks are made (a mid-training abort is not meaningful post-hoc).
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

DIAG_SAVE_DIR = "runs_proposal_ver1_diag_stageB_ext"
BASELINE_DIR = "runs_proposal_ver1"

TARGET_RUN_IDS = [54, 60, 83, 3, 7]

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


def find_extreme_row(update_csv, key, top_n=1):
    with open(update_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def safe_abs(r):
        v = r.get(key)
        try:
            return abs(float(v))
        except (TypeError, ValueError):
            return -1.0
    rows_sorted = sorted(rows, key=safe_abs, reverse=True)
    return rows_sorted[:top_n], rows


def analyze_run(run_id):
    update_csv = f"{DIAG_SAVE_DIR}/run_{run_id}_update_log.csv"
    top_mdir, all_rows = find_extreme_row(update_csv, "m_dir_real", top_n=1)
    top_gd, _ = find_extreme_row(update_csv, "grad_dot_raw", top_n=1)

    top_row = top_mdir[0] if top_mdir else None
    gd_row = top_gd[0] if top_gd else None

    def window(rows, center_row, radius=2):
        if center_row is None:
            return []
        ep = center_row["episode"]
        idx = center_row["update_index_in_episode"]
        ep_rows = [r for r in rows if r["episode"] == ep]
        ep_rows_sorted = sorted(ep_rows, key=lambda r: int(r["update_index_in_episode"]))
        try:
            center_pos = next(i for i, r in enumerate(ep_rows_sorted) if r["update_index_in_episode"] == idx)
        except StopIteration:
            return []
        lo = max(0, center_pos - radius)
        hi = min(len(ep_rows_sorted), center_pos + radius + 1)
        return ep_rows_sorted[lo:hi]

    mdir_window = window(all_rows, top_row)
    gd_window = window(all_rows, gd_row)

    # clip log summary
    clip_counts = {"critic_net": 0, "critic_safe": 0, "actor_net": 0, "unknown": 0}
    clip_triggered = {"critic_net": 0, "critic_safe": 0, "actor_net": 0, "unknown": 0}
    max_post_clip_norm = {"critic_net": 0.0, "critic_safe": 0.0, "actor_net": 0.0}
    with open(f"{DIAG_SAVE_DIR}/run_{run_id}_clip_log.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            role = row["clip_role"]
            clip_counts[role] = clip_counts.get(role, 0) + 1
            if row["clip_triggered"] == "True":
                clip_triggered[role] = clip_triggered.get(role, 0) + 1
            if role in max_post_clip_norm:
                try:
                    v = abs(float(row["post_clip_norm"]))
                    if v > max_post_clip_norm[role]:
                        max_post_clip_norm[role] = v
                except ValueError:
                    pass

    return {
        "top_mdir_row": top_row,
        "top_gd_row": gd_row,
        "mdir_window": mdir_window,
        "gd_window": gd_window,
        "clip_counts": clip_counts,
        "clip_triggered": clip_triggered,
        "max_post_clip_norm": max_post_clip_norm,
    }


def print_row_brief(label, row):
    if row is None:
        print(f"    {label}: (not found)")
        return
    print(f"    {label}: ep={row['episode']} idx={row['update_index_in_episode']}")
    for k in ["m1", "m2", "denom1", "denom2", "m_dir_real", "m0", "m_soft", "rel_cap",
              "m_rel", "m_scale", "e_norm_pre_agentclip", "e_norm_post_agentclip",
              "e_clip_triggered", "dot_e_pre_W1", "dot_e_pre_W2",
              "dot_e_post_W1", "dot_e_post_W2", "grad_dot_raw", "h_value", "chain_mismatch"]:
        print(f"      {k:26s} = {row.get(k)}")


def main():
    print(f">>> Stage B-extended, n={len(TARGET_RUN_IDS)}: installing EXTENDED diagnostic patches "
          f"(agent.py / train.py unmodified) <<<")
    diag_hooks.install_patches(agent, train)

    summary = {}

    for run_id in TARGET_RUN_IDS:
        print(f"\n{'='*70}\n>>> run_id={run_id} (known Stage B finding: {KNOWN_ANOMALIES.get(run_id)}) <<<\n{'='*70}")
        diag_hooks.reset_logs()

        try:
            train.run_experiment(run_id, save_dir=DIAG_SAVE_DIR)
            completed = True
            error = None
        except Exception as e:
            completed = False
            error = repr(e)
            print(f"!!! run_id={run_id} raised an exception: {error}")

        if not completed:
            summary[run_id] = {"completed": False, "error": error, "trustworthy": False}
            continue

        episode_violations = list(train.episode_violations)
        final_env_steps = diag_hooks.get_final_env_step_count()

        update_csv_path = f"{DIAG_SAVE_DIR}/run_{run_id}_update_log.csv"
        episode_csv_path = f"{DIAG_SAVE_DIR}/run_{run_id}_episode_log.csv"
        n_update_rows, n_clip_rows = diag_hooks.write_update_log_csv(update_csv_path)
        n_episode_rows = diag_hooks.write_episode_log_csv(episode_csv_path, episode_violations, final_env_steps)

        mismatch_count, total_shadow_calls = diag_hooks.get_mismatch_count()
        chain_mismatch_count, _ = diag_hooks.get_chain_mismatch_count()

        all_match, details = regression_check(run_id)
        trustworthy = all_match and (mismatch_count == 0) and (chain_mismatch_count == 0)

        analysis = analyze_run(run_id) if trustworthy else None

        summary[run_id] = {
            "completed": True,
            "error": None,
            "n_update_rows": n_update_rows,
            "mismatch_count": mismatch_count,
            "chain_mismatch_count": chain_mismatch_count,
            "total_shadow_calls": total_shadow_calls,
            "regression_all_match": all_match,
            "regression_details": details,
            "trustworthy": trustworthy,
            "analysis": analysis,
        }

        print(f"[run_id={run_id}] completed. regression_match={all_match} "
              f"m_dir_mismatches={mismatch_count} chain_mismatches={chain_mismatch_count} "
              f"-> trustworthy={trustworthy}")

        if not all_match:
            print(f"!!! run_id={run_id}: REGRESSION MISMATCH -- diagnostic patches altered training. "
                  f"Halting analysis for this run_id. !!!")
            for k, eq, diff in details:
                if not eq:
                    print(f"    {k}: array_equal=False max|diff|={diff}")

        if mismatch_count > 0 or chain_mismatch_count > 0:
            print(f"!!! run_id={run_id}: SHADOW SELF-CONSISTENCY FAILURE "
                  f"(m_dir_mismatches={mismatch_count}, chain_mismatches={chain_mismatch_count}). "
                  f"Halting analysis for this run_id -- do not trust m_soft/rel_cap/m_scale values. !!!")

    diag_hooks.uninstall_patches(agent, train)
    print("\n>>> Patches uninstalled <<<")

    print(f"\n\n{'#'*70}\n# STAGE B-EXTENDED FINAL SUMMARY\n{'#'*70}")
    for run_id in TARGET_RUN_IDS:
        s = summary.get(run_id, {})
        print(f"\n--- run_id={run_id} ---")
        if not s.get("completed"):
            print(f"  NOT COMPLETED. error={s.get('error')}")
            continue
        print(f"  regression_all_match: {s['regression_all_match']}  "
              f"m_dir_mismatches: {s['mismatch_count']}/{s['total_shadow_calls']}  "
              f"chain_mismatches: {s['chain_mismatch_count']}/{s['total_shadow_calls']}  "
              f"trustworthy: {s['trustworthy']}")
        if not s["trustworthy"]:
            print("  (skipping detailed analysis -- gates failed)")
            continue
        a = s["analysis"]
        print("  [max |m_dir_real| row]")
        print_row_brief("m_dir peak", a["top_mdir_row"])
        print("  [max |grad_dot_raw| row]")
        print_row_brief("grad_dot peak", a["top_gd_row"])
        print(f"  clip counts: {a['clip_counts']}")
        print(f"  clip triggered: {a['clip_triggered']}")
        print(f"  max post_clip_norm by role: {a['max_post_clip_norm']}")
        print("  [window around m_dir peak, key columns]")
        for r in a["mdir_window"]:
            print(f"    idx={r['update_index_in_episode']:>4s} m1={r['m1']:>16s} m2={r['m2']:>16s} "
                  f"m_dir={r['m_dir_real']:>16s} m_soft={r['m_soft']:>12s} m_scale={r['m_scale']:>10s} "
                  f"e_pre={r['e_norm_pre_agentclip']:>10s} e_post={r['e_norm_post_agentclip']:>10s}")

    n_completed = sum(1 for s in summary.values() if s.get("completed"))
    n_trustworthy = sum(1 for s in summary.values() if s.get("trustworthy"))
    print(f"\n>>> {n_completed}/{len(TARGET_RUN_IDS)} runs completed. "
          f"{n_trustworthy}/{n_completed} passed both gates (trustworthy). <<<")


if __name__ == "__main__":
    main()
