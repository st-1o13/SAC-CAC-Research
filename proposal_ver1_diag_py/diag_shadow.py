"""
Stage C: shadow (read-only) recomputation of restricted_direction()'s
internal intermediate values (m1, m2, denom1, denom2, dot12, ...).

This module does NOT modify, import as a replacement for, or otherwise
alter gradient_blend.py. It is a hand-verified line-for-line copy of the
math in gradient_blend.restricted_direction() (lines 34-90 of that file,
as of the Proposal Ver.1 100-run production commit), used ONLY to expose
intermediate values for diagnostic logging. It is never used to compute
values that feed back into training -- the actual training path always
calls the real, unmodified gradient_blend.restricted_direction().

Because this is a manual transcription, it carries a real risk of drift
from the original if that file ever changes, or if the transcription
itself has a mistake. Mitigation: compute_shadow_values() is designed to
be independently testable (see self_test() below and diag_hooks.py's
per-call consistency check against the real function's returned m_dir).
"""
import torch


def compute_shadow_values(
    W1: torch.Tensor,
    W2: torch.Tensor,
    progress: float,
    h_value: float,
    eps: float = 1e-8,
    use_log10_shrink: bool = True,
    log1p_variant: bool = True,
    k: float = 1.63,
    c: float = 0.55,
) -> dict:
    """Re-derive m1, m2, denom1, denom2, dot12, m_dir (pre-scale) from the
    same inputs restricted_direction() receives. Read-only / diagnostic use
    only. Returns a plain dict of Python floats (no tensors retained)."""

    norm1 = W1.norm()
    norm2 = W2.norm()

    if not torch.isfinite(norm1) or not torch.isfinite(norm2):
        return {"valid": False, "reason": "non_finite_norm"}
    if norm1 == 0 and norm2 == 0:
        return {"valid": False, "reason": "both_zero"}
    if norm1 == 0 or norm2 == 0:
        return {"valid": False, "reason": "one_zero"}

    # --- W2 log shrink (verbatim copy of gradient_blend.py lines 46-55) ---
    if use_log10_shrink:
        if log1p_variant:
            log_norm2 = torch.log10(norm2 + 1.0)
        else:
            log_norm2 = torch.log10(norm2 + eps)
        log_norm2_pos = torch.clamp(log_norm2, min=0.0)
        shrink = 1.0 / (1.0 + log_norm2_pos)
        W2_use = W2 * shrink
    else:
        W2_use = W2

    # --- m1, m2 (verbatim copy of gradient_blend.py lines 58-68) ---
    dot12 = torch.dot(W1, W2_use)
    w1_sq = (norm1 ** 2).clamp(min=eps)
    w2_sq = (norm2 ** 2).clamp(min=eps)

    denom1 = dot12 - w1_sq
    denom1_safe = torch.where(torch.abs(denom1) < eps, torch.sign(denom1) * eps + eps, denom1)
    m1 = dot12 / denom1_safe

    denom2 = w2_sq - dot12
    denom2_safe = torch.where(torch.abs(denom2) < eps, torch.sign(denom2) * eps + eps, denom2)
    m2 = w2_sq / denom2_safe

    # --- progress-based schedule (verbatim copy of lines 70-72) ---
    t = float(min(max(progress, 0.0), 1.0))
    m_progress = m2 * (1.0 - t) + m1 * t

    # --- safety-level blend (verbatim copy of lines 78-90) ---
    safety_input = torch.tensor(k * (c - h_value), dtype=W1.dtype, device=W1.device)
    safety_level = torch.sigmoid(safety_input)

    m_dir = safety_level * m1 + (1 - safety_level) * m_progress

    m_min = torch.min(m1, m2)
    m_max = torch.max(m1, m2)
    m_dir_clamped = torch.clamp(m_dir, min=float(m_min), max=float(m_max))

    return {
        "valid": True,
        "reason": "",
        "norm1_W1": float(norm1.item()),
        "norm2_W2": float(norm2.item()),
        "dot12_shrunk": float(dot12.item()),
        "w1_sq": float(w1_sq.item()),
        "w2_sq": float(w2_sq.item()),
        "denom1": float(denom1.item()),
        "denom2": float(denom2.item()),
        "m1": float(m1.item()),
        "m2": float(m2.item()),
        "m_progress": float(m_progress.item()),
        "safety_level": float(safety_level.item()),
        "m_dir_shadow_preclamp": float(m_dir.item()),
        "m_dir_shadow": float(m_dir_clamped.item()),
    }


def compute_shadow_e_pipeline(
    W1: torch.Tensor,
    W2: torch.Tensor,
    progress: float,
    h_value: float,
    eps: float = 1e-8,
    use_log10_shrink: bool = True,
    log1p_variant: bool = True,
    k: float = 1.63,
    c: float = 0.55,
    max_update_norm: float = 4e2,
) -> dict:
    """Extended shadow computation: verbatim copy of the FULL body of
    gradient_blend.restricted_direction() (norm1/norm2 through the final
    max_update_norm clip and nan_to_num), not just the m1/m2/m_dir portion
    covered by compute_shadow_values(). Independent, self-contained
    transcription (does not call compute_shadow_values()) so that a bug in
    one does not silently mask a bug in the other -- diag_hooks.py's
    consistency check cross-validates m_dir_shadow between the two AND
    compares the final shadow_e tensor against the real function's return
    value via torch.allclose().

    Diagnostic use only: never used to compute a value that feeds back into
    training. Returns a dict of Python floats/bools plus one tensor key
    ("shadow_e") for allclose comparison by the caller.
    """
    if W1.dim() != 1 or W2.dim() != 1:
        return {"valid": False, "reason": "not_1d"}

    norm1 = W1.norm()
    norm2 = W2.norm()

    if not torch.isfinite(norm1) or not torch.isfinite(norm2):
        return {"valid": False, "reason": "non_finite_norm"}
    if norm1 == 0 and norm2 == 0:
        return {"valid": False, "reason": "both_zero"}
    if norm1 == 0 or norm2 == 0:
        return {"valid": False, "reason": "one_zero"}

    if use_log10_shrink:
        if log1p_variant:
            log_norm2 = torch.log10(norm2 + 1.0)
        else:
            log_norm2 = torch.log10(norm2 + eps)
        log_norm2_pos = torch.clamp(log_norm2, min=0.0)
        shrink = 1.0 / (1.0 + log_norm2_pos)
        W2_use = W2 * shrink
    else:
        W2_use = W2

    dot12 = torch.dot(W1, W2_use)
    w1_sq = (norm1 ** 2).clamp(min=eps)
    w2_sq = (norm2 ** 2).clamp(min=eps)

    denom1 = dot12 - w1_sq
    denom1_safe = torch.where(torch.abs(denom1) < eps, torch.sign(denom1) * eps + eps, denom1)
    m1 = dot12 / denom1_safe

    denom2 = w2_sq - dot12
    denom2_safe = torch.where(torch.abs(denom2) < eps, torch.sign(denom2) * eps + eps, denom2)
    m2 = w2_sq / denom2_safe

    t = float(min(max(progress, 0.0), 1.0))
    m_progress = m2 * (1.0 - t) + m1 * t

    safety_input = torch.tensor(k * (c - h_value), dtype=W1.dtype, device=W1.device)
    safety_level = torch.sigmoid(safety_input)

    m_dir = safety_level * m1 + (1 - safety_level) * m_progress

    m_min = torch.min(m1, m2)
    m_max = torch.max(m1, m2)
    m_dir = torch.clamp(m_dir, min=float(m_min), max=float(m_max))

    # --- everything below is NEW relative to compute_shadow_values() ---
    w_dir = m_dir * W1 + (1.0 - m_dir) * W2_use

    if w_dir.norm() < eps or not torch.isfinite(w_dir).all():
        return {
            "valid": True, "reason": "w_dir_degenerate_returns_W2_use",
            "m_dir_shadow": float(m_dir.item()),
            "shadow_e": W2_use.clone(),
            "degenerate_branch": True,
        }

    w_dir_norm_sq = (w_dir.norm() ** 2).clamp(min=eps)
    m0 = torch.dot(W2_use, w_dir) / w_dir_norm_sq

    m_thresh = 100.0
    p_exp = 1.5
    m0_sign = torch.sign(m0)
    m0_abs = m0.abs()
    m_shrunk = (m0_abs / (1.0 + (m0_abs / m_thresh) ** p_exp)).clamp(min=0.0)
    m_soft = m0_sign * m_shrunk

    W2u_norm = W2_use.norm().clamp(min=1e-6)
    k_rel = 4.5
    rel_cap = k_rel * (W2u_norm + 1e-12)
    m_rel = torch.sign(m_soft) * torch.min(m_soft.abs(), rel_cap)

    M = 21.0
    m_scale = torch.clamp(m_rel, min=-M, max=M)

    w_scaled = m_scale * w_dir

    pre_final_clip_norm = float(w_scaled.norm().item())
    final_clip_triggered = pre_final_clip_norm > max_update_norm
    if final_clip_triggered:
        w_scaled = w_scaled * (max_update_norm / (w_scaled.norm() + eps))

    shadow_e = torch.nan_to_num(w_scaled, nan=0.0)

    return {
        "valid": True,
        "reason": "",
        "degenerate_branch": False,
        "m1": float(m1.item()),
        "m2": float(m2.item()),
        "denom1": float(denom1.item()),
        "denom2": float(denom2.item()),
        "m_dir_shadow": float(m_dir.item()),
        "m0": float(m0.item()),
        "m_soft": float(m_soft.item()),
        "rel_cap": float(rel_cap.item()),
        "m_rel": float(m_rel.item()),
        "m_scale": float(m_scale.item()),
        "pre_final_clip_norm": pre_final_clip_norm,
        "final_clip_triggered": final_clip_triggered,
        "shadow_e": shadow_e,          # tensor, for torch.allclose comparison
        "shadow_e_norm": float(shadow_e.norm().item()),
    }


def self_test_e_pipeline(n_trials: int = 2000, seed: int = 1, verbose: bool = True) -> bool:
    """Independent correctness check for compute_shadow_e_pipeline(): compares
    its returned shadow_e tensor (via torch.allclose) and its m_dir_shadow
    (cross-checked against the already-validated compute_shadow_values()) to
    the REAL gradient_blend.restricted_direction()'s actual return value, on
    random and near-degenerate inputs."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proposal_ver1_py"))
    from gradient_blend import restricted_direction  # unmodified, read-only import

    torch.manual_seed(seed)
    n_mismatch_e = 0
    n_mismatch_mdir_cross = 0
    n_invalid = 0
    max_e_abs_diff = 0.0

    for i in range(n_trials):
        dim = 16
        if i % 3 == 0:
            W1 = torch.randn(dim)
            W2 = torch.randn(dim)
        elif i % 3 == 1:
            W1 = torch.randn(dim)
            scale = torch.randn(1).item()
            W2 = W1 * scale + torch.randn(dim) * 1e-4
        else:
            W1 = torch.randn(dim) * 1e-3
            W2 = torch.randn(dim) * 1e-3

        progress = float(torch.rand(1).item())
        h_value = float(torch.randn(1).item())

        shadow_full = compute_shadow_e_pipeline(W1, W2, progress=progress, h_value=h_value)
        shadow_scalars = compute_shadow_values(W1, W2, progress=progress, h_value=h_value)
        real_e, real_m_dir = restricted_direction(W1, W2, progress=progress, h_value=h_value)

        if not shadow_full["valid"]:
            n_invalid += 1
            continue

        # cross-check against the already-validated scalar-only shadow function
        if shadow_scalars["valid"]:
            diff_cross = abs(shadow_full["m_dir_shadow"] - shadow_scalars["m_dir_shadow"])
            if diff_cross > 1e-6 * max(1.0, abs(shadow_scalars["m_dir_shadow"])):
                n_mismatch_mdir_cross += 1
                if verbose:
                    print(f"[CROSS MISMATCH] trial={i} full={shadow_full['m_dir_shadow']!r} "
                          f"scalars={shadow_scalars['m_dir_shadow']!r}")

        ok = torch.allclose(shadow_full["shadow_e"], real_e, rtol=1e-4, atol=1e-6)
        diff = float((shadow_full["shadow_e"] - real_e).abs().max().item())
        max_e_abs_diff = max(max_e_abs_diff, diff)
        if not ok:
            n_mismatch_e += 1
            if verbose:
                print(f"[E MISMATCH] trial={i} shadow_e_norm={shadow_full['shadow_e_norm']!r} "
                      f"real_e_norm={float(real_e.norm().item())!r} max_abs_diff={diff!r}")

    all_ok = (n_mismatch_e == 0) and (n_mismatch_mdir_cross == 0)
    if verbose:
        print(f"self_test_e_pipeline: trials={n_trials} invalid_skipped={n_invalid} "
              f"e_mismatches={n_mismatch_e} cross_mdir_mismatches={n_mismatch_mdir_cross} "
              f"max_e_abs_diff={max_e_abs_diff:.3e} -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def self_test(n_trials: int = 2000, seed: int = 0, verbose: bool = True) -> bool:
    """Independent correctness check: compares compute_shadow_values()'s
    m_dir against the REAL gradient_blend.restricted_direction()'s returned
    m_dir, on random inputs (including near-degenerate cases designed to
    stress the denom1/denom2-near-zero condition). Does not touch any
    Proposal Ver.1 file; only imports gradient_blend.py read-only.

    Returns True iff all trials match within tolerance.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proposal_ver1_py"))
    from gradient_blend import restricted_direction  # unmodified, read-only import

    torch.manual_seed(seed)
    n_mismatch = 0
    n_invalid = 0
    max_abs_diff = 0.0

    for i in range(n_trials):
        dim = 16
        if i % 3 == 0:
            # generic random case
            W1 = torch.randn(dim)
            W2 = torch.randn(dim)
        elif i % 3 == 1:
            # near-degenerate case: W2 close to a scaled copy of W1 to
            # stress denom1/denom2 -> near zero
            W1 = torch.randn(dim)
            scale = torch.randn(1).item()
            W2 = W1 * scale + torch.randn(dim) * 1e-4
        else:
            # small-norm case
            W1 = torch.randn(dim) * 1e-3
            W2 = torch.randn(dim) * 1e-3

        progress = float(torch.rand(1).item())
        h_value = float(torch.randn(1).item())

        shadow = compute_shadow_values(W1, W2, progress=progress, h_value=h_value)
        real_e, real_m_dir = restricted_direction(W1, W2, progress=progress, h_value=h_value)

        if not shadow["valid"]:
            n_invalid += 1
            continue

        diff = abs(shadow["m_dir_shadow"] - real_m_dir)
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-6 * max(1.0, abs(real_m_dir)):
            n_mismatch += 1
            if verbose:
                print(f"[MISMATCH] trial={i} shadow={shadow['m_dir_shadow']!r} real={real_m_dir!r} diff={diff!r}")

    ok = (n_mismatch == 0)
    if verbose:
        print(f"self_test: trials={n_trials} invalid_skipped={n_invalid} mismatches={n_mismatch} "
              f"max_abs_diff={max_abs_diff:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    self_test()
    self_test_e_pipeline()
