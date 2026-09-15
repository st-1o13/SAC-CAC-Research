# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import torch


def restricted_direction(
    W1: torch.Tensor,
    W2: torch.Tensor,
    progress: float = 0.0,
    eps: float = 1e-8,
    use_log10_shrink: bool = True,
    log1p_variant: bool = True,
    max_update_norm: float = 4e2,
    debug: bool = False,
    h_value: float = 1.0,     # ← ★追加：安全度 h(s) を渡す
    k: float = 1.63,           # ← ★追加：安全度の鋭さ
    c: float = 0.55            # ← ★追加：安全の閾値
) -> torch.Tensor:
    """
    restricted_direction + 安全度ベースの m_dir 調整（案1）
    h_value: h(s) の平均値（安全度）
    k:       sigmoid の鋭さ
    c:       安全と危険の境界
    """

    device = W1.device
    dtype = W1.dtype

    # --- 既存の安全チェック ---
    if W1.dim() != 1 or W2.dim() != 1:
        raise ValueError("W1 and W2 must be 1D tensors")
    norm1 = W1.norm()
    norm2 = W2.norm()
    if not torch.isfinite(norm1) or not torch.isfinite(norm2):
        return torch.zeros_like(W1)
    if norm1 == 0 and norm2 == 0:
        return torch.zeros_like(W1)
    if norm1 == 0:
        return W2.clone()
    if norm2 == 0:
        return W1.clone()

    # --- W2 の log shrink（既存） ---
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

    # --- m1, m2 の計算（既存） ---
    dot12 = torch.dot(W1, W2_use)
    w1_sq = (norm1 ** 2).clamp(min=eps)
    w2_sq = (norm2 ** 2).clamp(min=eps)

    denom1 = dot12 - w1_sq
    denom1_safe = torch.where(torch.abs(denom1) < eps, torch.sign(denom1) * eps + eps, denom1)
    m1 = dot12 / denom1_safe

    denom2 = w2_sq - dot12
    denom2_safe = torch.where(torch.abs(denom2) < eps, torch.sign(denom2) * eps + eps, denom2)
    m2 = w2_sq / denom2_safe

    # --- 既存の progress ベースのスケジュール ---
    t = float(min(max(progress, 0.0), 1.0))
    m_progress = m2 * (1.0 - t) + m1 * t

    # ============================================================
    # ★★★ ここが案1の核心：安全度ベースの m_dir を導入 ★★★
    # ============================================================

    # h_value が小さい（危険） → safety_level ≈ 1 → m_dir ≈ m1（安全方向）
    # h_value が大きい（安全） → safety_level ≈ 0 → m_dir ≈ m2（性能方向）
    safety_input = torch.tensor(k * (c - h_value), dtype=W1.dtype, device=W1.device)
    safety_level = torch.sigmoid(safety_input)

    # progress と安全度をブレンド
    # safety_level が強いときは安全方向を優先
    m_dir = safety_level * m1 + (1 - safety_level) * m_progress

    # --- m_dir を m1, m2 の範囲にクリップ ---
    m_min = torch.min(m1, m2)
    m_max = torch.max(m1, m2)
    m_dir = torch.clamp(m_dir, min=float(m_min), max=float(m_max))

    # --- 方向ベクトル ---
    w_dir = m_dir * W1 + (1.0 - m_dir) * W2_use

    if w_dir.norm() < eps or not torch.isfinite(w_dir).all():
        return W2_use.clone()

    # --- スケール計算（既存） ---
    w_dir_norm_sq = (w_dir.norm() ** 2).clamp(min=eps)
    m0 = torch.dot(W2_use, w_dir) / w_dir_norm_sq

    m_thresh = 100.0
    p = 1.5
    m0_sign = torch.sign(m0)
    m0_abs = m0.abs()
    m_shrunk = (m0_abs / (1.0 + (m0_abs / m_thresh) ** p)).clamp(min=0.0)
    m_soft = m0_sign * m_shrunk

    W2u_norm = W2_use.norm().clamp(min=1e-6)
    k_rel = 4.5
    rel_cap = k_rel * (W2u_norm + 1e-12)
    m_rel = torch.sign(m_soft) * torch.min(m_soft.abs(), rel_cap)

    M = 21.0
    m_scale = torch.clamp(m_rel, min=-M, max=M)

    w_scaled = m_scale * w_dir

    # --- 最終クリップ ---
    if w_scaled.norm() > max_update_norm:
        w_scaled = w_scaled * (max_update_norm / (w_scaled.norm() + eps))

    if debug:
        print(f"h_value = {h_value}")

    return torch.nan_to_num(w_scaled, nan=0.0), float(m_dir)


def restricted_direction1(
    W1: torch.Tensor,
    W2: torch.Tensor,
    progress: float = 0.0,
    eps: float = 1e-8,
    use_log10_shrink: bool = True,
    log1p_variant: bool = True,
    max_update_norm: float = 4e2,
    debug: bool = False,
    h_value: float = 1.0,     # ← ★追加：安全度 h(s) を渡す
    k: float = 0.8,#1.63,           # ← ★追加：安全度の鋭さ
    c: float = 0.55            # ← ★追加：安全の閾値
) -> torch.Tensor:
    """
    restricted_direction + 安全度ベースの m_dir 調整（案1）
    h_value: h(s) の平均値（安全度）
    k:       sigmoid の鋭さ
    c:       安全と危険の境界
    """

    device = W1.device
    dtype = W1.dtype

    # --- 既存の安全チェック ---
    if W1.dim() != 1 or W2.dim() != 1:
        raise ValueError("W1 and W2 must be 1D tensors")
    norm1 = W1.norm()
    norm2 = W2.norm()
    if not torch.isfinite(norm1) or not torch.isfinite(norm2):
        return torch.zeros_like(W1)
    if norm1 == 0 and norm2 == 0:
        return torch.zeros_like(W1)
    if norm1 == 0:
        return W2.clone()
    if norm2 == 0:
        return W1.clone()

    # --- W2 の log shrink（既存） ---
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

    # --- m1, m2 の計算（既存） ---
    dot12 = torch.dot(W1, W2_use)
    w1_sq = (norm1 ** 2).clamp(min=eps)
    w2_sq = (norm2 ** 2).clamp(min=eps)

    denom1 = dot12 - w1_sq
    denom1_safe = torch.where(torch.abs(denom1) < eps, torch.sign(denom1) * eps + eps, denom1)
    m1 = dot12 / denom1_safe

    denom2 = w2_sq - dot12
    denom2_safe = torch.where(torch.abs(denom2) < eps, torch.sign(denom2) * eps + eps, denom2)
    m2 = w2_sq / denom2_safe

    # --- 既存の progress ベースのスケジュール ---
    t = float(min(max(progress, 0.0), 1.0))
    m_progress = m1 * (1.0 - t) + m2 * t

    # ============================================================
    # ★★★ ここが案1の核心：安全度ベースの m_dir を導入 ★★★
    # ============================================================
    # m1 が表す方向（W1 に直交するところで W2 寄り）
    dir1 = m1 * W1 + (1.0 - m1) * W2_use

    # m2 が表す方向（W2 に直交するところで W1 寄り）
    dir2 = m2 * W1 + (1.0 - m2) * W2_use

    # h_value が小さい（危険） → safety_level ≈ 1 → m_dir ≈ m1（安全方向）
    # h_value が大きい（安全） → safety_level ≈ 0 → m_dir ≈ m2（性能方向）
    safety_input = torch.tensor(k * (c - h_value), dtype=W1.dtype, device=W1.device)
    safety_level = torch.sigmoid(safety_input)

    # progress と安全度をブレンド
    # safety_level が強いときは安全方向を優先
    #m_dir = safety_level * m2 + (1 - safety_level) * m_progress
    m_dir = (1.0 - safety_level) * dir1 + safety_level * dir2

    # --- m_dir を m1, m2 の範囲にクリップ ---
    #m_min = torch.min(m1, m2)
    #m_max = torch.max(m1, m2)
    #m_dir = torch.clamp(m_dir, min=float(m_min), max=float(m_max))

    # --- 方向ベクトル ---
    #w_dir = m_dir * W1 + (1.0 - m_dir) * W2_use

    if m_dir.norm() < eps or not torch.isfinite(m_dir).all():
        return W2_use.clone()

    # --- スケール計算（既存） ---
    m_dir_norm_sq = (m_dir.norm() ** 2).clamp(min=eps)
    m0 = torch.dot(W2_use, m_dir) / m_dir_norm_sq

    m_thresh = 100.0
    p = 1.5
    m0_sign = torch.sign(m0)
    m0_abs = m0.abs()
    m_shrunk = (m0_abs / (1.0 + (m0_abs / m_thresh) ** p)).clamp(min=0.0)
    m_soft = m0_sign * m_shrunk

    W2u_norm = W2_use.norm().clamp(min=1e-6)
    k_rel = 4.5
    rel_cap = k_rel * (W2u_norm + 1e-12)
    m_rel = torch.sign(m_soft) * torch.min(m_soft.abs(), rel_cap)

    M = 21.0
    m_scale = torch.clamp(m_rel, min=-M, max=M)

    w_scaled = m_scale * m_dir

    # --- 最終クリップ ---
    if w_scaled.norm() > max_update_norm:
        w_scaled = w_scaled * (max_update_norm / (w_scaled.norm() + eps))

    if debug:
        print(f"h_value = {h_value}")

    return torch.nan_to_num(w_scaled, nan=0.0), m_dir.clone()
