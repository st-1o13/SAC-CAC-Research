"""
Diagnostic instrumentation for Proposal Ver.1, implemented entirely as
runtime monkeypatches applied from OUTSIDE proposal_ver1_py/. No file in
proposal_ver1_py/ is imported for modification and none is edited.

Method A (pure observation, near-zero risk):
    - W1_norm, W2_norm, grad_dot_raw: SoftActorCriticModel.update_critics_and_actor()
      already computes and returns these (as sa_norm_raw, st_norm_raw, dot_raw);
      train.py just never saves them. We patch the class method to intercept its
      return value, log it, and pass it back UNCHANGED to the real caller.
    - episode_length: gymnasium.make() is patched to wrap the returned env in a
      pass-through step/reset counting gym.Wrapper. All other behavior (spaces,
      reset(), step()) is delegated to the real env untouched.

Method B (thin wrap of a library function, near-zero risk):
    - clipping flags: torch.nn.utils.clip_grad_norm_ already returns the
      pre-clip total norm as documented PyTorch behavior. We patch it to
      capture that return value and pass it straight through.

Method C (shadow recomputation, higher risk -- isolated in diag_shadow.py):
    - m1, m2, denom1, denom2, dot12_shrunk: gradient_blend.restricted_direction()
      never exposes these. We patch the name `restricted_direction` as bound
      inside agent.py's namespace (NOT gradient_blend.py's own namespace,
      since `from gradient_blend import restricted_direction` already copied
      the reference into agent.py at import time). The wrapper ALWAYS calls
      the real, unmodified function to obtain the actual (e, m_dir) used for
      training, and separately calls diag_shadow.compute_shadow_values() for
      logging only. Every call's shadow-computed m_dir is compared against
      the real m_dir; any mismatch is recorded (see get_mismatch_count()).

All patches are applied by install_patches(agent_module, train_module) and
can be reverted by uninstall_patches(). Nothing is patched at import time.
"""
import csv
import os
import functools

import torch

import diag_shadow

_UPDATE_ROWS = []       # one row per update_critics_and_actor call (Stage2 only)
_EPISODE_LENGTHS = []   # completed episode lengths, in episode order
_MISMATCH_COUNT = 0
_TOTAL_SHADOW_CALLS = 0
_CHAIN_MISMATCH_COUNT = 0   # Stage B-extended: shadow_e vs real e (torch.allclose) mismatches

# Mirrors agent.py's `max_e_norm = 102.5` constant (agent.py:285). Used ONLY
# to derive e_norm_post_agentclip / post-clip dot products analytically from
# already-observed e_norm_pre_agentclip, since agent.py itself is not
# patched. If agent.py's constant ever changes, this must be updated too;
# the e_clip_triggered flag (independently computed the same way in both
# places) is cross-checked as a partial safeguard.
_AGENT_MAX_E_NORM = 102.5

_ctx = {
    "episode": None,
    "update_idx_in_episode": -1,
    "agent": None,
}

_originals = {}  # holds references to the real functions/methods for uninstall
_env_wrapper_instance = {"ref": None}


def reset_logs():
    global _MISMATCH_COUNT, _TOTAL_SHADOW_CALLS, _CHAIN_MISMATCH_COUNT
    _UPDATE_ROWS.clear()
    _EPISODE_LENGTHS.clear()
    _MISMATCH_COUNT = 0
    _TOTAL_SHADOW_CALLS = 0
    _CHAIN_MISMATCH_COUNT = 0
    _ctx["episode"] = None
    _ctx["update_idx_in_episode"] = -1
    _ctx["agent"] = None
    _env_wrapper_instance["ref"] = None


def get_mismatch_count():
    return _MISMATCH_COUNT, _TOTAL_SHADOW_CALLS


def get_chain_mismatch_count():
    """Stage B-extended: (# of shadow_e vs real_e allclose mismatches, total calls)."""
    return _CHAIN_MISMATCH_COUNT, _TOTAL_SHADOW_CALLS


def get_update_rows():
    return list(_UPDATE_ROWS)


def get_episode_lengths(final_env_step_count=None):
    lengths = list(_EPISODE_LENGTHS)
    if final_env_step_count is not None:
        lengths.append(final_env_step_count)
    return lengths


# ---------------------------------------------------------------------------
# Method A(1) + B: patch SoftActorCriticModel.update_critics_and_actor
# ---------------------------------------------------------------------------

def _make_update_wrapper(real_update_method):
    @functools.wraps(real_update_method)
    def wrapped(self, batch, episode_index):
        if _ctx["episode"] != episode_index:
            _ctx["episode"] = episode_index
            _ctx["update_idx_in_episode"] = -1
        _ctx["agent"] = self

        result = real_update_method(self, batch, episode_index=episode_index)
        # result is UNCHANGED here; we only read it for logging.
        return result

    return wrapped


# ---------------------------------------------------------------------------
# Method C (shadow) + Method A(2): patch agent.restricted_direction
# ---------------------------------------------------------------------------

def _make_restricted_direction_wrapper(real_restricted_direction):
    @functools.wraps(real_restricted_direction)
    def wrapped(W1, W2, progress=0.0, h_value=1.0, **kwargs):
        global _MISMATCH_COUNT, _TOTAL_SHADOW_CALLS, _CHAIN_MISMATCH_COUNT

        # --- the ACTUAL call used for training: unchanged, real function ---
        e, m_dir_real = real_restricted_direction(W1, W2, progress=progress, h_value=h_value, **kwargs)

        # --- diagnostic-only shadow recomputation (Method C) ---
        # (1) scalar-only shadow (already validated in Stage A/B, kept as-is)
        shadow = diag_shadow.compute_shadow_values(W1, W2, progress=progress, h_value=h_value)
        # (2) Stage B-extended: full pipeline through to the final vector e
        shadow_full = diag_shadow.compute_shadow_e_pipeline(W1, W2, progress=progress, h_value=h_value)

        e_norm_pre = float(e.norm().item()) if torch.is_tensor(e) else float("nan")

        _ctx["update_idx_in_episode"] += 1
        row = {
            "episode": _ctx["episode"],
            "update_index_in_episode": _ctx["update_idx_in_episode"],
            # Method A: direct tensor properties, same computation agent.py itself does
            "W1_norm": float(W1.norm().item()),
            "W2_norm": float(W2.norm().item()),
            "grad_dot_raw": float(torch.dot(W1, W2).item()),
            "progress": float(progress),
            "h_value": float(h_value),
            "e_norm_pre_agentclip": e_norm_pre,
            "m_dir_real": float(m_dir_real),
            # Method A: dot(e, W1)/dot(e, W2) using the REAL e (pre-agent-clip) --
            # direct tensor ops on already-available objects, no shadow risk
            "dot_e_pre_W1": float(torch.dot(e, W1).item()) if torch.is_tensor(e) else None,
            "dot_e_pre_W2": float(torch.dot(e, W2).item()) if torch.is_tensor(e) else None,
        }

        _TOTAL_SHADOW_CALLS += 1
        if shadow["valid"]:
            row.update({
                "dot12_shrunk": shadow["dot12_shrunk"],
                "w1_sq": shadow["w1_sq"],
                "w2_sq": shadow["w2_sq"],
                "denom1": shadow["denom1"],
                "denom2": shadow["denom2"],
                "m1": shadow["m1"],
                "m2": shadow["m2"],
                "m_progress": shadow["m_progress"],
                "safety_level": shadow["safety_level"],
                "m_dir_shadow": shadow["m_dir_shadow"],
            })
            diff = abs(shadow["m_dir_shadow"] - m_dir_real)
            tol = 1e-6 * max(1.0, abs(m_dir_real))
            row["m_dir_mismatch"] = diff > tol
            row["m_dir_abs_diff"] = diff
            if row["m_dir_mismatch"]:
                _MISMATCH_COUNT += 1
        else:
            row.update({
                "dot12_shrunk": None, "w1_sq": None, "w2_sq": None,
                "denom1": None, "denom2": None, "m1": None, "m2": None,
                "m_progress": None, "safety_level": None, "m_dir_shadow": None,
                "m_dir_mismatch": False, "m_dir_abs_diff": None,
            })

        # --- Stage B-extended: post-m_dir pipeline (m0, m_soft, rel_cap, m_scale) ---
        if shadow_full["valid"] and not shadow_full.get("degenerate_branch", False):
            row.update({
                "m0": shadow_full["m0"],
                "m_soft": shadow_full["m_soft"],
                "rel_cap": shadow_full["rel_cap"],
                "m_rel": shadow_full["m_rel"],
                "m_scale": shadow_full["m_scale"],
                "e_norm_shadow": shadow_full["shadow_e_norm"],
            })
            chain_diff = float((shadow_full["shadow_e"] - e).abs().max().item()) if torch.is_tensor(e) else None
            chain_ok = torch.allclose(shadow_full["shadow_e"], e, rtol=1e-4, atol=1e-6) if torch.is_tensor(e) else False
            row["chain_mismatch"] = not chain_ok
            row["chain_abs_diff"] = chain_diff
            if not chain_ok:
                _CHAIN_MISMATCH_COUNT += 1
        else:
            row.update({
                "m0": None, "m_soft": None, "rel_cap": None, "m_rel": None,
                "m_scale": None, "e_norm_shadow": None,
                "chain_mismatch": False, "chain_abs_diff": None,
            })

        row["e_clip_triggered"] = e_norm_pre > _AGENT_MAX_E_NORM

        # --- derived (not hooked) post-agent-clip quantities ---
        # agent.py's 102.5 clip is a pure scalar rescale that preserves direction,
        # so post-clip norm/dot-products are exactly derivable from the pre-clip
        # values already observed above, with no need to patch agent.py itself.
        if e_norm_pre > 0:
            post_scale = min(1.0, _AGENT_MAX_E_NORM / e_norm_pre)
        else:
            post_scale = 1.0
        row["e_norm_post_agentclip"] = e_norm_pre * post_scale
        row["dot_e_post_W1"] = row["dot_e_pre_W1"] * post_scale if row["dot_e_pre_W1"] is not None else None
        row["dot_e_post_W2"] = row["dot_e_pre_W2"] * post_scale if row["dot_e_pre_W2"] is not None else None

        _UPDATE_ROWS.append(row)
        return e, m_dir_real  # unchanged pass-through

    return wrapped


# ---------------------------------------------------------------------------
# Method B: patch torch.nn.utils.clip_grad_norm_
# ---------------------------------------------------------------------------

def _make_clip_grad_norm_wrapper(real_clip_fn):
    @functools.wraps(real_clip_fn)
    def wrapped(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        pre_clip_norm = real_clip_fn(params, max_norm, *args, **kwargs)

        agent = _ctx.get("agent")
        role = "unknown"
        if agent is not None:
            def _first_param_id(mod):
                try:
                    return id(next(iter(mod.parameters())))
                except StopIteration:
                    return None
            first_id = id(params[0]) if params else None
            if first_id == _first_param_id(agent.actor_net):
                role = "actor_net"
            elif first_id == _first_param_id(agent.critic_net):
                role = "critic_net"
            elif first_id == _first_param_id(agent.critic_safe):
                role = "critic_safe"

        norm_val = float(pre_clip_norm.item()) if torch.is_tensor(pre_clip_norm) else float(pre_clip_norm)

        # Stage B-extended: clip_grad_norm_ clips .grad IN-PLACE, so re-reading
        # the same tensors' norm right after the real call gives the actual
        # post-clip gradient norm -- a pure observation of already-mutated
        # tensors, not a re-derivation of any algorithm.
        post_total_sq = 0.0
        for p in params:
            if p.grad is not None:
                post_total_sq += float(p.grad.detach().norm(2).item()) ** 2
        post_clip_norm = post_total_sq ** 0.5

        row = {
            "episode": _ctx["episode"],
            "update_index_in_episode": _ctx["update_idx_in_episode"],
            "clip_role": role,
            "max_norm": float(max_norm),
            "pre_clip_norm": norm_val,
            "post_clip_norm": post_clip_norm,
            "clip_triggered": norm_val > float(max_norm),
        }
        _UPDATE_ROWS.append({"_clip_event": row})
        return pre_clip_norm

    return wrapped


# ---------------------------------------------------------------------------
# Method A(3): patch gymnasium.make to attach a step/reset counting wrapper
# ---------------------------------------------------------------------------

def _make_env_step_counter_class(gym_module):
    class _StepCountingWrapper(gym_module.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.episode_step_count = 0
            self.completed_episode_lengths = []

        def reset(self, **kwargs):
            if self.episode_step_count > 0:
                self.completed_episode_lengths.append(self.episode_step_count)
                _EPISODE_LENGTHS.append(self.episode_step_count)
            self.episode_step_count = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            self.episode_step_count += 1
            return self.env.step(action)

    return _StepCountingWrapper


def _make_gym_make_wrapper(real_make, step_counting_cls):
    @functools.wraps(real_make)
    def wrapped(*args, **kwargs):
        env = real_make(*args, **kwargs)
        wrapped_env = step_counting_cls(env)
        _env_wrapper_instance["ref"] = wrapped_env
        return wrapped_env

    return wrapped


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install_patches(agent_module, train_module):
    """Apply all Method A/B/C patches. Call uninstall_patches() when done."""
    reset_logs()

    # Method A(1)+B(context): update_critics_and_actor
    _originals["update_critics_and_actor"] = agent_module.SoftActorCriticModel.update_critics_and_actor
    agent_module.SoftActorCriticModel.update_critics_and_actor = _make_update_wrapper(
        _originals["update_critics_and_actor"]
    )

    # Method C + A(2): restricted_direction (patched in agent.py's namespace,
    # since that's where `from gradient_blend import restricted_direction`
    # bound the name train-time code actually calls)
    _originals["restricted_direction"] = agent_module.restricted_direction
    agent_module.restricted_direction = _make_restricted_direction_wrapper(
        _originals["restricted_direction"]
    )

    # Method B: torch.nn.utils.clip_grad_norm_ (global)
    _originals["clip_grad_norm_"] = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = _make_clip_grad_norm_wrapper(_originals["clip_grad_norm_"])

    # Method A(3): gymnasium.make, patched via train_module's own bound `gym` name
    # (train.py did `import gymnasium as gym`, so train_module.gym IS the real
    # gymnasium module object -- patching train_module.gym.make patches the
    # same module object gymnasium itself refers to).
    gym_module = train_module.gym
    _originals["gym_make"] = gym_module.make
    step_counting_cls = _make_env_step_counter_class(gym_module)
    gym_module.make = _make_gym_make_wrapper(_originals["gym_make"], step_counting_cls)


def uninstall_patches(agent_module, train_module):
    if "update_critics_and_actor" in _originals:
        agent_module.SoftActorCriticModel.update_critics_and_actor = _originals["update_critics_and_actor"]
    if "restricted_direction" in _originals:
        agent_module.restricted_direction = _originals["restricted_direction"]
    if "clip_grad_norm_" in _originals:
        torch.nn.utils.clip_grad_norm_ = _originals["clip_grad_norm_"]
    if "gym_make" in _originals:
        train_module.gym.make = _originals["gym_make"]
    _originals.clear()


def get_final_env_step_count():
    ref = _env_wrapper_instance["ref"]
    return ref.episode_step_count if ref is not None else None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_update_log_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    update_rows = [r for r in _UPDATE_ROWS if "_clip_event" not in r]
    clip_rows = [r["_clip_event"] for r in _UPDATE_ROWS if "_clip_event" in r]

    fieldnames = [
        "episode", "update_index_in_episode", "W1_norm", "W2_norm", "grad_dot_raw",
        "dot12_shrunk", "w1_sq", "w2_sq", "denom1", "denom2", "m1", "m2",
        "m_progress", "safety_level", "m_dir_shadow", "m_dir_real",
        "m_dir_mismatch", "m_dir_abs_diff",
        # Stage B-extended additions:
        "m0", "m_soft", "rel_cap", "m_rel", "m_scale", "e_norm_shadow",
        "chain_mismatch", "chain_abs_diff",
        "e_norm_pre_agentclip", "e_norm_post_agentclip", "e_clip_triggered",
        "dot_e_pre_W1", "dot_e_pre_W2", "dot_e_post_W1", "dot_e_post_W2",
        "progress", "h_value",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in update_rows:
            writer.writerow(r)

    clip_path = path.replace("_update_log.csv", "_clip_log.csv")
    with open(clip_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "update_index_in_episode", "clip_role", "max_norm", "pre_clip_norm", "post_clip_norm", "clip_triggered"])
        writer.writeheader()
        for r in clip_rows:
            writer.writerow(r)

    return len(update_rows), len(clip_rows)


def write_episode_log_csv(path, episode_violations, final_env_step_count):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lengths = get_episode_lengths(final_env_step_count)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "episode_length", "episode_violations", "violation_rate"])
        for i, viol in enumerate(episode_violations):
            length = lengths[i] if i < len(lengths) else None
            rate = (viol / length) if length else None
            writer.writerow([i + 1, length, viol, rate])
    return len(episode_violations)
