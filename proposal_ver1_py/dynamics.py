# Auto-extracted from Proposal_Ver1.ipynb.ipynb.
# Do not hand-edit the extracted logic; this file is a verbatim relocation,
# not a rewrite. Only import statements were added/adjusted for standalone execution.

import torch

def model_dynamics(x: torch.Tensor, a: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
    """
    Pendulum-v1 の離散時間 Dynamics を分析的に実装。
      x = [cosθ, sinθ, θ̇]
      u = a.squeeze(-1)
      θ̈ = −3g/(2l) sinθ + 3/(m l^2) u
      θ_{t+1} = θ + θ̇·dt
      θ̇_{t+1} = θ̇ + θ̈·dt
    """
    g, l, m = 10.0, 1.0, 1.0
    cos_th, sin_th, thdot = x[:,0], x[:,1], x[:,2]
    th   = torch.atan2(sin_th, cos_th)
    u    = a.squeeze(-1)
    thdd = -3 * g/(2*l) * torch.sin(th) + 3.0/(m*l**2) * u

    th_next    = th + thdot * dt
    thdot_next = thdot + thdd  * dt

    return torch.stack([torch.cos(th_next),
                        torch.sin(th_next),
                        thdot_next],
                       dim=1)
