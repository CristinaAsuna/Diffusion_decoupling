"""Conditional rectified-flow matching for the existing D-only interface.

The model shares the diffusion U-Net and D-only encoding/decoding helpers, but
learns the velocity of the straight path from a Gaussian latent to D.  Sampling
is an ODE solve, hence deterministic once the initial latent is fixed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .diffusion import GaussianConditionalDiffusion


class ConditionalFlowMatching(GaussianConditionalDiffusion):
    """Rectified flow with the same public D-only contract as diffusion."""

    def __init__(self, *args, flow_cfg: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        flow_cfg = dict(flow_cfg or {})
        if not self.is_d_only:
            raise ValueError("ConditionalFlowMatching currently supports d_only targets only")
        if self.denoise_fn.branch_decoder:
            raise ValueError("ConditionalFlowMatching d_only forbids a branch decoder")
        self.flow_loss = str(flow_cfg.get("loss", "mse")).lower()
        if self.flow_loss not in {"mse", "huber"}:
            raise ValueError("flow_matching.loss must be mse or huber")
        self.huber_delta = float(flow_cfg.get("huber_delta", 1.0))
        self.time_eps = float(flow_cfg.get("time_eps", 1e-4))
        if not 0.0 <= self.time_eps < 0.5:
            raise ValueError("flow_matching.time_eps must be in [0, 0.5)")
        self.default_solver = str(flow_cfg.get("solver", "heun")).lower()
        self.default_steps = int(flow_cfg.get("steps", 30))
        self.is_flow_matching = True

    def forward(self, target, condition, side_labels=None, global_step=None):
        del global_step
        x0 = self.encode_target(target, condition)
        batch = x0.shape[0]
        t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
        t = t * (1.0 - 2.0 * self.time_eps) + self.time_eps
        t_image = t.view(batch, 1, 1, 1)
        x1 = torch.randn_like(x0)
        xt = (1.0 - t_image) * x0 + t_image * x1
        velocity_target = x1 - x0
        velocity_pred = self.denoise_fn(torch.cat((condition, xt), dim=1), t, side=side_labels)
        if self.flow_loss == "mse":
            loss = F.mse_loss(velocity_pred, velocity_target)
        else:
            loss = F.huber_loss(velocity_pred, velocity_target, delta=self.huber_delta)
        self.last_loss_breakdown = {
            "flow_matching_loss": float(loss.detach().item()),
            "total_loss": float(loss.detach().item()),
        }
        return loss

    def _velocity(self, x, t_value: float, condition, side_labels=None):
        t = torch.full((x.shape[0],), float(t_value), device=x.device, dtype=x.dtype)
        return self.denoise_fn(torch.cat((condition, x), dim=1), t, side=side_labels)

    @torch.no_grad()
    def sample_flow(self, condition, sample_steps=None, solver=None, side_labels=None):
        steps = int(self.default_steps if sample_steps is None else sample_steps)
        if steps <= 0:
            raise ValueError("flow-matching sample_steps must be positive")
        solver = str(self.default_solver if solver is None else solver).lower()
        if solver not in {"euler", "heun"}:
            raise ValueError("flow-matching solver must be euler or heun")
        x = torch.randn(condition.shape[0], self.target_channels, condition.shape[2], condition.shape[3], device=condition.device)
        # Integrate dx/dt=v(x,t) from t=1 (Gaussian) to t=0 (data).
        dt = -1.0 / steps
        for index in range(steps):
            t = 1.0 - index / steps
            velocity = self._velocity(x, t, condition, side_labels)
            if solver == "euler" or index == steps - 1:
                x = x + dt * velocity
            else:
                proposal = x + dt * velocity
                next_velocity = self._velocity(proposal, t + dt, condition, side_labels)
                x = x + dt * 0.5 * (velocity + next_velocity)
        return self.decode_target(x, condition)
