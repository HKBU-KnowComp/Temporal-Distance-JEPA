"""JEPA world model used by TD-JEPA (LeWM backbone + temporal-distance planning)."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        dist_head=None,
        reach_head=None,
        planning_cost_mode: str = "mse",
        planning_cost_temperature: float = 0.1,
        planning_cost_terminal_weight: float = 1.0,
        planning_cost_mse_blend: float = 0.0,
        planning_cost_contact_dist: float = 40.0,
        planning_cost_far_mse_blend: float = 0.0,
        planning_cost_near_mse_blend: float = 1.0,
        planning_cost_gate_temperature: float = 0.0,
        gaussian_predictor: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Optional quasimetric head for plan-aware (temporal-distance) cost.
        self.dist_head = dist_head
        # Optional RC-aux budget reachability classifier (non-directed).
        self.reach_head = reach_head
        self.gaussian_predictor = bool(gaussian_predictor)
        self.set_planning_cost(
            mode=planning_cost_mode,
            temperature=planning_cost_temperature,
            terminal_weight=planning_cost_terminal_weight,
            mse_blend=planning_cost_mse_blend,
            contact_dist=planning_cost_contact_dist,
            far_mse_blend=planning_cost_far_mse_blend,
            near_mse_blend=planning_cost_near_mse_blend,
            gate_temperature=planning_cost_gate_temperature,
        )

    def set_planning_cost(
        self,
        mode: str | None = None,
        temperature: float | None = None,
        terminal_weight: float | None = None,
        mse_blend: float | None = None,
        contact_dist: float | None = None,
        far_mse_blend: float | None = None,
        near_mse_blend: float | None = None,
        gate_temperature: float | None = None,
    ) -> None:
        """Configure inference-time planning cost used by ``get_cost``."""
        if mode is not None:
            mode = str(mode).lower()
            if mode == "gated":
                mode = "contact_gate"
            valid_modes = {
                "mse",
                "normalized_mse",
                "cosine",
                "softjepa_energy",
                "softjepa_l2_energy",
                "softjepa_gaussian_energy",
                "temporal_distance",
                "td_jepa",
                "contact_gate",
            }
            if mode not in valid_modes:
                raise ValueError(
                    f"Unknown planning cost mode '{mode}'. Valid: {sorted(valid_modes)}"
                )
            self.planning_cost_mode = mode

        if temperature is not None:
            temperature = float(temperature)
            if temperature <= 0:
                raise ValueError("planning_cost_temperature must be positive")
            self.planning_cost_temperature = temperature

        if terminal_weight is not None:
            terminal_weight = float(terminal_weight)
            if not (0.0 <= terminal_weight <= 1.0):
                raise ValueError("planning_cost_terminal_weight must be in [0, 1]")
            self.planning_cost_terminal_weight = terminal_weight

        if mse_blend is not None:
            mse_blend = float(mse_blend)
            if not (0.0 <= mse_blend <= 1.0):
                raise ValueError("planning_cost_mse_blend must be in [0, 1]")
            self.planning_cost_mse_blend = mse_blend

        if contact_dist is not None:
            contact_dist = float(contact_dist)
            if contact_dist <= 0:
                raise ValueError("planning_cost_contact_dist must be positive")
            self.planning_cost_contact_dist = contact_dist

        if far_mse_blend is not None:
            far_mse_blend = float(far_mse_blend)
            if not (0.0 <= far_mse_blend <= 1.0):
                raise ValueError("planning_cost_far_mse_blend must be in [0, 1]")
            self.planning_cost_far_mse_blend = far_mse_blend

        if near_mse_blend is not None:
            near_mse_blend = float(near_mse_blend)
            if not (0.0 <= near_mse_blend <= 1.0):
                raise ValueError("planning_cost_near_mse_blend must be in [0, 1]")
            self.planning_cost_near_mse_blend = near_mse_blend

        if gate_temperature is not None:
            gate_temperature = float(gate_temperature)
            if gate_temperature < 0:
                raise ValueError("planning_cost_gate_temperature must be >= 0")
            self.planning_cost_gate_temperature = gate_temperature

    @staticmethod
    def _as_bs_feat(x: torch.Tensor, n_feat: int) -> torch.Tensor:
        """Normalize env geometry tensors to shape ``(B, S, n_feat)``."""
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        # (B, S, ...)
        b, s = x.shape[0], x.shape[1]
        return x.reshape(b, s, -1)[..., :n_feat]

    def _contact_gate_alpha(self, info_dict: dict) -> torch.Tensor:
        """Per-(env, sample) ℓ₂ blend weight: far→d_ψ, near-contact→ℓ₂.

        Uses raw ``pos_agent`` / ``block_pose`` (px) and/or ``n_contacts``.
        Never uses StandardScaled ``state``.
        """
        far = float(getattr(self, "planning_cost_far_mse_blend", 0.0))
        near = float(getattr(self, "planning_cost_near_mse_blend", 1.0))
        thresh = float(getattr(self, "planning_cost_contact_dist", 40.0))
        tau = float(getattr(self, "planning_cost_gate_temperature", 0.0))

        ref = None
        for key in ("pixels", "emb", "action", "goal"):
            if key in info_dict and torch.is_tensor(info_dict[key]):
                ref = info_dict[key]
                break
        if ref is None:
            raise RuntimeError("contact_gate: no tensor in info_dict to infer batch shape")
        if ref.ndim == 1:
            b, s = 1, 1
        elif ref.ndim == 2:
            b, s = ref.shape[0], 1
        else:
            b, s = ref.shape[0], ref.shape[1]
        device = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32

        dist = None
        if "pos_agent" in info_dict and "block_pose" in info_dict:
            pa = self._as_bs_feat(info_dict["pos_agent"].to(device=device, dtype=dtype), 2)
            bp = self._as_bs_feat(info_dict["block_pose"].to(device=device, dtype=dtype), 2)
            dist = (pa - bp).pow(2).sum(dim=-1).sqrt()  # (B, S)

        contacts = None
        if "n_contacts" in info_dict and torch.is_tensor(info_dict["n_contacts"]):
            nc = info_dict["n_contacts"].to(device=device, dtype=dtype)
            contacts = self._as_bs_feat(nc, 1).squeeze(-1)  # (B, S)

        if dist is None and contacts is None:
            # Fail soft: stay on far (temporal) cost so eval still runs.
            if not getattr(self, "_contact_gate_warned", False):
                print(
                    "WARNING: contact_gate missing pos_agent/block_pose/n_contacts; "
                    f"using far_mse_blend={far}"
                )
                self._contact_gate_warned = True
            return torch.full((b, s), far, device=device, dtype=dtype)

        if dist is None:
            near_mask = contacts > 0
            alpha = torch.where(
                near_mask,
                torch.full(near_mask.shape, near, device=device, dtype=dtype),
                torch.full(near_mask.shape, far, device=device, dtype=dtype),
            )
            return alpha

        if tau > 0:
            # Soft: α = far + (near-far) * σ((thresh - dist) / τ)
            gate = torch.sigmoid((thresh - dist) / tau)
            alpha = far + (near - far) * gate
        else:
            alpha = torch.where(
                dist < thresh,
                torch.full_like(dist, near),
                torch.full_like(dist, far),
            )

        if contacts is not None:
            alpha = torch.where(contacts > 0, torch.full_like(alpha, near), alpha)
        return alpha

    def encode(self, info):
        """Encode observations and actions into embeddings."""
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def _project_prediction(self, preds, b: int):
        flat = rearrange(preds, "b t d -> (b t) d")
        if getattr(self, "gaussian_predictor", False):
            mu, log_var = self.pred_proj(flat)
            mu = rearrange(mu, "(b t) d -> b t d", b=b)
            log_var = rearrange(log_var, "(b t) d -> b t d", b=b)
            return mu, log_var
        out = self.pred_proj(flat)
        return rearrange(out, "(b t) d -> b t d", b=b)

    def predict(self, emb, act_emb, return_gaussian: bool = False):
        """Predict next-state embeddings (mean if Gaussian predictor)."""
        preds = self.predictor(emb, act_emb)
        projected = self._project_prediction(preds, emb.size(0))
        if getattr(self, "gaussian_predictor", False):
            mu, log_var = projected
            if return_gaussian:
                return mu, log_var
            return mu
        return projected

    def predict_gaussian(self, emb, act_emb):
        """Return (mean, log_var) for Gaussian SoftJEPA."""
        if not getattr(self, "gaussian_predictor", False):
            raise RuntimeError("predict_gaussian requires gaussian_predictor=True")
        return self.predict(emb, act_emb, return_gaussian=True)

    def distance(self, z_s, z_g):
        """Directed quasimetric latent distance d(z_s -> z_g)."""
        if self.dist_head is None:
            raise RuntimeError("distance() requires a quasimetric dist_head")
        return self.dist_head(z_s, z_g)

    def reachability_logits(self, z_s, z_g):
        """Budget-conditioned reachability logits from the RC-aux head."""
        if self.reach_head is None:
            raise RuntimeError("reachability_logits() requires a reach_head")
        return self.reach_head(z_s, z_g)

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Roll out the latent model given initial observations and actions."""
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        init = self.encode(init)
        emb = info["emb"] = init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        init = {k: detach_clone(v) for k, v in init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        history = history_size
        log_var_steps = []
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -history:]
            act_trunc = act_emb[:, -history:]
            if getattr(self, "gaussian_predictor", False):
                pred_emb, pred_log_var = self.predict_gaussian(emb_trunc, act_trunc)
                pred_emb = pred_emb[:, -1:]
                log_var_steps.append(pred_log_var[:, -1:])
            else:
                pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -history:]
        act_trunc = act_emb[:, -history:]
        if getattr(self, "gaussian_predictor", False):
            pred_emb, pred_log_var = self.predict_gaussian(emb_trunc, act_trunc)
            pred_emb = pred_emb[:, -1:]
            log_var_steps.append(pred_log_var[:, -1:])
        else:
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        if getattr(self, "gaussian_predictor", False):
            log_var_traj = torch.stack(log_var_steps, dim=1)
            info["predicted_log_var"] = rearrange(
                log_var_traj, "(b s) ... -> b s ...", b=B, s=S
            )
        return info

    def criterion(self, info_dict: dict):
        """Compute latent cost between predicted rollout and goal embedding."""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        predicted_log_var = info_dict.get("predicted_log_var")

        # Ensure goal has explicit sample and time dimensions: (B, S, T, D)
        if goal_emb.ndim == 2:  # (B, D)
            goal_emb = goal_emb.unsqueeze(1).unsqueeze(2)
        elif goal_emb.ndim == 3:  # (B, T, D)
            goal_emb = goal_emb.unsqueeze(1)
        goal_emb = goal_emb[..., -1:, :].expand(
            pred_emb.size(0), pred_emb.size(1), pred_emb.size(2), pred_emb.size(3)
        )

        mode = self.planning_cost_mode
        if mode == "mse":
            step_cost = (pred_emb - goal_emb.detach()).pow(2).sum(dim=-1)
        elif mode == "normalized_mse":
            pred_n = F.normalize(pred_emb, dim=-1)
            goal_n = F.normalize(goal_emb.detach(), dim=-1)
            step_cost = (pred_n - goal_n).pow(2).sum(dim=-1)
        elif mode == "cosine":
            pred_n = F.normalize(pred_emb, dim=-1)
            goal_n = F.normalize(goal_emb.detach(), dim=-1)
            step_cost = 1.0 - (pred_n * goal_n).sum(dim=-1)
        elif mode == "softjepa_energy":
            pred_n = F.normalize(pred_emb, dim=-1)
            goal_n = F.normalize(goal_emb.detach(), dim=-1)
            logits = (pred_n * goal_n).sum(dim=-1) / self.planning_cost_temperature
            step_cost = -logits
        elif mode == "softjepa_l2_energy":
            step_cost = (
                (pred_emb - goal_emb.detach()).pow(2).sum(dim=-1)
                / self.planning_cost_temperature
            )
        elif mode in ("temporal_distance", "td_jepa", "contact_gate"):
            if self.dist_head is None:
                raise RuntimeError(f"{mode} planning cost requires a dist_head")
            b, s, t, d = pred_emb.shape
            flat_pred = pred_emb.reshape(-1, d)
            flat_goal = goal_emb.detach().reshape(-1, d)
            step_cost = self.dist_head(flat_pred, flat_goal).reshape(b, s, t)
            mse_cost = (pred_emb - goal_emb.detach()).pow(2).sum(dim=-1)
            if mode == "contact_gate":
                alpha = info_dict.get("_contact_alpha")
                if alpha is None:
                    alpha = self._contact_gate_alpha(info_dict)
                if alpha.ndim == 1:
                    alpha = alpha.view(b, 1).expand(b, s)
                elif alpha.shape != (b, s):
                    alpha = alpha.reshape(b, s)
                alpha = alpha.to(device=step_cost.device, dtype=step_cost.dtype)
                step_cost = (1.0 - alpha.unsqueeze(-1)) * step_cost + alpha.unsqueeze(
                    -1
                ) * mse_cost
            else:
                alpha = getattr(self, "planning_cost_mse_blend", 0.0)
                if alpha > 0.0:
                    step_cost = (1.0 - alpha) * step_cost + alpha * mse_cost
        elif mode == "softjepa_gaussian_energy":
            if not getattr(self, "gaussian_predictor", False):
                raise RuntimeError(
                    "softjepa_gaussian_energy requires a Gaussian predictor"
                )
            if predicted_log_var is None:
                raise RuntimeError("predicted_log_var missing from rollout")
            log_var = predicted_log_var
            inv_var = torch.exp(-log_var)
            diff = goal_emb.detach() - pred_emb
            step_cost = 0.5 * (
                (diff.pow(2) * inv_var).sum(dim=-1) + log_var.sum(dim=-1)
            ) / self.planning_cost_temperature
        else:
            raise RuntimeError(f"Unexpected planning_cost_mode: {mode}")

        terminal = step_cost[..., -1]
        w = self.planning_cost_terminal_weight
        if w >= 1.0:
            return terminal
        mean_cost = step_cost.mean(dim=-1)
        if w <= 0.0:
            return mean_cost
        return w * terminal + (1.0 - w) * mean_cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Return planning cost for action candidates."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for key in list(info_dict.keys()):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for key in info_dict:
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        if self.planning_cost_mode == "contact_gate":
            info_dict["_contact_alpha"] = self._contact_gate_alpha(info_dict)
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
