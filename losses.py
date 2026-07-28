"""Losses and collapse diagnostics for TD-JEPA experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _build_temporal_false_negative_mask(
    batch_size: int,
    seq_len: int,
    exclusion_window: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Mask same-clip nearby timesteps as false negatives.

    Returns a boolean mask of shape (B*T, B*T) where True entries should be
    excluded from negative set. Diagonal is always False (positives).
    """
    if exclusion_window <= 0:
        return None

    sample_idx = torch.arange(batch_size, device=device).repeat_interleave(seq_len)
    time_idx = torch.arange(seq_len, device=device).repeat(batch_size)
    same_clip = sample_idx[:, None] == sample_idx[None, :]
    nearby_time = (time_idx[:, None] - time_idx[None, :]).abs() <= exclusion_window
    mask = same_clip & nearby_time
    mask.fill_diagonal_(False)
    return mask


def _apply_latent_transform(x: torch.Tensor, latent_norm: str) -> torch.Tensor:
    """Optional pre-energy transform to stabilize metric learning."""
    mode = str(latent_norm).lower()
    if mode in ("none", ""):
        return x
    if mode == "layer_norm":
        return F.layer_norm(x, (x.size(-1),))
    raise ValueError(f"Unknown latent_norm '{latent_norm}'. Use none or layer_norm.")


def gaussian_nll_energy(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood energy, shape (N, M)."""
    diff = target.unsqueeze(0) - mu.unsqueeze(1)
    inv_var = torch.exp(-log_var).unsqueeze(1)
    mahal = (diff.pow(2) * inv_var).sum(dim=-1)
    log_det = log_var.sum(dim=-1).unsqueeze(1).expand_as(mahal)
    return 0.5 * (mahal + log_det)


def _compute_logits(
    pred: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    energy: str,
    pred_log_var: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return softmax logits for the configured SoftJEPA energy."""
    if energy in ("dot", "dot_product", "cosine"):
        return pred @ candidates.T / temperature
    if energy in ("negative_squared_l2", "squared_l2", "l2"):
        distances = torch.cdist(pred.float(), candidates.float()).pow(2)
        return -distances.to(dtype=pred.dtype) / temperature
    if energy in ("gaussian_nll", "gaussian"):
        if pred_log_var is None:
            raise ValueError("gaussian_nll energy requires pred_log_var")
        energy_matrix = gaussian_nll_energy(pred, pred_log_var, candidates)
        return -energy_matrix / temperature
    raise ValueError(
        f"Unknown SoftJEPA energy '{energy}'. "
        "Valid options: dot_product, negative_squared_l2, gaussian_nll"
    )


def softjepa_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    temperature: float = 0.1,
    label_smoothing: float = 0.1,
    normalize: bool = True,
    latent_norm: str = "none",
    compute_metrics: bool = True,
    queue_negatives: torch.Tensor | None = None,
    temporal_exclusion_window: int = 0,
    energy: str = "dot_product",
    pred_log_var: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute in-batch softmax loss over predicted and target latents.

    Args:
        pred_emb: Predicted latents with shape ``(B, T, D)``.
        target_emb: Stop-gradient target latents with shape ``(B, T, D)``.
        temperature: Softmax temperature.
        label_smoothing: Smoothing mass distributed across all in-batch targets.
        normalize: Whether to L2-normalize latents before energy computation.
        compute_metrics: Whether to compute additional no-grad diagnostics.
        queue_negatives: Optional queued negatives with shape ``(Q, D)``.
        temporal_exclusion_window: Exclude negatives in same clip with
            ``|delta_t| <= window``.
        energy: Logit energy: ``dot_product`` or ``negative_squared_l2``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    energy = str(energy).lower()
    pred = pred_emb.reshape(-1, pred_emb.size(-1))
    target = target_emb.detach().reshape(-1, target_emb.size(-1))
    log_var = None
    if pred_log_var is not None:
        log_var = pred_log_var.reshape(-1, pred_log_var.size(-1))

    if energy not in ("gaussian_nll", "gaussian"):
        pred = _apply_latent_transform(pred, latent_norm)
        target = _apply_latent_transform(target, latent_norm)
        if normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)

    inbatch_logits = _compute_logits(
        pred=pred,
        candidates=target,
        temperature=temperature,
        energy=energy,
        pred_log_var=log_var,
    )
    batch = inbatch_logits.size(0)
    labels = torch.arange(batch, device=inbatch_logits.device)

    temporal_mask = _build_temporal_false_negative_mask(
        batch_size=pred_emb.size(0),
        seq_len=pred_emb.size(1),
        exclusion_window=int(temporal_exclusion_window),
        device=inbatch_logits.device,
    )
    if temporal_mask is not None:
        inbatch_logits = inbatch_logits.masked_fill(temporal_mask, float("-inf"))

    queue_logits = None
    if queue_negatives is not None and queue_negatives.numel() > 0:
        queue = queue_negatives.detach().to(
            device=pred.device, dtype=pred.dtype, non_blocking=True
        )
        if energy not in ("gaussian_nll", "gaussian"):
            queue = _apply_latent_transform(queue, latent_norm)
            if normalize:
                queue = F.normalize(queue, dim=-1)
        queue_logits = _compute_logits(
            pred=pred,
            candidates=queue,
            temperature=temperature,
            energy=energy,
            pred_log_var=log_var,
        )
        logits = torch.cat([inbatch_logits, queue_logits], dim=1)
    else:
        logits = inbatch_logits

    if label_smoothing > 0:
        valid = torch.isfinite(logits)
        valid_count = valid.sum(dim=1)
        denom = (valid_count - 1).clamp_min(1).to(logits.dtype)
        off_value = (label_smoothing / denom).unsqueeze(1)
        target_probs = torch.where(valid, off_value, torch.zeros_like(logits))
        target_probs.scatter_(1, labels[:, None], 1.0 - label_smoothing)
        only_positive = valid_count <= 1
        if only_positive.any():
            target_probs[only_positive] = 0.0
            target_probs[only_positive, labels[only_positive]] = 1.0
        log_probs = F.log_softmax(logits, dim=1)
        weighted_log_probs = torch.where(
            target_probs > 0, target_probs * log_probs, torch.zeros_like(log_probs)
        )
        loss = -weighted_log_probs.sum(dim=1).mean()
    else:
        loss = F.cross_entropy(logits, labels)

    metrics = {}
    if compute_metrics:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean()
            acc = (logits.argmax(dim=1) == labels).float().mean()
            pos_logits = inbatch_logits.diag().mean()

            neg_terms = []
            inbatch_neg_mask = ~torch.eye(
                batch, dtype=torch.bool, device=inbatch_logits.device
            )
            if temporal_mask is not None:
                inbatch_neg_mask = inbatch_neg_mask & (~temporal_mask)
            if inbatch_neg_mask.any():
                neg_terms.append(inbatch_logits[inbatch_neg_mask])
            if queue_logits is not None and queue_logits.numel() > 0:
                neg_terms.append(queue_logits.reshape(-1))
            if neg_terms:
                neg_logits = torch.cat(neg_terms, dim=0).mean()
            else:
                neg_logits = pos_logits.new_zeros(())

        metrics = {
            "softjepa_entropy": entropy,
            "softjepa_acc": acc,
            "softjepa_pos_logit": pos_logits,
            "softjepa_neg_logit": neg_logits,
        }
        metrics["softjepa_energy_negative_squared_l2"] = pos_logits.new_tensor(
            float(energy in ("negative_squared_l2", "squared_l2", "l2"))
        )
        metrics["softjepa_energy_gaussian_nll"] = pos_logits.new_tensor(
            float(energy in ("gaussian_nll", "gaussian"))
        )
        if queue_negatives is not None:
            metrics["softjepa_queue_size"] = pos_logits.new_tensor(
                float(queue_negatives.shape[0])
            )
    return loss, metrics


def latent_diagnostics(emb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return lightweight collapse diagnostics for a latent batch."""

    with torch.amp.autocast(device_type=emb.device.type, enabled=False):
        z = emb.detach().reshape(-1, emb.size(-1)).float()
        z = z - z.mean(dim=0, keepdim=True)
        denom = max(z.size(0) - 1, 1)
        cov = z.T @ z / denom
        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
        total = eigvals.sum().clamp_min(1e-12)
        participation_rank = total.square() / eigvals.square().sum().clamp_min(1e-12)

        z_norm = F.normalize(z, dim=-1)
        cosine = z_norm @ z_norm.T
        if cosine.size(0) > 1:
            mask = ~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)
            mean_pairwise_cosine = cosine[mask].mean()
        else:
            mean_pairwise_cosine = cosine.new_tensor(1.0)

    return {
        "latent_rank": participation_rank,
        "latent_mean_pairwise_cosine": mean_pairwise_cosine,
        "latent_std": z.std(dim=0).mean(),
    }


def sample_wrong_actions(
    actions: torch.Tensor,
    num_wrong: int,
    noise_std: float = 0.3,
) -> list[torch.Tensor]:
    """Sample wrong actions for counterfactual contrastive negatives."""
    if num_wrong <= 0:
        raise ValueError("num_wrong must be positive")
    wrong: list[torch.Tensor] = []
    for k in range(num_wrong):
        if k % 2 == 0:
            wrong.append(actions.roll(shifts=k + 1, dims=0))
        else:
            wrong.append(actions + noise_std * torch.randn_like(actions))
    return wrong


def softjepa_wrong_action_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    wrong_pred_emb: torch.Tensor,
    temperature: float = 1.0,
    label_smoothing: float = 0.1,
    normalize: bool = False,
    latent_norm: str = "layer_norm",
    energy: str = "negative_squared_l2",
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Counterfactual contrastive over action-conditioned predictions.

    The anchor is the true next latent ``z_{t+1} = sg(Enc(o_{t+1}))`` -- exactly the
    goal latent that latent MPC scores candidate rollouts against. The candidate set
    is ``{Pred(s, a), Pred(s, a'_1), ..., Pred(s, a'_K)}`` and the positive is the
    prediction under the executed action ``a``. Training therefore teaches: among
    candidate actions at the same state, the executed action's prediction must be the
    closest to the realized next latent. This matches ``JEPA.criterion`` at plan time.

    Args:
        pred_emb: Executed-action prediction ``Pred(s, a)`` with shape ``(B, T, D)``.
        target_emb: Stop-gradient true next latent ``z_{t+1}`` with shape ``(B, T, D)``.
        wrong_pred_emb: Wrong-action predictions with shape ``(B, K, T, D)``.
        normalize: L2-normalize latents before the energy (cosine-style stability).
        latent_norm: Optional pre-energy transform (``none`` or ``layer_norm``).
        energy: Logit energy: ``dot_product`` or ``negative_squared_l2``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    energy = str(energy).lower()
    pred = pred_emb.reshape(-1, pred_emb.size(-1))
    target = target_emb.detach().reshape(-1, target_emb.size(-1))
    wrong = wrong_pred_emb.reshape(
        wrong_pred_emb.size(0), -1, wrong_pred_emb.size(-1)
    )
    if energy not in ("gaussian_nll", "gaussian"):
        pred = _apply_latent_transform(pred, latent_norm)
        target = _apply_latent_transform(target, latent_norm)
        wrong = _apply_latent_transform(wrong, latent_norm)
        if normalize:
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)
            wrong = F.normalize(wrong, dim=-1)
    batch, num_wrong, _ = wrong.shape
    # Anchor on the true next latent; candidates are action-conditioned predictions.
    pos_logits = (
        _compute_logits(target, pred, temperature, energy).diag().unsqueeze(1)
    )
    neg_terms = [
        _compute_logits(target, wrong[:, k, :], temperature, energy)
        .diag()
        .unsqueeze(1)
        for k in range(num_wrong)
    ]
    neg_logits = torch.cat(neg_terms, dim=1)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(batch, dtype=torch.long, device=logits.device)
    if label_smoothing > 0:
        num_classes = logits.size(1)
        off_value = label_smoothing / max(num_classes - 1, 1)
        target_probs = torch.full_like(logits, off_value)
        target_probs.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
        loss = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(logits, labels)
    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        with torch.no_grad():
            metrics = {
                "softjepa_acc": (logits.argmax(1) == labels).float().mean(),
                "softjepa_pos_logit": pos_logits.mean(),
                "softjepa_neg_logit": neg_logits.mean(),
                "softjepa_cf_margin": (
                    pos_logits - neg_logits.max(dim=1, keepdim=True).values
                ).mean(),
            }
    return loss, metrics


def mse_alignment_loss(pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    """Plain next-latent MSE used by LeWM and hybrid SoftJEPA variants."""
    return (pred_emb - target_emb.detach()).pow(2).mean()


def rollout_mse_loss(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    horizon: int,
) -> torch.Tensor:
    """Multi-step open-loop MSE on expert actions (planning-relevant signal)."""
    if horizon <= 0:
        raise ValueError("rollout horizon must be positive")

    state_emb = emb[:, :history_size]
    state_act = act_emb[:, :history_size]
    losses = []
    for step in range(horizon):
        pred = model.predict(state_emb, state_act)[:, -1:]
        target = emb[:, history_size + step : history_size + step + 1]
        losses.append((pred - target.detach()).pow(2).mean())
        next_act = act_emb[:, history_size + step : history_size + step + 1]
        state_emb = torch.cat([state_emb[:, 1:], pred], dim=1)
        state_act = torch.cat([state_act[:, 1:], next_act], dim=1)
    return torch.stack(losses).mean()


def multi_horizon_rollout_mse_loss(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    horizons,
    n_preds: int,
) -> torch.Tensor:
    """Average open-loop MSE over several rollout horizons (RC-aux style)."""
    hs = [int(h) for h in horizons]
    if not hs:
        raise ValueError("rollout_horizons must be a non-empty list")
    capped = []
    for h in hs:
        if h <= 0:
            raise ValueError(f"rollout horizon must be positive, got {h}")
        capped.append(min(h, int(n_preds)))
    losses = [
        rollout_mse_loss(
            model,
            emb,
            act_emb,
            history_size=history_size,
            horizon=h,
        )
        for h in capped
    ]
    return torch.stack(losses).mean()


def budget_reachability_loss(
    model,
    emb: torch.Tensor,
    plan_horizon: int,
    hard_neg_weight: float = 1.0,
    cross_neg_weight: float = 1.0,
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RC-aux budget-conditioned reachability with temporal hard negatives.

    Positives: same-trajectory pairs with ``j - i <= H``.
    Temporal hard negatives: same-trajectory pairs with ``j - i > H``.
    Cross-trajectory goals are additional negatives (label 0).
    """
    if emb.dim() != 3:
        raise ValueError("budget_reachability_loss expects emb of shape (B, T, D)")
    if plan_horizon <= 0:
        raise ValueError("plan_horizon must be positive")
    if getattr(model, "reach_head", None) is None:
        raise RuntimeError("budget_reachability_loss requires model.reach_head")

    batch, seq_len, _ = emb.shape
    if seq_len < 2:
        raise ValueError("budget_reachability_loss needs sequence length >= 2")
    device = emb.device
    rows = torch.arange(batch, device=device)
    H = float(plan_horizon)

    # Sample a mixed pool of same-traj pairs (near and far).
    i = torch.randint(0, seq_len - 1, (batch,), device=device)
    offset = torch.randint(1, seq_len, (batch,), device=device)
    j = torch.minimum(i + offset, torch.full_like(i, seq_len - 1))
    gap = (j - i).float()
    z_s = emb[rows, i]
    z_g = emb[rows, j]
    logits = model.reachability_logits(z_s, z_g)
    reachable = (gap <= H).float()
    pos_hard_loss = F.binary_cross_entropy_with_logits(logits, reachable)

    # Explicit temporal hard-negative pass: force far same-traj pairs when possible.
    hard_loss = logits.new_zeros(())
    max_gap = seq_len - 1
    if max_gap > plan_horizon:
        i_hard = torch.randint(0, seq_len - plan_horizon - 1, (batch,), device=device)
        # Gaps strictly beyond the planner budget.
        far_offset = torch.randint(
            plan_horizon + 1, max_gap + 1, (batch,), device=device
        )
        j_hard = torch.minimum(i_hard + far_offset, torch.full_like(i_hard, seq_len - 1))
        # Keep only pairs that remain beyond H after clamping.
        keep = (j_hard - i_hard) > plan_horizon
        if keep.any():
            z_s_h = emb[rows[keep], i_hard[keep]]
            z_g_h = emb[rows[keep], j_hard[keep]]
            logits_h = model.reachability_logits(z_s_h, z_g_h)
            hard_loss = F.binary_cross_entropy_with_logits(
                logits_h, torch.zeros_like(logits_h)
            )

    # Cross-trajectory negatives.
    perm = torch.randperm(batch, device=device)
    clash = perm == rows
    if clash.any():
        perm[clash] = (perm[clash] + 1) % batch
    z_g_neg = emb[perm, j]
    logits_neg = model.reachability_logits(z_s, z_g_neg)
    cross_loss = F.binary_cross_entropy_with_logits(
        logits_neg, torch.zeros_like(logits_neg)
    )

    loss = (
        pos_hard_loss
        + float(hard_neg_weight) * hard_loss
        + float(cross_neg_weight) * cross_loss
    )

    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        with torch.no_grad():
            metrics = {
                "reach_loss": loss,
                "reach_pos_hard_loss": pos_hard_loss,
                "reach_temporal_hard_loss": hard_loss,
                "reach_cross_loss": cross_loss,
                "reach_pos_frac": reachable.mean(),
                "reach_acc": ((logits > 0).float() == reachable).float().mean(),
                "reach_cross_neg_rate": (logits_neg <= 0).float().mean(),
            }
    return loss, metrics


def temporal_distance_loss(
    model,
    emb: torch.Tensor,
    negative_margin_scale: float = 1.0,
    negative_weight: float = 1.0,
    negative_margin_steps: int | None = None,
    plan_horizon: int = 0,
    budget_td_weight: float = 0.0,
    sym_l2_weight: float = 0.0,
    target_transform: str = "raw",
    near_goal_weight_steps: int = 0,
    near_goal_weight: float = 1.0,
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plan-aware grounding: regress the quasimetric latent distance to the
    temporal gap between a state and a sampled future state (hindsight).

    For each sequence in the batch we sample a state index ``i`` and a future
    index ``j > i`` and train ``d(z_i -> z_j) ~= (j - i)``. Goals drawn from a
    different trajectory are treated as heuristic negatives, so their distance
    is pushed beyond the in-window horizon via a weighted hinge. This makes the
    latent distance monotonic in steps-to-goal -- the quantity latent MPC needs.

    ``target_transform='log1p'`` regresses to log-scaled temporal gaps, giving
    small gaps more resolution without changing the sampled pairs. Optional
    ``near_goal_weight_steps`` and ``near_goal_weight`` upweight small raw gaps.
    Optional ``plan_horizon`` + ``budget_td_weight`` add horizon-budget
    reachability supervision (RC-aux style): classify whether ``j - i`` is
    within the planner rollout horizon. ``sym_l2_weight`` encourages the
    symmetric distance component to track latent L2 geometry (LeWM-style).
    """
    if emb.dim() != 3:
        raise ValueError("temporal_distance_loss expects emb of shape (B, T, D)")
    batch, seq_len, _ = emb.shape
    if seq_len < 2:
        raise ValueError("temporal_distance_loss needs sequence length >= 2")
    device = emb.device
    rows = torch.arange(batch, device=device)
    i = torch.randint(0, seq_len - 1, (batch,), device=device)
    offset = torch.randint(1, seq_len, (batch,), device=device)
    j = torch.minimum(i + offset, torch.full_like(i, seq_len - 1))
    raw_target = (j - i).float()
    if target_transform == "raw":
        target = raw_target
    elif target_transform == "log1p":
        target = torch.log1p(raw_target)
    else:
        raise ValueError(f"Unknown temporal-distance target transform: {target_transform}")

    z_s = emb[rows, i]
    z_g = emb[rows, j]
    d_pos = model.distance(z_s, z_g)
    per_pair_pos_loss = F.smooth_l1_loss(d_pos, target, reduction="none")
    if near_goal_weight_steps > 0 and near_goal_weight != 1.0:
        weights = torch.ones_like(per_pair_pos_loss)
        weights = torch.where(
            raw_target <= float(near_goal_weight_steps),
            weights * float(near_goal_weight),
            weights,
        )
        pos_loss = (per_pair_pos_loss * weights).sum() / weights.sum().clamp_min(1e-12)
    else:
        pos_loss = per_pair_pos_loss.mean()

    # Cross-trajectory goals are heuristic negatives -> distance >= horizon.
    perm = torch.randperm(batch, device=device)
    clash = perm == rows
    if clash.any():
        perm[clash] = (perm[clash] + 1) % batch
    z_g_neg = emb[perm, j]
    d_neg = model.distance(z_s, z_g_neg)
    # Default margin scales with the loaded window (seq_len-1). Long-window v2
    # configs (num_preds=28) pushed this to ~30 and over-penalized cross-trajectory
    # pairs, collapsing distance calibration. Use td_negative_margin_steps to
    # decouple from window length (short-window baseline: seq_len=8 -> margin=7).
    margin_steps = float(
        seq_len - 1 if negative_margin_steps is None else negative_margin_steps
    )
    margin = negative_margin_scale * margin_steps
    neg_loss_raw = torch.relu(margin - d_neg).mean()
    neg_loss = float(negative_weight) * neg_loss_raw

    loss = pos_loss + neg_loss
    budget_loss = d_pos.new_zeros(())
    sym_l2_loss = d_pos.new_zeros(())
    if budget_td_weight > 0 and plan_horizon > 0:
        reachable = (raw_target <= float(plan_horizon)).float()
        budget_logits = float(plan_horizon) - d_pos
        budget_loss = F.binary_cross_entropy_with_logits(budget_logits, reachable)
        loss = loss + budget_td_weight * budget_loss

    dist_head = getattr(model, "dist_head", None)
    if sym_l2_weight > 0 and dist_head is not None and hasattr(dist_head, "forward_parts"):
        sym, _ = dist_head.forward_parts(z_s, z_g)
        l2 = (z_s - z_g).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        sym_l2_loss = F.smooth_l1_loss(sym, l2.detach())
        loss = loss + sym_l2_weight * sym_l2_loss

    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        with torch.no_grad():
            metrics = {
                "td_pos_loss": pos_loss,
                "td_neg_loss": neg_loss,
                "td_neg_loss_raw": neg_loss_raw,
                "td_neg_weight": d_pos.new_tensor(float(negative_weight)),
                "td_pred_dist_mean": d_pos.mean(),
                "td_target_mean": target.mean(),
                "td_raw_target_mean": raw_target.mean(),
                "td_neg_dist_mean": d_neg.mean(),
            }
            if target_transform != "raw":
                metrics["td_target_transform_log1p"] = d_pos.new_tensor(1.0)
            if near_goal_weight_steps > 0 and near_goal_weight != 1.0:
                metrics["td_neargoal_frac"] = (
                    raw_target <= float(near_goal_weight_steps)
                ).float().mean()
            if budget_td_weight > 0 and plan_horizon > 0:
                metrics["td_budget_loss"] = budget_loss
                metrics["td_budget_acc"] = (
                    (budget_logits > 0).float() == reachable
                ).float().mean()
            if sym_l2_weight > 0:
                metrics["td_sym_l2_loss"] = sym_l2_loss
    return loss, metrics


def compute_prediction_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    loss_cfg,
    compute_metrics: bool = True,
    queue_negatives: torch.Tensor | None = None,
    temporal_exclusion_window: int = 0,
    pred_log_var: torch.Tensor | None = None,
    wrong_pred_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch configured prediction loss."""

    name = str(loss_cfg.prediction.name).lower()
    if name in ("contrastive", "softjepa"):
        name = "softjepa"
    if name in ("contrastive_wrong_action", "softjepa_wrong_action"):
        name = "softjepa_wrong_action"
    if name == "mse":
        return mse_alignment_loss(pred_emb, target_emb), {}
    if name == "softjepa":
        return softjepa_loss(
            pred_emb=pred_emb,
            target_emb=target_emb,
            temperature=float(loss_cfg.prediction.temperature),
            label_smoothing=float(loss_cfg.prediction.label_smoothing),
            normalize=bool(loss_cfg.prediction.normalize),
            latent_norm=str(loss_cfg.prediction.get("latent_norm", "none")),
            compute_metrics=compute_metrics,
            queue_negatives=queue_negatives,
            temporal_exclusion_window=temporal_exclusion_window,
            energy=str(loss_cfg.prediction.get("energy", "dot_product")),
            pred_log_var=pred_log_var,
        )
    if name == "softjepa_wrong_action":
        if wrong_pred_emb is None:
            raise ValueError("wrong_pred_emb required for softjepa_wrong_action")
        return softjepa_wrong_action_loss(
            pred_emb,
            target_emb,
            wrong_pred_emb,
            temperature=float(loss_cfg.prediction.get("temperature", 1.0)),
            label_smoothing=float(loss_cfg.prediction.get("label_smoothing", 0.1)),
            normalize=bool(loss_cfg.prediction.get("normalize", False)),
            latent_norm=str(loss_cfg.prediction.get("latent_norm", "layer_norm")),
            energy=str(loss_cfg.prediction.get("energy", "negative_squared_l2")),
            compute_metrics=compute_metrics,
        )
    raise ValueError(f"Unknown prediction loss: {loss_cfg.prediction.name}")
