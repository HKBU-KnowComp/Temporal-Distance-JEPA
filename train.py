import logging
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from compat_dm_control import apply_dm_control_mujoco_compat

apply_dm_control_mujoco_compat()

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict


def configure_training_runtime() -> None:
    """Reduce SLURM log volume and keep Lightning artifacts off home quota."""
    if os.environ.get("SLURM_JOB_ID"):
        spt.set(
            default_callbacks={
                "logging": False,
                "env_dump": False,
                "module_summary": False,
                "slurm_info": False,
                "trainer_info": False,
            }
        )
        for name in (
            "lightning.pytorch",
            "lightning.fabric",
            "pytorch_lightning",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

from losses import (
    budget_reachability_loss,
    compute_prediction_loss,
    latent_diagnostics,
    mse_alignment_loss,
    multi_horizon_rollout_mse_loss,
    rollout_mse_loss,
    sample_wrong_actions,
    temporal_distance_loss,
)
from module import SIGReg
from utils import PlanningEvalCallback, SaveCkptCallback, get_column_normalizer, get_img_preprocessor


def configure_wandb_logger(cfg):
    if not cfg.wandb.enabled:
        return None

    wandb_kwargs = OmegaConf.to_container(cfg.wandb.config, resolve=True)
    tags = list(wandb_kwargs.pop("tags", None) or [])
    tags.append(f"pred:{cfg.loss.prediction.name}")
    if float(cfg.loss.sigreg.weight) > 0:
        tags.append("sigreg")
    if slurm_job_id := os.environ.get("SLURM_JOB_ID"):
        tags.append(f"slurm:{slurm_job_id}")
    wandb_kwargs["tags"] = tags
    if wandb_kwargs.get("group") in (None, "null"):
        wandb_kwargs.pop("group", None)
    if wandb_kwargs.get("entity") in (None, "null"):
        wandb_kwargs.pop("entity", None)

    save_dir = Path(os.environ.get("WANDB_DIR", str(Path(__file__).resolve().parent)))
    if save_dir.name == "wandb":
        save_dir = save_dir.parent
    wandb_kwargs.setdefault("save_dir", str(save_dir))
    logger = WandbLogger(**wandb_kwargs)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    return logger


def should_compute_diagnostics(module, stage: str, cfg) -> bool:
    """Keep expensive diagnostics off the critical path for most train steps."""

    if stage not in ("train", "fit"):
        return True

    interval = int(cfg.get("diagnostics_interval", 100))
    if interval <= 0:
        return False

    return int(getattr(module, "global_step", 0)) % interval == 0


def jepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, and compute configured losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    # Next-frame targets for the context window (one-step prediction). This is
    # equivalent to emb[:, n_preds:] when num_preds=1, but stays aligned with the
    # predictor's per-position outputs when num_preds>1 (longer windows loaded for
    # multi-step rollout / temporal-distance grounding).
    tgt_emb = emb[:, 1 : ctx_len + 1]

    compute_diagnostics = should_compute_diagnostics(self, stage, cfg)
    prediction_cfg = cfg.loss.prediction
    prediction_name = str(prediction_cfg.get("name", ""))
    queue_size = int(prediction_cfg.get("queue_size", 0))
    temporal_exclusion_window = int(
        prediction_cfg.get("temporal_exclusion_window", 0)
    )
    energy = str(prediction_cfg.get("energy", "dot_product")).lower()
    gaussian_mode = energy in ("gaussian_nll", "gaussian")

    pred_log_var = None
    if gaussian_mode:
        pred_emb, pred_log_var = self.model.predict_gaussian(ctx_emb, ctx_act)
    else:
        pred_emb = self.model.predict(ctx_emb, ctx_act)

    queue_negatives = None
    if queue_size > 0 and stage in ("train", "fit"):
        queue = getattr(self, "_softjepa_queue", None)
        if queue is not None and queue.numel() > 0:
            queue_negatives = queue.to(device=pred_emb.device, dtype=pred_emb.dtype)

    wrong_pred_emb = None
    if prediction_name == "softjepa_wrong_action":
        num_wrong = int(prediction_cfg.get("num_wrong_actions", 4))
        noise_std = float(prediction_cfg.get("wrong_action_noise_std", 0.3))
        expert_act = act_emb[:, ctx_len - 1 : ctx_len, :]
        expert_flat = expert_act.reshape(expert_act.size(0), -1)
        wrong_actions = sample_wrong_actions(expert_flat, num_wrong, noise_std)
        state_emb = emb[:, :ctx_len]
        state_act = act_emb[:, :ctx_len]
        wrong_preds = []
        for a_wrong in wrong_actions:
            wrong_act_seq = state_act.clone()
            wrong_act_seq[:, -1:, :] = a_wrong.view(expert_act.shape)
            wrong_preds.append(self.model.predict(state_emb, wrong_act_seq)[:, -1:])
        wrong_pred_emb = torch.stack(wrong_preds, dim=1)

    loss_pred_emb = pred_emb
    loss_tgt_emb = tgt_emb
    if prediction_name == "softjepa_wrong_action":
        loss_pred_emb = pred_emb[:, -1:]
        # Counterfactual target is the immediate next latent after the context
        # window, independent of num_preds. This lets num_preds be raised to load
        # extra future frames for the multi-step rollout loss without shifting the
        # one-step CF target. For num_preds=1 this equals tgt_emb[:, -1:].
        loss_tgt_emb = emb[:, ctx_len : ctx_len + 1]

    pred_loss, pred_metrics = compute_prediction_loss(
        loss_pred_emb,
        loss_tgt_emb,
        cfg.loss,
        compute_metrics=compute_diagnostics,
        queue_negatives=queue_negatives,
        temporal_exclusion_window=temporal_exclusion_window,
        pred_log_var=pred_log_var,
        wrong_pred_emb=wrong_pred_emb,
    )

    if (
        queue_size > 0
        and prediction_name != "softjepa_wrong_action"
        and stage in ("train", "fit")
        and not gaussian_mode
    ):
        queue_entries = tgt_emb.detach().reshape(-1, tgt_emb.size(-1))
        if bool(prediction_cfg.get("normalize", True)):
            queue_entries = F.normalize(queue_entries, dim=-1)
        elif str(prediction_cfg.get("latent_norm", "none")).lower() == "layer_norm":
            queue_entries = F.layer_norm(queue_entries, (queue_entries.size(-1),))

        prev_queue = getattr(self, "_softjepa_queue", None)
        if prev_queue is None or prev_queue.numel() == 0:
            new_queue = queue_entries[-queue_size:]
        else:
            prev_queue = prev_queue.to(
                device=queue_entries.device, dtype=queue_entries.dtype
            )
            new_queue = torch.cat([prev_queue, queue_entries], dim=0)
            if new_queue.size(0) > queue_size:
                new_queue = new_queue[-queue_size:]
        self._softjepa_queue = new_queue.detach()

    output["pred_loss"] = pred_loss
    total_loss = pred_loss

    mse_weight = float(prediction_cfg.get("mse_weight", 0.0))
    if mse_weight > 0:
        mse_loss = mse_alignment_loss(
            loss_pred_emb if prediction_name == "softjepa_wrong_action" else pred_emb,
            loss_tgt_emb if prediction_name == "softjepa_wrong_action" else tgt_emb,
        )
        output["mse_aux_loss"] = mse_loss
        total_loss = total_loss + mse_weight * mse_loss

    rollout_horizon = int(prediction_cfg.get("rollout_horizon", 0))
    rollout_weight = float(prediction_cfg.get("rollout_weight", 0.0))
    rollout_horizons = prediction_cfg.get("rollout_horizons", None)
    if rollout_weight > 0:
        if rollout_horizons is not None:
            rollout_loss = multi_horizon_rollout_mse_loss(
                self.model,
                emb,
                act_emb,
                history_size=ctx_len,
                horizons=list(rollout_horizons),
                n_preds=n_preds,
            )
        elif rollout_horizon > 0:
            rollout_loss = rollout_mse_loss(
                self.model,
                emb,
                act_emb,
                history_size=ctx_len,
                horizon=min(rollout_horizon, n_preds),
            )
        else:
            rollout_loss = None
        if rollout_loss is not None:
            output["rollout_loss"] = rollout_loss
            total_loss = total_loss + rollout_weight * rollout_loss

    td_weight = float(prediction_cfg.get("td_weight", 0.0))
    td_metrics = {}
    if td_weight > 0 and getattr(self.model, "dist_head", None) is not None:
        margin_steps = prediction_cfg.get("td_negative_margin_steps", None)
        td_loss, td_metrics = temporal_distance_loss(
            self.model,
            emb,
            negative_margin_scale=float(
                prediction_cfg.get("td_negative_margin_scale", 1.0)
            ),
            negative_weight=float(prediction_cfg.get("td_negative_weight", 1.0)),
            negative_margin_steps=(
                None if margin_steps is None else int(margin_steps)
            ),
            plan_horizon=int(prediction_cfg.get("td_plan_horizon", 0)),
            budget_td_weight=float(prediction_cfg.get("budget_td_weight", 0.0)),
            sym_l2_weight=float(prediction_cfg.get("sym_l2_weight", 0.0)),
            target_transform=str(prediction_cfg.get("td_target_transform", "raw")),
            near_goal_weight_steps=int(
                prediction_cfg.get("td_neargoal_weight_steps", 0)
            ),
            near_goal_weight=float(prediction_cfg.get("td_neargoal_weight", 1.0)),
            compute_metrics=compute_diagnostics,
        )
        output["td_loss"] = td_loss
        total_loss = total_loss + td_weight * td_loss

    reach_metrics = {}
    budget_reach_weight = float(prediction_cfg.get("budget_reach_weight", 0.0))
    if budget_reach_weight > 0 and getattr(self.model, "reach_head", None) is not None:
        reach_loss, reach_metrics = budget_reachability_loss(
            self.model,
            emb,
            plan_horizon=int(prediction_cfg.get("td_plan_horizon", 5)),
            hard_neg_weight=float(
                prediction_cfg.get("budget_reach_hard_neg_weight", 1.0)
            ),
            cross_neg_weight=float(
                prediction_cfg.get("budget_reach_cross_neg_weight", 1.0)
            ),
            compute_metrics=compute_diagnostics,
        )
        output["reach_loss"] = reach_loss
        total_loss = total_loss + budget_reach_weight * reach_loss

    sigreg_weight = float(cfg.loss.sigreg.weight)
    if sigreg_weight > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    else:
        output["sigreg_loss"] = pred_loss.new_zeros(())
    total_loss = total_loss + sigreg_weight * output["sigreg_loss"]

    output["loss"] = total_loss

    metrics = {}
    metrics.update(pred_metrics)
    metrics.update(td_metrics)
    metrics.update(reach_metrics)
    if compute_diagnostics:
        metrics.update(latent_diagnostics(tgt_emb))
    for key, value in metrics.items():
        output[key] = value

    log_dict = {}
    for key, value in output.items():
        if not (key.endswith("loss") or key in metrics):
            continue
        log_dict[f"{stage}/{key}"] = value.detach()

    self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=False)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    configure_training_runtime()

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    init_from_policy = cfg.get("init_from_policy")
    if init_from_policy:
        world_model = swm.wm.utils.load_pretrained(str(init_from_policy))
        logging.getLogger(__name__).info(
            "Initialized world model from policy checkpoint: %s", init_from_policy
        )
    else:
        world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(jepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = configure_wandb_logger(cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    ckpt_interval = int(cfg.get("checkpoint_epoch_interval", 1))
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=ckpt_interval,
        keep_last_n=int(cfg.get("checkpoint_keep_last", 0)),
        epoch_offset=int(cfg.get("checkpoint_epoch_offset", 0)),
    )
    callbacks = [object_dump_callback]
    if cfg.get("planning_eval", {}).get("enabled", False):
        callbacks.append(PlanningEvalCallback(cfg))

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("enable_model_summary", None)
    trainer_kwargs.pop("enable_checkpointing", None)
    if os.environ.get("SLURM_JOB_ID"):
        trainer_kwargs["enable_progress_bar"] = False

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
    )

    # sbatch launches (unlike submitit) leave SIGUSR2 unbound. spt's default
    # SIGTERM→USR2 forwarder then hits the OS default and hard-kills the job.
    if os.environ.get("SPT_DISABLE_PREEMPT_HANDLER", "").lower() in {"1", "true", "yes"}:
        import stable_pretraining.manager as _spt_manager

        _spt_manager._install_sigterm_preempt_handler = lambda: None

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=None,
    )
    manager()


if __name__ == "__main__":
    # Optional: set MP_START_METHOD=spawn to avoid fork-related DataLoader issues.
    if os.environ.get("MP_START_METHOD", "").lower() == "spawn":
        import multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
    run()
