"""Shared PushT planning evaluation used by eval.py and training callbacks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

os.environ.setdefault("MUJOCO_GL", "egl")

from compat_dm_control import apply_dm_control_mujoco_compat

apply_dm_control_mujoco_compat()


def img_transform(img_size: int):
    import stable_pretraining as spt

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(dataset_name: str, keys_to_cache, cache_dir=None):
    dataset_path = Path(cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=list(keys_to_cache),
        cache_dir=dataset_path,
    )


def configure_planning_cost(model, planning_cost_cfg) -> None:
    if planning_cost_cfg is None or not hasattr(model, "set_planning_cost"):
        return

    mode = normalize_planning_cost_mode(str(planning_cost_cfg.get("mode", "mse")))
    kwargs = {
        "mode": mode,
        "temperature": float(planning_cost_cfg.get("temperature", 0.1)),
        "terminal_weight": float(planning_cost_cfg.get("terminal_weight", 1.0)),
    }
    if "mse_blend" in planning_cost_cfg:
        kwargs["mse_blend"] = float(planning_cost_cfg.get("mse_blend", 0.0))
    # Contact-gated planner: d_ψ far from contact, ℓ₂ (or high blend) near contact.
    for src, dst in (
        ("contact_dist", "contact_dist"),
        ("far_mse_blend", "far_mse_blend"),
        ("near_mse_blend", "near_mse_blend"),
        ("gate_temperature", "gate_temperature"),
    ):
        if src in planning_cost_cfg:
            kwargs[dst] = float(planning_cost_cfg.get(src))
    model.set_planning_cost(**kwargs)


def normalize_planning_cost_mode(mode: str) -> str:
    """Map deprecated planning-cost aliases to canonical names."""
    aliases = {
        "temporal_distance": "td_jepa",
        "softjepa_energy": "contrastive_cosine_energy",
        "softjepa_l2_energy": "contrastive_l2_energy",
        "softjepa_gaussian_energy": "contrastive_gaussian_energy",
        "gated": "contact_gate",
    }
    return aliases.get(mode, mode)


def default_planning_cost_mode(prediction_loss_name: str, energy: str = "dot_product") -> str:
    if str(prediction_loss_name).lower() == "mse":
        return "mse"
    energy = str(energy).lower()
    if energy in ("negative_squared_l2", "squared_l2", "l2"):
        return "softjepa_l2_energy"
    if energy in ("gaussian_nll", "gaussian"):
        return "softjepa_gaussian_energy"
    return "softjepa_energy"


def build_processors(dataset, keys_to_cache):
    process = {}
    for col in keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_indices(dataset, eval_cfg, seed=42, rng=None):
    manifest_path = eval_cfg.get("manifest_path", None)
    if manifest_path:
        path = Path(str(manifest_path)).expanduser()
        payload = json.loads(path.read_text())
        entries = payload.get("indices", payload)
        if not isinstance(entries, list):
            raise ValueError(f"Invalid evaluation manifest format: {path}")
        eval_episodes = [
            int(item.get("episode", item.get("episode_idx", item.get("ep_idx"))))
            for item in entries
        ]
        eval_start_idx = [
            int(item.get("start", item.get("start_idx", item.get("step_idx"))))
            for item in entries
        ]
        if len(eval_episodes) != int(eval_cfg.num_eval):
            raise ValueError(
                f"Manifest {path} has {len(eval_episodes)} entries but "
                f"eval.num_eval={eval_cfg.num_eval}."
            )
        return eval_episodes, eval_start_idx

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - eval_cfg.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        raise ValueError("No valid starting points found for planning evaluation.")

    rng = rng or np.random.default_rng(seed)
    random_episode_indices = rng.choice(
        len(valid_indices) - 1, size=eval_cfg.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < eval_cfg.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")
    return eval_episodes.tolist(), eval_start_idx.tolist()


def build_policy(model, cfg: DictConfig, process, transform):
    configure_planning_cost(model, cfg.get("planning_cost"))
    config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


def load_policy_from_checkpoint(cfg: DictConfig):
    policy_path = Path(cfg.policy)
    object_ckpt = Path(
        swm.data.utils.get_cache_dir(cfg.get("cache_dir")),
        policy_path.parent,
        f"{policy_path.name}_object.ckpt",
    )
    if object_ckpt.exists():
        model = torch.load(object_ckpt, map_location="cpu", weights_only=False)
    else:
        model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    elif hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config.interpolate_pos_encoding = True

    dataset = get_dataset(cfg.eval.dataset_name, cfg.dataset.keys_to_cache, cfg.get("cache_dir"))
    process = build_processors(dataset, cfg.dataset.keys_to_cache)
    transform = {
        "pixels": img_transform(cfg.eval.img_size),
        "goal": img_transform(cfg.eval.img_size),
    }
    policy = build_policy(model, cfg, process, transform)
    return policy, dataset


def run_planning_eval(model, cfg: DictConfig, dataset=None, process=None) -> dict[str, Any]:
    """Evaluate a world model on PushT planning and return metrics."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    was_training = model.training
    model.eval()
    model.requires_grad_(False)

    if dataset is None:
        dataset = get_dataset(cfg.eval.dataset_name, cfg.dataset.keys_to_cache, cfg.get("cache_dir"))
    if process is None:
        process = build_processors(dataset, cfg.dataset.keys_to_cache)

    transform = {
        "pixels": img_transform(cfg.eval.img_size),
        "goal": img_transform(cfg.eval.img_size),
    }

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = cfg.eval.num_eval
    world_cfg["max_episode_steps"] = 2 * cfg.eval.eval_budget
    world = swm.World(**world_cfg, image_shape=(224, 224))

    eval_model = model.to("cuda")
    policy = build_policy(eval_model, cfg, process, transform)
    eval_episodes, eval_start_idx = sample_eval_indices(
        dataset,
        cfg.eval,
        seed=int(cfg.get("seed", cfg.eval.get("seed", 42))),
    )

    world.set_policy(policy)
    start_time = time.time()
    with torch.inference_mode():
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=cfg.eval.get("save_video", None),
        )
    elapsed = time.time() - start_time

    if was_training:
        model.train()
        model.requires_grad_(True)

    success_rate = float(metrics.get("success_rate", 0.0))
    return {
        "success_rate": success_rate,
        "metrics": metrics,
        "evaluation_time": elapsed,
        "num_eval": cfg.eval.num_eval,
    }


def append_results_file(results_path: Path, cfg: DictConfig, result: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {result['metrics']}\n")
        f.write(f"evaluation_time: {result['evaluation_time']} seconds\n")
