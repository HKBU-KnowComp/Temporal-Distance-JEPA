import os

os.environ["MUJOCO_GL"] = "egl"

from compat_dm_control import apply_dm_control_mujoco_compat

apply_dm_control_mujoco_compat()

import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from planning_eval import (
    append_results_file,
    build_processors,
    get_dataset,
    img_transform,
    run_planning_eval,
    sample_eval_indices,
)


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run planning evaluation for a trained world model or random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    dataset = get_dataset(cfg.eval.dataset_name, cfg.dataset.keys_to_cache, cfg.get("cache_dir"))
    process = build_processors(dataset, cfg.dataset.keys_to_cache)
    transform = {
        "pixels": img_transform(cfg.eval.img_size),
        "goal": img_transform(cfg.eval.img_size),
    }

    policy = cfg.get("policy", "random")
    if policy != "random":
        policy_path = Path(cfg.policy)
        object_ckpt = Path(
            swm.data.utils.get_cache_dir(cfg.cache_dir),
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

        result = run_planning_eval(model, cfg, dataset=dataset, process=process)
        print(result["metrics"])
    else:
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        world = swm.World(**cfg.world, image_shape=(224, 224), num_envs=cfg.eval.num_eval)
        policy = swm.policy.RandomPolicy()
        eval_episodes, eval_start_idx = sample_eval_indices(
            dataset, cfg.eval, seed=int(cfg.get("seed", 42))
        )
        world.set_policy(policy)
        start_time = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes,
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=cfg.eval.get("save_video", None),
        )
        result = {
            "metrics": metrics,
            "success_rate": float(metrics.get("success_rate", 0.0)),
            "evaluation_time": time.time() - start_time,
        }
        print(result["metrics"])

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    results_path = results_path / cfg.output.filename
    append_results_file(results_path, cfg, result)


if __name__ == "__main__":
    # See train.py: keep the default "fork" start method; "spawn" forces every
    # DataLoader worker to re-import the full heavy stack and stalls startup.
    run()
