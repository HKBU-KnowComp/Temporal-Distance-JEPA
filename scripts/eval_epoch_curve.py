#!/usr/bin/env python3
"""Evaluate planning success for epochs 1..N of a training run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from planning_eval import run_planning_eval  # noqa: E402

ENV_CONFIG = {"tworoom": "tworoom", "dmc": "reacher", "reacher": "reacher", "ogb": "cube", "cube": "cube"}


def load_eval_cfg(env: str, seed: int, policy: str, mse_blend: float | None) -> OmegaConf:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(REPO / "config/eval")):
        cfg = compose(config_name=ENV_CONFIG[env])
    cfg.policy = policy
    cfg.seed = seed
    OmegaConf.set_struct(cfg, False)
    cfg.eval.seed = seed
    if mse_blend is not None and mse_blend > 0:
        if "planning_cost" not in cfg or cfg.planning_cost is None:
            cfg.planning_cost = {}
        cfg.planning_cost.mode = "td_jepa"
        cfg.planning_cost.mse_blend = mse_blend
    elif "planning_cost" not in cfg:
        cfg.planning_cost = {"mode": "td_jepa", "temperature": 0.1, "terminal_weight": 1.0}
    else:
        cfg.planning_cost.mode = "td_jepa"
    OmegaConf.set_struct(cfg, True)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted(ENV_CONFIG))
    parser.add_argument("--run", required=True, help="e.g. tworoom/td_jepa/seed_3072_10ep")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--mse-blend", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import stable_worldmodel as swm

    rows = []
    for ep in range(1, args.epochs + 1):
        policy = f"{args.run}/weights_epoch_{ep}.pt"
        ckpt = Path(swm.data.utils.get_cache_dir(), "checkpoints", policy)
        if not ckpt.exists():
            print(f"skip missing {ckpt}")
            continue
        object_ckpt = ckpt.parent / f"{ckpt.name}_object.ckpt"
        if object_ckpt.exists():
            model = torch.load(object_ckpt, map_location="cpu", weights_only=False)
        else:
            model = swm.wm.utils.load_pretrained(policy)
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        if hasattr(model, "interpolate_pos_encoding"):
            model.interpolate_pos_encoding = True
        cfg = load_eval_cfg(args.env, args.seed, policy, args.mse_blend)
        r = run_planning_eval(model, cfg)
        sr = r["success_rate"]
        rows.append({"epoch": ep, "success_rate": sr})
        print(f"ep{ep}: {sr:.1f}%")
        del model
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": args.env,
        "run": args.run,
        "seed": args.seed,
        "mse_blend": args.mse_blend,
        "epochs": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
