#!/usr/bin/env python3
"""Sweep planning cost on a saved checkpoint (no retraining).

Supports Push-T and OGB-Cube eval configs. Re-evaluates the same weights with
latent L2, learned d_psi, or blends.

Example:
  python scripts/sweep_planning_cost.py --env pusht \\
    --policy pusht/td_jepa/seed_3072_10ep/weights_epoch_10.pt

  python scripts/sweep_planning_cost.py --env cube \\
    --policy ogb/td_jepa/seed_3072_10ep/weights_epoch_10.pt
"""
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

ENV_CONFIG = {
    "pusht": "pusht",
    "cube": "cube",
    "ogb": "cube",
}


def parse_alphas(text: str) -> list[float]:
    if not text.strip():
        return [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def load_eval_cfg(env: str, seed: int, policy: str) -> OmegaConf:
    from hydra import compose, initialize_config_dir

    config_name = ENV_CONFIG[env]
    config_dir = str(REPO / "config/eval")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    cfg.policy = policy
    cfg.seed = seed
    OmegaConf.set_struct(cfg, False)
    if "eval" not in cfg:
        cfg.eval = {}
    cfg.eval.seed = seed
    OmegaConf.set_struct(cfg, True)
    return cfg


def eval_mode(model, cfg, mode: str, mse_blend: float) -> dict:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if mode == "mse":
        cfg.planning_cost.mode = "mse"
        cfg.planning_cost.pop("mse_blend", None)
    else:
        cfg.planning_cost.mode = "td_jepa"
        cfg.planning_cost.mse_blend = mse_blend
    return run_planning_eval(model, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=sorted(ENV_CONFIG),
        default="pusht",
        help="Eval config: pusht or cube/ogb",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Checkpoint path relative to STABLEWM_HOME/checkpoints/",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--alphas",
        default="",
        help="Comma-separated mse_blend values for td_jepa mode (default: sweep grid)",
    )
    parser.add_argument("--include-mse", action="store_true", default=True)
    parser.add_argument("--no-include-mse", action="store_false", dest="include_mse")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import stable_worldmodel as swm

    policy_path = Path(args.policy)
    object_ckpt = Path(
        swm.data.utils.get_cache_dir(),
        policy_path.parent,
        f"{policy_path.name}_object.ckpt",
    )
    if object_ckpt.exists():
        model = torch.load(object_ckpt, map_location="cpu", weights_only=False)
    else:
        model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    base_cfg = load_eval_cfg(args.env, args.seed, args.policy)

    rows: list[dict] = []

    if args.include_mse:
        r = eval_mode(model, base_cfg, "mse", 0.0)
        rows.append(
            {
                "mode": "mse",
                "mse_blend": None,
                "success_rate": r["success_rate"],
                "time_s": r["evaluation_time"],
            }
        )
        print(f"mse: {r['success_rate']:.1f}% ({r['evaluation_time']:.1f}s)")

    for alpha in parse_alphas(args.alphas):
        label = f"td_jepa+blend({alpha:g})" if alpha > 0 else "td_jepa"
        r = eval_mode(model, base_cfg, "td_jepa", alpha)
        rows.append(
            {
                "mode": "td_jepa",
                "mse_blend": alpha,
                "success_rate": r["success_rate"],
                "time_s": r["evaluation_time"],
            }
        )
        print(f"{label}: {r['success_rate']:.1f}% ({r['evaluation_time']:.1f}s)")

    best = max(rows, key=lambda x: x["success_rate"])
    print(f"\nBest: {best['success_rate']:.1f}% (mode={best['mode']}, blend={best['mse_blend']})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "env": args.env,
            "policy": args.policy,
            "seed": args.seed,
            "results": rows,
            "best": best,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
