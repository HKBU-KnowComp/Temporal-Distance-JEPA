#!/usr/bin/env python3
"""Analyze how Push-T planning costs rank temporal progress.

The key diagnostic is whether the learned directed cost ``d_psi`` and latent
L2 rank same-trajectory state pairs consistently with true step distance, split
by whether the target state is far from or near the end of the demonstration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from planning_eval import get_dataset, img_transform  # noqa: E402


@dataclass(frozen=True)
class Pair:
    start_row: int
    goal_row: int
    tau: int
    remaining_to_end: int


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation sufficient for Spearman diagnostics."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_model(policy: str, cache_dir: str | None):
    import stable_worldmodel as swm

    policy_path = Path(policy)
    object_ckpt = Path(
        swm.data.utils.get_cache_dir(cache_dir),
        policy_path.parent,
        f"{policy_path.name}_object.ckpt",
    )
    if object_ckpt.exists():
        model = torch.load(object_ckpt, map_location="cpu", weights_only=False)
    else:
        model = swm.wm.utils.load_pretrained(policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    elif hasattr(model, "encoder") and hasattr(model.encoder, "config"):
        model.encoder.config.interpolate_pos_encoding = True
    if getattr(model, "dist_head", None) is None:
        raise RuntimeError("Policy does not expose a TD-JEPA distance head.")
    return model


def episode_index_column(dataset) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise KeyError("Dataset must contain episode_idx or ep_idx.")


def collect_episode_rows(dataset) -> dict[int, np.ndarray]:
    ep_col = episode_index_column(dataset)
    episodes = dataset.get_col_data(ep_col)
    steps = dataset.get_col_data("step_idx")
    rows_by_ep: dict[int, np.ndarray] = {}
    for ep in np.unique(episodes):
        rows = np.nonzero(episodes == ep)[0]
        rows_by_ep[int(ep)] = rows[np.argsort(steps[rows])]
    return rows_by_ep


def sample_pairs(
    rows_by_ep: dict[int, np.ndarray],
    *,
    num_pairs: int,
    max_tau: int,
    seed: int,
) -> list[Pair]:
    rng = np.random.default_rng(seed)
    usable = [rows for rows in rows_by_ep.values() if len(rows) > 2]
    if not usable:
        raise ValueError("No episodes with enough steps for pair sampling.")

    pairs: list[Pair] = []
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 50:
        attempts += 1
        rows = usable[int(rng.integers(0, len(usable)))]
        ep_len = len(rows)
        i = int(rng.integers(0, ep_len - 1))
        max_j = min(ep_len - 1, i + max_tau)
        if max_j <= i:
            continue
        j = int(rng.integers(i + 1, max_j + 1))
        pairs.append(
            Pair(
                start_row=int(rows[i]),
                goal_row=int(rows[j]),
                tau=j - i,
                remaining_to_end=ep_len - 1 - j,
            )
        )
    if len(pairs) < num_pairs:
        raise RuntimeError(f"Sampled only {len(pairs)} pairs out of {num_pairs}.")
    return pairs


def encode_rows(model, dataset, rows: np.ndarray, batch_size: int, image_size: int) -> dict[int, torch.Tensor]:
    transform = img_transform(image_size)
    encoded: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            batch = dataset.get_row_data(batch_rows)
            pixels = batch["pixels"]
            frames = [transform(frame) for frame in pixels]
            tensor = torch.stack(frames, dim=0).unsqueeze(1).to("cuda")
            info = model.encode({"pixels": tensor})
            emb = info["emb"][:, 0].detach().cpu()
            for row, z in zip(batch_rows, emb, strict=True):
                encoded[int(row)] = z
    return encoded


def summarize_split(name: str, mask: np.ndarray, tau: np.ndarray, dpsi: np.ndarray, l2: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "count": int(mask.sum()),
        "dpsi_spearman_tau": spearman(dpsi[mask], tau[mask]),
        "l2_spearman_tau": spearman(l2[mask], tau[mask]),
        "dpsi_mean": float(dpsi[mask].mean()) if mask.any() else None,
        "l2_mean": float(l2[mask].mean()) if mask.any() else None,
        "tau_mean": float(tau[mask].mean()) if mask.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Checkpoint path relative to STABLEWM_HOME/checkpoints.")
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--max-tau", type=int, default=25)
    parser.add_argument("--near-goal-steps", type=int, default=10)
    parser.add_argument("--far-goal-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("logs/pusht_distance_alignment.json"))
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    # Do not cache pixels here: Push-T images are large; stream sampled rows only.
    dataset = get_dataset(args.dataset_name, [], args.cache_dir)
    rows_by_ep = collect_episode_rows(dataset)
    pairs = sample_pairs(
        rows_by_ep,
        num_pairs=args.num_pairs,
        max_tau=args.max_tau,
        seed=args.seed,
    )
    unique_rows = np.array(
        sorted({p.start_row for p in pairs} | {p.goal_row for p in pairs}),
        dtype=np.int64,
    )

    model = load_model(args.policy, args.cache_dir)
    embeddings = encode_rows(model, dataset, unique_rows, args.batch_size, args.image_size)

    start = torch.stack([embeddings[p.start_row] for p in pairs], dim=0).to("cuda")
    goal = torch.stack([embeddings[p.goal_row] for p in pairs], dim=0).to("cuda")
    with torch.inference_mode():
        dpsi_values = model.distance(start, goal).detach().cpu().numpy()
        l2_values = (start - goal).pow(2).sum(dim=-1).detach().cpu().numpy()

    tau = np.array([p.tau for p in pairs], dtype=np.float64)
    remaining = np.array([p.remaining_to_end for p in pairs], dtype=np.int64)
    near_mask = remaining <= args.near_goal_steps
    far_mask = remaining >= args.far_goal_steps

    payload = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "num_pairs": len(pairs),
        "max_tau": args.max_tau,
        "near_goal_steps": args.near_goal_steps,
        "far_goal_steps": args.far_goal_steps,
        "splits": [
            summarize_split("all", np.ones(len(pairs), dtype=bool), tau, dpsi_values, l2_values),
            summarize_split("far_from_goal", far_mask, tau, dpsi_values, l2_values),
            summarize_split("near_goal", near_mask, tau, dpsi_values, l2_values),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
