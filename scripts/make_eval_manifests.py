#!/usr/bin/env python3
"""Create fixed evaluation manifests for planning experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from planning_eval import get_dataset, get_episodes_length


def valid_start_rows(dataset, goal_offset_steps: int) -> tuple[np.ndarray, str]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_by_ep = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_by_ep[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    return np.nonzero(valid_mask)[0], col_name


def build_manifest(dataset, row_indices: np.ndarray, col_name: str, args) -> dict:
    rows = dataset.get_row_data(row_indices)
    return {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "goal_offset_steps": args.goal_offset_steps,
        "indices": [
            {
                "episode": int(ep),
                "start": int(step),
            }
            for ep, step in zip(rows[col_name], rows["step_idx"])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--keys-to-cache", default="action")
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_manifests"))
    args = parser.parse_args()

    keys_to_cache = [key for key in args.keys_to_cache.split(",") if key]
    dataset = get_dataset(args.dataset_name, keys_to_cache)
    valid_rows, col_name = valid_start_rows(dataset, args.goal_offset_steps)
    needed = args.val_size + args.test_size
    if valid_rows.size < needed:
        raise ValueError(
            f"Need {needed} valid starts but only found {valid_rows.size} "
            f"for dataset {args.dataset_name}."
        )

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(valid_rows, size=needed, replace=False)
    val_rows = np.sort(chosen[: args.val_size])
    test_rows = np.sort(chosen[args.val_size :])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.dataset_name.replace("/", "_")
    val_path = args.output_dir / f"{safe_name}_val_seed{args.seed}_n{args.val_size}.json"
    test_path = args.output_dir / f"{safe_name}_test_seed{args.seed}_n{args.test_size}.json"
    val_path.write_text(json.dumps(build_manifest(dataset, val_rows, col_name, args), indent=2))
    test_path.write_text(json.dumps(build_manifest(dataset, test_rows, col_name, args), indent=2))
    print(f"Wrote {val_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
