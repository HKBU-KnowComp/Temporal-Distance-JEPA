#!/usr/bin/env python3
"""Aggregate planning-evaluation logs and paired confidence intervals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SUCCESS_RE = re.compile(r"'success_rate':\s*([0-9.]+)")
EPISODES_RE = re.compile(r"'episode_successes':\s*array\(\[(.*?)\]\)", re.S)


def parse_eval_file(path: Path) -> dict:
    text = path.read_text(errors="replace")
    success_match = SUCCESS_RE.search(text)
    if not success_match:
        raise ValueError(f"No success_rate found in {path}")
    episode_match = EPISODES_RE.search(text)
    episodes = None
    if episode_match:
        tokens = re.findall(r"\bTrue\b|\bFalse\b", episode_match.group(1))
        episodes = np.array([tok == "True" for tok in tokens], dtype=bool)
    return {
        "path": str(path),
        "success_rate": float(success_match.group(1)),
        "episode_successes": episodes,
    }


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have same shape, got {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = a.size
    diffs = np.empty(n_boot, dtype=float)
    a_float = a.astype(float)
    b_float = b.astype(float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = 100.0 * (a_float[idx].mean() - b_float[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(100.0 * (a_float.mean() - b_float.mean())),
        "ci95": [float(lo), float(hi)],
        "n": int(n),
        "n_boot": int(n_boot),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Label and path, formatted as label=/path/to/log_or_result.txt",
    )
    parser.add_argument(
        "--paired",
        action="append",
        default=[],
        help="Paired comparison as label_a:label_b",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = {}
    for item in args.input:
        if "=" not in item:
            raise ValueError(f"Invalid --input {item!r}; expected label=path")
        label, path_text = item.split("=", 1)
        grouped.setdefault(label, []).append(parse_eval_file(Path(path_text)))

    summary = {}
    for label, rows in grouped.items():
        rates = np.array([row["success_rate"] for row in rows], dtype=float)
        summary[label] = {
            "n_runs": int(len(rows)),
            "success_rates": rates.tolist(),
            "mean": float(rates.mean()),
            "std": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
            "paths": [row["path"] for row in rows],
        }

    paired = {}
    for spec in args.paired:
        left, right = spec.split(":", 1)
        if len(grouped[left]) != 1 or len(grouped[right]) != 1:
            raise ValueError("Paired comparisons currently require exactly one file per label")
        a = grouped[left][0]["episode_successes"]
        b = grouped[right][0]["episode_successes"]
        if a is None or b is None:
            raise ValueError(f"Missing episode_successes for paired comparison {spec}")
        paired[spec] = bootstrap_diff(a, b, args.n_boot, args.seed)

    payload = {"summary": summary, "paired": paired}
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
