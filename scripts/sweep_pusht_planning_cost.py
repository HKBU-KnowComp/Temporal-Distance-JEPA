#!/usr/bin/env python3
"""Backward-compatible wrapper for Push-T planning-cost sweeps."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (str(REPO), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sweep_planning_cost import main  # noqa: E402

if __name__ == "__main__":
    if "--env" not in sys.argv:
        sys.argv[1:1] = ["--env", "pusht"]
    main()
