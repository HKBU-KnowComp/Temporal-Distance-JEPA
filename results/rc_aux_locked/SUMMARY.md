# Locked RC-aux planning results

Concurrent baseline reimplementation (`config/train/variant/rc_aux.yaml`).
Protocol: train seed `3072`, 10 epochs, locked 50-episode val manifests,
10 plan seeds `{20260714,7,11,13,17,19,23,29,31,37}`, latent L2 / mse planning cost
(auxiliary reachability shaping only; no budgeted reachability cost at plan time).

| Env | Protocol | RC-aux | LeWM (locked) | TD-JEPA (locked) |
| --- | --- | --- | --- | --- |
| Two-Room | iCEM-30, latent L2, w=1 | **98.6±1.0** | 97.4±1.3 | 100.0±0.0 |
| Reacher | iCEM-30, latent L2, w=0.3 | **96.8±1.4** | 96.0±1.9 | 97.0±2.4 |
| Push-T | CEM-30, latent L2, w=1 | **81.4±1.9** | 83.6±3.2 | 86.0±4.2 |
| OGB-Cube | CEM-10, latent L2, w=1 | **81.6±2.8** | 68.0±2.8 | 82.2±2.9 |

Raw per-seed eval dumps: `results/rc_aux_locked/<env>/rcaux_*_planseed*_val.txt`.
Aggregate logs with `python scripts/aggregate_planning_results.py`.
