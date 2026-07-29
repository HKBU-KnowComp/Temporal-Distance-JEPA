# Temporal-Distance-JEPA
### Temporal-Distance-JEPA: Plan-Aware Representation Learning for Latent World Model Predictive Control

**Temporal-Distance-JEPA** keeps the LeWM encoder–predictor and SIGReg backbone, and mines a directed temporal cost from reward-free demonstration logs. Same-trajectory step order supplies positive targets, cross-trajectory pairs act as heuristic negatives, and a rollout-consistency term matches the planner horizon. At plan time the mined cost \(d_\psi\) is deployed on topology-dominated tasks (Two-Room, Reacher), while contact-rich tasks (Push-T, OGB-Cube) plan with latent \(\ell_2\) on the same temporally trained checkpoint.

This repository is a public paper-reproduction release. Cluster/Slurm wrappers are intentionally omitted; use the Python CLI below and wrap it for your own scheduler if needed.

**Author:** [Jiaxin Bai](https://github.com/marcos0318) (HKBU KnowComp)

## Setup

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) and [stable-pretraining](https://github.com/galilai-group/stable-pretraining), following the same layout as [LeWM](https://github.com/lucas-maes/le-wm).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a writable cache for datasets and checkpoints (default `~/.stable-wm`):

```bash
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
```

Before training with WandB, set your entity (or disable logging with `wandb.enabled=false`):

```bash
# example override
python train.py --config-name=pusht_train wandb.config.entity=your_entity ...
```

## Data

Datasets are HDF5 files under `$STABLEWM_HOME` (or a `datasets/` subdirectory, depending on how you organize the cache). LeWM releases complementary data and checkpoints on [Hugging Face](https://huggingface.co/collections/quentinll/lewm).

| Config | Typical HDF5 | Env |
|--------|--------------|-----|
| `data=pusht` | `pusht_expert_train.h5` | Push-T |
| `data=tworoom` | `tworoom.h5` | Two-Room |
| `data=dmc` | `reacher.h5` | DMC Reacher |
| `data=ogb` | `ogbench/cube_single_expert.h5` | OGB-Cube |

Checkpoints are written under `$STABLEWM_HOME/checkpoints/<run_name>/`.

## Training

Hydra configs live under `config/train/`. Canonical method variant: `td_jepa`. Paper protocol uses **10 epochs**.

```bash
# Temporal-Distance-JEPA (main method; Hydra variant key remains td_jepa)
python train.py --config-name=pusht_train data=pusht variant=td_jepa seed=3072 \
  output_model_name=pusht/td_jepa/seed_3072_10ep

python train.py --config-name=tworoom_train data=tworoom variant=td_jepa seed=3072 \
  output_model_name=tworoom/td_jepa/seed_3072_10ep

python train.py --config-name=dmc_train data=dmc variant=td_jepa seed=3072 \
  output_model_name=dmc/td_jepa/seed_3072_10ep

python train.py --config-name=ogb_train data=ogb variant=td_jepa seed=3072 \
  output_model_name=ogb/td_jepa/seed_3072_10ep
```

Baselines and paper ablations:

```bash
# LeWM baseline
python train.py --config-name=pusht_train data=pusht variant=lewm seed=3072 \
  output_model_name=pusht/lewm/seed_3072_10ep

# RC-aux concurrent baseline
python train.py --config-name=pusht_train data=pusht variant=rc_aux seed=3072 \
  output_model_name=pusht/rc_aux/seed_3072_10ep

# Component ablations (Push-T / Reacher tables)
variant=td_jepa_euclidean_head   # symmetric Euclidean head instead of MRN
variant=td_jepa_hinge_off        # no cross-trajectory negative hinge
variant=td_jepa_no_rollout       # no multi-step rollout consistency
```

## Evaluation / planning

Eval configs live under `config/eval/`. Pass `policy` as a path **relative to `$STABLEWM_HOME/checkpoints`**, including the `.pt` weights file used by this codebase:

```bash
# Two-Room / Reacher: pure d_psi (defaults: iCEM-30)
python eval.py --config-name=tworoom \
  policy=tworoom/td_jepa/seed_3072_10ep/weights_epoch_10.pt \
  seed=20260714

python eval.py --config-name=reacher \
  policy=dmc/td_jepa/seed_3072_10ep/weights_epoch_10.pt \
  seed=20260714

# Push-T / OGB-Cube: latent L2 on the Temporal-Distance-JEPA checkpoint (defaults: CEM)
python eval.py --config-name=pusht \
  policy=pusht/td_jepa/seed_3072_10ep/weights_epoch_10.pt \
  seed=20260714

python eval.py --config-name=cube \
  policy=ogb/td_jepa/seed_3072_10ep/weights_epoch_10.pt \
  seed=20260714
```

Override cost or solver when needed:

```bash
# Plan the Temporal-Distance-JEPA checkpoint with d_psi instead of L2
python eval.py --config-name=pusht \
  policy=pusht/td_jepa/seed_3072_10ep/weights_epoch_10.pt \
  planning_cost.mode=td_jepa planning_cost.mse_blend=0.0

# Switch solver
python eval.py --config-name=tworoom solver=cem ...
```

### Locked evaluation protocol

Locked **50-episode** manifests are shipped in `eval_manifests/` (seed `20260714`). Primary paper stats use **10 plan seeds**:

`{20260714, 7, 11, 13, 17, 19, 23, 29, 31, 37}`

| Env | Solver | Plan cost | Terminal weight \(w\) |
|-----|--------|-----------|------------------------|
| Two-Room | iCEM-30 | \(d_\psi\) | 1.0 |
| Reacher | iCEM-30 | \(d_\psi\) | 0.3 |
| Push-T | CEM-30 | latent \(\ell_2\) | 1.0 |
| OGB-Cube | CEM-10 | latent \(\ell_2\) | 1.0 |

Shared planner budget: horizon \(H{=}5\), goal offset \(G{=}25\), 300 CEM/iCEM candidates. iCEM uses `noise_beta=2.0`, smoothing `0.1`, `n_elite_keep=5`.

Compact locked summaries used in the draft tables live under:

- `results/rc_aux_locked/` — RC-aux 10-seed dumps + `SUMMARY.md`
- `results/paper_locked/` — cost matrix, contact-gate, sweeps, diagnostics

## Helper CLIs

```bash
# Rebuild locked manifests (requires datasets on disk)
python scripts/make_eval_manifests.py --dataset-name pusht_expert_train

# Offline planning-cost sweep (L2 vs d_psi vs blends)
python scripts/sweep_planning_cost.py --env pusht \
  --policy pusht/td_jepa/seed_3072_10ep/weights_epoch_10.pt

# Temporal alignment diagnostic (Spearman vs step distance)
python scripts/analyze_pusht_distance_alignment.py \
  --policy pusht/td_jepa/seed_3072_10ep/weights_epoch_10.pt

# Epoch curve (e.g. Two-Room)
python scripts/eval_epoch_curve.py --env tworoom \
  --run tworoom/td_jepa/seed_3072_10ep --output logs/curve_tworoom.json

# Aggregate planning log success rates
python scripts/aggregate_planning_results.py \
  --input td_jepa=path/to/result.txt --input lewm=path/to/other.txt
```

## Repo layout

| Path | Contents |
|------|----------|
| `train.py` / `eval.py` | Hydra entry points |
| `jepa.py` / `module.py` / `losses.py` | Model, MRN head, TD / rollout / SIGReg losses |
| `planning_eval.py` | Shared planning evaluation helpers |
| `config/train/variant/` | `td_jepa`, `lewm`, `rc_aux`, paper ablations |
| `config/eval/` | Per-env planning defaults matching the paper protocol |
| `eval_manifests/` | Locked episode indices |
| `results/` | Locked table artifacts |
| `scripts/` | Public reproduction CLIs (no Slurm) |

## Notes

- Display / paper name: **Temporal-Distance-JEPA**. Config and checkpoint short names remain `td_jepa` (e.g. `variant=td_jepa`, `planning_cost.mode=td_jepa`).
- Historical contrastive SoftJEPA loss helpers may still appear in `losses.py` / `jepa.py` for compatibility; they are not part of the public paper configs.
- Do not commit local `outputs/`, `logs/`, or checkpoints; see `.gitignore`.
