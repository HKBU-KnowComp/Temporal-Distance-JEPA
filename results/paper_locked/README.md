# Paper-locked experiment summaries

Compact artifacts promoted from gitignored `logs/` for the KDD draft.
Raw per-seed dumps for RC-aux remain in `results/rc_aux_locked/`.

| Path | Contents |
| --- | --- |
| `cost_matrix/summary.json` | Locked cost composition matrix (dψ / ℓ₂ / blend) across envs |
| `cost_matrix/status.json` | Completion status for the cost-matrix jobs |
| `pusht_contact_gate/` | Contact-gated planning vs fixed costs (v1/v2) |
| `pusht_multiseed/` | Push-T multi training-seed locked planning |
| `reacher_ablation/` | Reacher 10ep component / protocol ablations |
| `pusht_diagnostics/` | Alignment, failure, contact-neighborhood diagnostics |
| `sweeps_curves/` | OGB/Push-T offline CEM sweeps and epoch curves |

## Files

- `cost_matrix/status.json`
- `cost_matrix/summary.json`
- `pusht_contact_gate/pusht_contact_gate_summary.json`
- `pusht_contact_gate/pusht_contact_gate_v2_summary.json`
- `pusht_diagnostics/pusht_align_td_jepa_seed3072_ep10.json`
- `pusht_diagnostics/pusht_align_td_jepa_seed3073_ep10.json`
- `pusht_diagnostics/pusht_align_td_jepa_seed3074_ep10.json`
- `pusht_diagnostics/pusht_contact_neighborhood.json`
- `pusht_diagnostics/pusht_cost_disagreement.json`
- `pusht_diagnostics/pusht_failure_td_jepa_seed3072_ep10.json`
- `pusht_diagnostics/pusht_failure_td_jepa_seed3073_ep10.json`
- `pusht_diagnostics/pusht_failure_td_jepa_seed3074_ep10.json`
- `pusht_diagnostics/strengthen_exp123_analysis.json`
- `pusht_multiseed/pusht_multiseed_train_summary.json`
- `reacher_ablation/reacher_ablation_10ep_partial_summary.txt`
- `reacher_ablation/summary.txt`
- `reacher_ablation/summary_val_seed20260714.json`
- `sweeps_curves/curve_dmc_td_jepa.json`
- `sweeps_curves/curve_ogb_td_jepa.json`
- `sweeps_curves/curve_ogb_td_jepa_blend01.json`
- `sweeps_curves/curve_tworoom_td_jepa.json`
- `sweeps_curves/sweep_ogb_cube_local.json`
- `sweeps_curves/sweep_pusht_td_jepa_log_tau_seed3074_epoch10.json`
- `sweeps_curves/sweep_pusht_td_jepa_neargoal_weighted_seed3072_epoch10.json`
- `sweeps_curves/sweep_pusht_td_jepa_seed3072_epoch10.json`
- `sweeps_curves/sweep_pusht_td_jepa_seed3072_epoch10_rerun.json`
