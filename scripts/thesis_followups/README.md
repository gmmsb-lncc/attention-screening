# Thesis follow-up experiments

Scripts in this directory implement non-architectural analyses requested during
the DT-Kinase v7 thesis review. All scripts keep the v7 architecture fixed
(see `configs/v7.yaml`) and only vary calibration, evaluation protocol, or
data scope.

| Script                          | Purpose                                        | Runs v7? |
|---------------------------------|------------------------------------------------|:--------:|
| `run_platt_vs_temperature.sh`   | #5  Platt vs Temperature calibration table     | yes (x2 per seed/dataset) |
| `run_pchembl_sensitivity.sh`    | #6  Activity threshold sensitivity (pChEMBL)   | yes (x5 thresholds) |
| `run_cross_species.sh`          | #7  Train Human -> test Non-Human (and vice)   | yes (x2 directions) |
| `bootstrap_ci.py`               | #3  Bootstrap CI + paired Wilcoxon over seeds  | post-hoc (no retraining) |

## Pre-requisites

- Activate the project env: `source env/bin/activate`
- The benchmark expects pre-computed ESM-2 / MoLFormer matrices under
  `results/protein_model_benchmark_{human|non_human}_v2/...`.
- All scripts use the canonical 5 seeds `[42, 123, 456, 789, 1024]` defined in
  `examples/crossattention_split_analysis/config.py:240`.

## Outputs

Each script writes under `results/thesis_followups/<experiment>/...` so the
main benchmark directory stays untouched.

## Aggregation into the thesis

After the scripts finish, run `bootstrap_ci.py` to emit the LaTeX table
fragments (median, 95% CI, paired Wilcoxon p-value) that should replace the
current "x ± sigma" rows in Chapter 5 tables 17 and 18.
