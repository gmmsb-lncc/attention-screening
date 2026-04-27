# Cross-dataset evaluation matrix

Full 3×3 matrix (train ∈ {H, NH, All} × test ∈ {H, NH, All}) over the four thesis models (DT-Kinase v7, ConPLex, DrugBAN, GraphBAN), reusing the existing diagonal checkpoints. No retraining — only inference.

## Leakage status (verified)

The `scaffolds_splits/output/` pipeline uses `test_mode=shared_scaffold` (see `manifest.json`), meaning test scaffolds are **propagated** across H, NH, and All partitions. A direct check confirms zero pair overlap in every off-diagonal combination:

| train → test | n_original | n_clean | frac_leaked |
|--------------|-----------:|--------:|------------:|
| all → human | 39 739 | 39 739 | 0.0000 |
| all → non_human | 1 702 | 1 702 | 0.0000 |
| human → all | 41 441 | 41 441 | 0.0000 |
| non_human → all | 41 441 | 41 441 | 0.0000 |
| human → non_human | 1 702 | 1 702 | 0.0000 |
| non_human → human | 39 739 | 39 739 | 0.0000 |

`leakage_filter.py` remains in the pipeline as a rigor safeguard. It rebuilds the `(seq_hash, canonical_smiles)` index from the training corpus train+val, compares against the test corpus, writes a filtered TSV, and emits a JSON report. With the current splits it is a no-op; if splits are ever rebuilt independently, it will catch regressions.

## Protocol

- Threshold: MCC-optimal on the **training corpus** val set (not the evaluated corpus). Matches deploy scenario.
- Calibrator: Platt scaling fit on training corpus val logits (DT-Kinase) or via `raw_predictions.npz` val probs (baselines).
- Seeds: `42 123 456 789 1024` (override via `SEEDS`).
- Cells: 6 off-diagonal evaluated by this pipeline. Diagonal imported from the existing benchmark runs.

## Pipeline

```
scripts/thesis_followups/cross_dataset_matrix/
├── leakage_filter.py       (train, test) -> test_clean.tsv + leakage_report.json
├── run_cross_matrix.sh     6 cells x 4 models x 5 seeds orchestration
├── aggregate.py            walks results/cross_matrix + diagonal dirs -> CSV + TeX
└── README.md
```

Plus these touched files outside this directory:
- `scripts/thesis_followups/eval_checkpoint_on_dataset.py` — rewritten to handle cross-corpus DT-Kinase v7 eval (accepts `--train-corpus` + `--eval-dataset` explicitly; rebuilds model from `configs/v7.yaml`).
- `infer_{drugban,graphban,conplex}_universal.py` — added `--val-tsv` + `--test-tsv` overrides.

## Running

```bash
# 1. Ensure diagonal v7 checkpoints exist under V7_CKPT_ROOT/{corpus}/seed_{s}/level4_cnn_model.pt
#    (thesis Objective 1 — rerun with Platt-on-val)

# 2. Run the off-diagonal matrix
V7_CKPT_ROOT=results/v7_diagonal \
SEEDS="42 123 456 789 1024" \
bash scripts/thesis_followups/cross_dataset_matrix/run_cross_matrix.sh

# 3. Aggregate (import diagonal via --diagonal-<model>-<corpus>)
python3 scripts/thesis_followups/cross_dataset_matrix/aggregate.py \
    --results-root results/cross_matrix \
    --out-dir results/cross_matrix/summary \
    --diagonal-dtkinase-human  results/benchmark_human_8M_13_04_2026/level4_cnn_8M \
    --diagonal-dtkinase-non_human results/benchmark_non_human_8M_13_04_2026/level4_cnn_8M \
    --diagonal-dtkinase-all results/benchmark_all_8M_13_04_2026/level4_cnn_8M \
    --diagonal-drugban-human DrugBAN/results_universal/human \
    --diagonal-drugban-non_human DrugBAN/results_universal/non_human \
    --diagonal-drugban-all DrugBAN/results_universal/all \
    --diagonal-graphban-human GraphBAN/results_universal/human \
    --diagonal-graphban-non_human GraphBAN/results_universal/non_human \
    --diagonal-graphban-all GraphBAN/results_universal/all \
    --diagonal-conplex-human ConPLex/results_universal/human \
    --diagonal-conplex-non_human ConPLex/results_universal/non_human \
    --diagonal-conplex-all ConPLex/results_universal/all
```

Outputs:
- `cross_matrix.csv` — per-(model, train, test) row with MCC/AUROC/AUPRC/F1 mean+std
- `cross_matrix.tex` — LaTeX 3×3 fragment (diagonal rendered in italic)
- `cross_matrix.json` — full raw report, includes leakage snapshot under `_leakage`

## Output layout

```
results/cross_matrix/
├── filters/{train}_to_{test}/test_clean.tsv + leakage_report.json
├── dtkinase/{train}_to_{test}/seed_{s}/metrics.json
├── drugban/{train}_to_{test}/seed_{s}/raw_predictions.npz
├── graphban/{train}_to_{test}/seed_{s}/raw_predictions.npz
├── conplex/{train}_to_{test}/seed_{s}/raw_predictions.npz
└── summary/cross_matrix.{csv,tex,json}
```

## Verification checklist

- [ ] Each leakage report shows `frac_leaked == 0.0` (sanity: splits propagated correctly).
- [ ] Diagonal cells reproduced from cross-matrix infra match existing thesis tables 17/18 to within numerical noise.
- [ ] `H→NH` MCC from cross-matrix matches `run_cross_species.sh` output for DT-Kinase v7.
- [ ] LaTeX fragment compiles inside `~/PhD/tex/` without package changes.

## Out of scope

- Retraining. All 4 models use existing diagonal checkpoints.
- Rebuilding scaffold splits. Current splits already propagate scaffolds across H/NH/All.
- Bootstrap CIs / paired Wilcoxon. `scripts/thesis_followups/bootstrap_ci.py` handles that post-hoc over the saved logits; wire it in after the 3×3 matrix stabilizes.
