# CLAUDE.md

Guidance for Claude Code operating on this repository.

## Project overview

**semantic-screening** — framework for protein-ligand interaction prediction on kinases. Implements **DT-Kinase** (Level 4 CNN v7, CNN 2D + bi-modal cross-attention) and evaluates it against three baselines (**DrugBAN**, **GraphBAN**, **ConPLex**) under an identical protocol.

Scientific thesis: *semantic screening* — predicting bioactivity from 1D linear notations (amino acids, SMILES) without 3D structures or hand-crafted descriptors. Some models use PLMs (DT-Kinase, GraphBAN, ConPLex), one is from scratch (DrugBAN); the paradigm is defined by the input modality, not the encoder.

Repo: `gmmsb-lncc/semantic-screening` · Python 3.12 in `env/` · PyTorch 2.0+ · MIT.

## Active work (2026-04)

1. **Rerun v7 benchmark with Platt-on-val** (PR #206). All thesis Chapter 5 MCC numbers regenerated. Source of truth: `configs/v7.yaml` (B=128, dropout=0.35, λ=0.04, patience=15, float64).
2. **Cross-dataset evaluation matrix (3×3)** — train ∈ {H, NH, All} × test ∈ {H, NH, All}. Infra in `scripts/thesis_followups/cross_dataset_matrix/`. Reuses diagonal checkpoints, no retraining. Off-diagonal runs on diamante-02 via `run_cross_matrix.sh`.
3. **Post-hoc statistics** — `scripts/thesis_followups/bootstrap_ci.py` on saved logits (CI 95% + paired Wilcoxon) replaces "x ± σ" in thesis tables 17 & 18.

## PR workflow

Open PRs against `cross_attention_lite`, not `main`. `main` = stable releases; `cross_attention_lite` = active integration. `gh pr create --base cross_attention_lite`.

## Thesis repository link

Sibling repo `/Users/sulfierry/PhD` (LaTeX). Defines "semantic screening", scaffold-split protocol, multi-seed protocol (5 seeds), MCC-optimal threshold, standardized model order (DT-Kinase, ConPLex, DrugBAN, GraphBAN). Any metric/architecture/protocol change here requires update under `~/PhD/tex/`.

## Environment setup

```bash
# Main env (v7 + utilities)
python setup.py && source env/bin/activate   # creates env/, installs deps
# or: source activate_env.sh                 # also sets PYTHONPATH

# ESM-2 cloned locally (pip version segfaults)
git clone https://github.com/facebookresearch/esm.git llm/ESM
```

Baselines need their own conda envs (`drugban`, `graphban`, `conplex`) — see each `<Model>/setup_env.sh`.

## Testing

```bash
pytest                                       # all
pytest -m "unit and not slow"                # fast unit only
pytest tests/test_cross_attention_model.py   # one file
```

Markers: `slow`, `integration`, `unit`, `regression`, `classifier`, `build`, `requires_gpu`, `requires_data`.

## DT-Kinase v7 (production)

Canonical path: **`benchmark/levels/level4_cnn.py::InteractionMapCNN`** trained via `run_from_config.py configs/v7.yaml`. Exploratory variants under `src/classifier/` and `src/attention_matrix/` are **not** the benchmark.

- **Encoders (frozen)**: ESM-2 8M (320-dim) + MoLFormer (768-dim)
- **Adapters**: `EmbeddingAdapter` 4-head self-attn + MLP bottleneck (prot=256, lig=512), output zero-init (identity at start)
- **Interaction maps**: `M_k = P_k L_k^T / √d_h`, K=8 heads, d_head=32, raw (no softmax marginalization)
- **CNN 2D**: 4 layers (8→32→64→64→64), dilation=2 only layer 3, BN2d+GELU
- **Pooling**: hierarchical attention (lig axis → prot axis, single learnable query per stage)
- **Head**: dropout 0.35 → Linear(64,1) → Focal Loss (γ=2.0) → BCEWithLogitsLoss; α_+ = clip(N_-/N_+, 1, 20)
- **Calibration**: Platt (logistic reg on val logits). Temperature scaling opt-in.
- **Threshold**: MCC-optimal on val, 2-pass (100 coarse + 100 fine ±0.05).

Checkpoint on disk is **bare state_dict** at `{output_dir}/level4_cnn_model.pt`; metrics in `level4_cnn_results.json`; predictions in `raw_predictions.npz` (keys `y_true`, `y_prob`).

### Running

```bash
python3 run_from_config.py configs/v7.yaml --dataset {human|non_human|all} [--dry-run]
bash run_benchmark.sh                         # shell orchestrator
```

## Baselines (equitable protocol)

| Model | Dir | Encoders | Publication |
|-------|-----|----------|-------------|
| DrugBAN | `DrugBAN/` | CNN 1D + GCN (from scratch, no PLM) | Nat. Mach. Intell. |
| GraphBAN | `GraphBAN/` | ESM-1b + ChemBERTa | Nat. Commun. |
| ConPLex | `ConPLex/` | ProtBERT + Morgan FP (contrastive) | PNAS |
| KinBAN (WIP) | `KinBAN/` | experimental fork | — |

Shared protocol: identical scaffold splits, seeds `[42, 123, 456, 789, 1024]`, MCC-optimal threshold on val, MCC primary metric (AUROC/AUPRC/F1 secondary). `raw_predictions.npz` for baselines has keys `val_y_{true,prob}, test_y_{true,prob}`.

## Data layout

**Inputs** (gitignored, ~415 MB): `tests/datasets/kinase_{human,non_human,all}_compounds.tsv`.

**Scaffold splits** — shared-scaffold propagation (`test_mode=shared_scaffold` in `scaffolds_splits/output/manifest.json`): H_test, NH_test, All_test all share the same scaffolds. **Cross-corpus leakage verified = 0** across the 6 off-diagonal pairs. Files:
- `scaffolds_splits/output/{human,non_human,universal}_{train,val,test}.tsv` (13 cols; universal adds `dataset_source`)
- `DrugBAN/datasets/kinase/{human,non_human,all}/scaffold/{train,val,test}.csv` (3 cols: `SMILES,Protein,Y`)

**Pre-computed embeddings**: `./results/protein_model_benchmark_{human|non_human}_v2/{embed}/build/{protein_matrices,ligand_matrices,molformer_matrix}/`. Run once, reused across all seeds/epochs.

## Cross-dataset matrix infra

Directory: `scripts/thesis_followups/cross_dataset_matrix/`

| File | Role |
|------|------|
| `leakage_filter.py` | `(seq_hash, canonical_smiles)` index + filter + report. No-op under shared_scaffold; kept as rigor safeguard. |
| `run_cross_matrix.sh` | Orchestrator, 6 off-diagonal × N models × 5 seeds. Per-corpus V7 checkpoint vars: `V7_CKPT_{HUMAN,NON_HUMAN,ALL}`. |
| `run_conplex_only.sh` | Same orchestrator pinned to ConPLex. Use after all 5 reps train. |
| `aggregate.py` | Walks `results/cross_matrix/{model}/{train}_to_{test}/seed_*/` + imports 9 diagonals. Emits CSV + LaTeX + JSON. Vectorised MCC sweep. |

Supporting (repo root):
- `scripts/thesis_followups/eval_checkpoint_on_dataset.py` — rebuilds `InteractionMapCNN` from `configs/v7.yaml`, fits Platt + τ on training-corpus val, evaluates on `--eval-dataset` test/val split.
- `infer_{drugban,graphban,conplex}_universal.py` — have `--val-tsv` + `--test-tsv` overrides (default: training-corpus splits).

Diagonal-only aggregation with data from diamante-02:
```bash
python3 scripts/thesis_followups/cross_dataset_matrix/aggregate.py \
    --diagonal-dtkinase-{human,non_human,all}   results/semantic-screening-results/dt-kinase/benchmark_{c}_8M_<DATE>/test/level4_cnn_8M/{c} \
    --diagonal-{drugban,graphban}-{human,non_human,all} results/semantic-screening-results/{m}/{c} \
    --diagonal-conplex-{human,non_human}        ConPLex/results_universal/{c} \
    --results-root results/cross_matrix --out-dir results/cross_matrix/summary
```

## Key development notes

- **Production config**: `configs/v7.yaml`. Override via env vars (`BENCHMARK_LEVEL4CNN_*`) or CLI flags only.
- **Threshold selection** (all 4 models): MCC-optimal on val. Enforced for equitable comparison.
- **Calibration**: Platt on val (default `BENCHMARK_LEVEL4CNN_PLATT=1`). Temperature on val opt-in (`_TEMPERATURE=1`).
- **`MultiTaskLoss`**: vestigial, NOT used in v7.
- **ESM**: `src/__init__.py` adds `llm/ESM/` to `sys.path`. Never `pip install esm`.
- **Dataset `all`**: H + NH combined; 386 099 post-filter samples. Embeddings loaded from both `benchmark_{human,non_human}_v2` dirs.
- **Seq limits**: protein MAX_SEQ_LEN=1024 (C-terminal truncation of long kinases, e.g. ULK1); ligand SMILES 512 tokens (rarely hit).
- **Non-determinism**: cuDNN non-deterministic reductions ON by default. σ over seeds captures init + reduction noise. Strict determinism via `BENCHMARK_LEVEL4CNN_DETERMINISTIC=1` (~20–30 % slower).
- **No k-fold**: fixed scaffold split + 5-seed variance estimation.

## Thesis follow-up scripts

`scripts/thesis_followups/`:

| Script | Purpose |
|--------|---------|
| `run_platt_vs_temperature.sh` | Calibration comparison table |
| `run_pchembl_sensitivity.sh` | τ sweep ∈ {5.5, 6.0, 6.5, 7.0, 7.5} |
| `run_cross_species.sh` | H↔NH transfer (legacy, superseded by `cross_dataset_matrix/`) |
| `eval_checkpoint_on_dataset.py` | Single-cell v7 eval (used by matrix) |
| `bootstrap_ci.py` | Post-hoc CI 95% + paired Wilcoxon (no retraining) |
| `cross_dataset_matrix/` | Full 3×3 infra (see section above) |

`scripts/count_v7_params.py` — trainable/frozen parameter breakdown for v7.

## context-mode

Routing rules are injected per-session via hook (`ctx_batch_execute`, `ctx_search`, `ctx_execute` etc., and `curl`/`wget`/`WebFetch` blocks). Follow them. `ctx stats` / `ctx doctor` / `ctx upgrade` / `ctx purge` are the user-visible commands.
