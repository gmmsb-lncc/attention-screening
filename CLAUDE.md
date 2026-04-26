# CLAUDE.md

Guidance for Claude Code operating on this repository.

## Context files (versioned, sync across machines)

When resuming work after `/clear`, host change, or fresh repo clone,
these documents reconstitute the full project state. None are
gitignored; all sync via `git pull`.

| File | Purpose | Read when |
|---|---|---|
| `CLAUDE.md` | this file — overview, configs, env knobs, hosts, key dev notes | always (auto-loaded) |
| `licoes_aprendidas.md` | optimization track narrative + **20 methodological lessons** + §9 operational snapshot + §10 future directions | when planning next experiment or interpreting a result |
| `experiments_log.md` | persistent table of every benchmark run with full extracted metrics (MCC, F1, AUROC, etc.) per host + raw JSON yaml stanzas | when comparing configs, validating reproducibility, or cross-checking values |
| `v8.md` | v8 multi-source POC architecture document (ChemBERTa/BioBERT/ADMET/ClassyFire injection); separate experimental track from main optimization | only when working on multi-source feature injection |
| `README.md` | repo-level onboarding (high-level setup, install) | new contributors |
| `configs/*.yaml` | training configurations (each carries inline docstring) | when running or comparing variants |
| `git log --oneline -30` | recent commits with detailed messages explaining each change | when reconstructing recent decisions |

**Update rule**: any new methodological insight goes to
`licoes_aprendidas.md` (with explicit lesson number). Any new
multi-seed result goes to `experiments_log.md` (table row + YAML
stanza). Any new config/env knob/runner script gets a row in the
`CLAUDE.md` tables below.

## Project overview

**semantic-screening** — framework for protein-ligand interaction prediction on kinases. Implements **DT-Kinase** (Level 4 CNN v7, CNN 2D + bi-modal cross-attention) and evaluates it against three baselines (**DrugBAN**, **GraphBAN**, **ConPLex**) under an identical protocol.

Scientific thesis: *semantic screening* — predicting bioactivity from 1D linear notations (amino acids, SMILES) without 3D structures or hand-crafted descriptors. Some models use PLMs (DT-Kinase, GraphBAN, ConPLex), one is from scratch (DrugBAN); the paradigm is defined by the input modality, not the encoder.

Repo: `gmmsb-lncc/semantic-screening` · Python 3.12 in `env/` · PyTorch 2.0+ · MIT.

## Active work (2026-04, 20 lessons; canonical revalidated under LEGACY adapter)

1. **DT-Kinase optimization track** — push v7 MCC from baseline (~0.506 NH 5-seed) toward 0.55-0.60 target. Tracked in `licoes_aprendidas.md` (sections §4-§11, **20 methodological lessons** documented). State actual:
   - **Canônico MCC primário** — `v7+F` (Tier A + C + F) sob `ADAPTER_LEGACY=1` (default desde commit `de2ef0e`): **0.5266 ± 0.010** NH 3-seed em d01 cuDNN ON. Re-validado pelo experimento de isolamento §6.9.1 (lição 17).
   - **Canônico p/ comparação F1 baselines** — `v7+F_adapt` (matched THR/SEL=f1): 0.4929 ± 0.016, F1=0.787 (competitivo com DrugBAN/GraphBAN F1 nativo).
   - **Best variance candidate** — `v7+F + Direção A+D` (lig 2L/12h + lr_mult_lig=5x): ~0.530 ± **0.004** (cross-host adj para d01) — MCC empate, σ 2.5× melhor que base. Lição 19 (capacidade ↔ otimização mutuamente dependentes).
   - **Em execução / pendente** — Grid sweep A+D (d02 NH, opcionalmente H+All); BAN-residual (`v7_ban_res`, lição 12 reformulada com α-gate identity-init, pendente).
   - **Refutados (descartados)** — Tier B (Xavier head_proj), D vanilla (SWA), E (Mixup), §6.5 fixes (zero-cascade), v7_asymF (3 mudanças simultâneas), v7+F_adapt_v2 (lição 16 decoupled), v7_ban_F (Xavier W_ban substitutivo), LoRA-MLM offline (lição 20 objetivo desalinhado).
2. **Cross-dataset evaluation matrix (3×3)** — completo. Off-diagonal v7=0.298, DrugBAN=0.348, GraphBAN=0.342, ConPLex=0.209. **Documentação completa migrada para `~/PhD/cross_matrix/`** (figures + Beamer source + Anexo A da tese).
3. **Rerun v7 benchmark with Platt-on-val** (PR #206) — completed. All thesis Chapter 5 MCC numbers regenerated.
4. **Post-hoc statistics** — `scripts/thesis_followups/bootstrap_ci.py` on saved logits (CI 95% + paired Wilcoxon) replaces "x ± σ" in thesis tables 17 & 18.

## PR workflow

Open PRs against `cross_attention_lite`, not `main`. `main` = stable releases; `cross_attention_lite` = active integration. `gh pr create --base cross_attention_lite`.

## Thesis repository link

Sibling repo `/Users/sulfierry/PhD` (LaTeX). Defines "semantic screening", scaffold-split protocol, multi-seed protocol (5 seeds), MCC-optimal threshold, standardized model order (DT-Kinase, ConPLex, DrugBAN, GraphBAN). Any metric/architecture/protocol change here requires update under `~/PhD/tex/`.

Recently added (2026-04-26):
- `~/PhD/tex/anexoA.tex` — Anexo A: Matriz Cross-Dataset 3×3 (heatmaps + análise).
- `~/PhD/tex/apendiceF.tex` — Apêndice F: 20 lições metodológicas com rigor acadêmico.
- `~/PhD/cross_matrix/` — Beamer source + figures + alternate versions (migrado de `slides/`).
- `~/PhD/figures/{heatmap_*,bar_*}.pdf` — figuras cross-matrix referenciadas pelo Anexo A.

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

## DT-Kinase optimization variants (2026-04, snapshot)

Track: improve v7 toward MCC ≥ 0.55-0.60 on NH. Detailed log + lessons in `licoes_aprendidas.md` (20 methodological lessons across §4-§11).

**Status flags resumidos**:
- **CONFIRMED (§6.9.1, lição 17)**: §6.5 EmbeddingAdapter fixes (pre-norm + LoRA gates + zero-init self-attn) **prejudicam** v7+F por −0.053 MCC, AUROC −0.033, σ ×5.6. Default flipped (commit `de2ef0e`): `BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY=1` é o default; v7+F reproduz 0.5266 ± 0.010 em d01.
- **NEW (§6.10, lição 19)**: capacity (Direção A) e LR (Direção D) sobre o mesmo módulo são **mutuamente dependentes**. A+D combo = ~0.530 (adj.) ± **0.004** (σ 2.5× melhor que base); A só ou D só regridem (−0.027/−0.031). Devem ser tratadas como intervenção atômica.
- **REFUTED (§6.11, lição 20)**: LoRA-MLM-offline em MoLFormer top-2 layers regrediu −0.025 MCC, AUROC −0.010. MLM ≠ tarefa downstream + corpus pequeno (5276 SMILES) → encoder literalmente pior. Pivotar para LoRA-end-to-end se houver retomada.
- **PENDING**: `v7_ban_res` (lição 12 reformulada — BAN-residual com α-gate identity-init); grid sweep A+D em H/All; LoRA end-to-end (não implementado).

| Config | Tiers / Variante | Status (NH) | Notas |
|---|---|---|---|
| `configs/v7.yaml` | baseline | 0.486 (seed 42) / 0.506 (5-seed ref) | thesis reference |
| `configs/v7_plus.yaml` | A + C | **0.5143 ± 0.0079** (5-seed) | validado multi-seed |
| `configs/v7_plus_F.yaml` | A + C + F | **0.5266 ± 0.010** (3-seed, LEGACY default) | **CANÔNICO MCC primário — re-validado §6.9.1** |
| `configs/v7_plus_F_adapt.yaml` | A + C + F + §6.5 + matched F1 | 0.4929 ± 0.016 / F1=0.787 | **canônico p/ comparação baselines F1** |
| `v7+F + A+D combo` (env-only, sem yaml) | A + C + F + lig 2L/12h + lig_lr=5x | ~0.530 ± **0.004** (cross-host adj) | **best-σ candidato** — lição 19 |
| `configs/v7_ban_res.yaml` | A + C + F + BAN-residual α-gate | **PENDENTE** | Lição 12 reformulada; sanity OK; aguarda 3-seed |
| `configs/v7_asymF.yaml` | A + C + F + asym adapter | regredido (0.461 ± 0.028) | confounded com §6.5; redundante após lição 19 |
| `configs/v7_plus_F_adapt_v2.yaml` | A + C + F + §6.5 + THR=f1, SEL=mcc | **DESCARTADO** (0.459 ± 0.052) | lição 16: matched objective; AUROC regrediu |
| `configs/v7_ban_F.yaml` | A + C + F + BAN puro (variant=v8) | regredido (0.503 ± 0.046) | W_ban Xavier viola identity-init; lição 12 |
| `configs/v7_pro.yaml` | A + C + E + F | regredido (0.496 ± 0.025) | Mixup deletério |
| `configs/v7_plus_E.yaml` | A + C + E (Mixup) | regredido (0.499 ± 0.025) | confirmed Mixup harmful |
| `results/lora/molformer_*` | A + C + F + LoRA-MLM cache | **REFUTADO** (~0.502 adj) | lição 20 |

**Tier glossary** (sigla → mecanismo):
- **A** (capacidade adapter): `num_heads=16`, `head_dim=64`, `mlp_head=true`, adapter `prot=512/lig=1024`, `patience=15`, `lr_mult=2.0`. ✓ aceito.
- **C** (contrastive aux): `contrastive_weight=0.3`, `cosine_feat=true`. ✓ aceito.
- **F** (label smoothing): `label_smooth=0.05`. ✓ aceito.
- **E** (Mixup): `mixup_alpha=0.3`. ✗ REJEITADO (lição 11).
- **B** (multi-head pool=4): Xavier `head_proj`. ✗ REJEITADO (lição 2).
- **SWA vanilla** (`swa_start=5`): ✗ REJEITADO (lição 4).
- **§6.5 fixes** (pre-norm + LoRA gates + zero-init self-attn): ✗ REJEITADO (lição 17).
- **Direção A** (asimetria estrutural lig adapter): `_ADAPTER_LAYERS_LIG=2`, `_ADAPTER_ATTN_HEADS_LIG=12`. Atomicamente acoplado a Direção D (lição 19).
- **Direção D** (asimetria LR adapter): `_ADAPTER_LR_MULT_LIG=5.0`, `_ADAPTER_LR_MULT_PROT=2.0`. Atomicamente acoplado a Direção A (lição 19).
- **BAN-residual** (lição 12 reformulada): `BENCHMARK_LEVEL4CNN_BAN_RESIDUAL=1`. ⏳ pendente.

**Runners disponíveis** (todos em `scripts/v8/`):

| Script | Propósito |
|---|---|
| `run_v7_yaml.sh` | runner genérico, lê `V7_CONFIG`, `CORPUS`, `SEEDS` env |
| `run_v7_plus_F_adapt.sh` | v7+F + §6.5 + matched F1 (THR=SEL=f1) |
| `run_v7_plus_F_adapt_v2.sh` | v7+F + §6.5 + THR=f1 SEL=mcc — **DESCARTADO** |
| `run_v7_ban_res.sh` | v7+F + BAN-residual α-gate (lição 12 reformulada) |
| `run_AD_grid_d02.sh` | grid 6 cells: lr_mult_lig {3,5,8} × layers_lig {2,3} |
| `run_v7_lora_d03.sh` | LoRA-MLM offline pipeline (3 stages) — **REFUTADO** |
| `aggregate_AD_grid.py` | aggregator multi-corpus do grid sweep |
| `lora_finetune_molformer.py` | LoRA MLM trainer (standalone) |
| `recache_molformer_lora.py` | inference + cache embeddings via LoRA delta |
| `run_v7_pro_validation.sh` | validação NH→Human sequencial 5-seed |
| `run_ablation_E_F_3seeds.sh` | ablation E vs F isolada |

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

Documentação principal (figures, slides, anexo da tese) está em **`~/PhD/cross_matrix/`** + `~/PhD/tex/anexoA.tex`. Scripts de execução permanecem aqui.

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

- **Production config (thesis baseline)**: `configs/v7.yaml`. Best validated multi-seed: `configs/v7_plus_F.yaml` LEGACY default (0.5266 ± 0.010).
- **Hosts atuais (3 paralelos)**:
  - `diamante-01` (cuDNN HEALTHY): host primário, MCC reference. Default `BENCHMARK_LEVEL4CNN_DISABLE_CUDNN=0`.
  - `diamante-02` (cuDNN 9.x ABI mismatch driver 12.4): precisa `_DISABLE_CUDNN=1`. Cross-host drift +0.009 MCC vs d01.
  - `diamante-03` (RTX 4090, cuDNN 9.10.2 disponível MAS runner default desabilita): cuDNN OFF na prática. Drift similar a d02. Fastest GPU.
  - **Comparações MCC válidas só dentro do mesmo host**. Para inter-host, ajustar drift +0.009 (d02/d03→d01).
- **Adapter LEGACY default** (lição 17): `BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY=1` é o default. Toda execução de `v7+F` herda adapter histórico (post-norm + Xavier self-attn + zero-init só na última Linear MLP). Setar `=0` opt-in para §6.5 fixes (regressivos).
- **Threshold + Selection metric** (env, matched obrigatório por lição 16):
  - `BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC` ∈ {`mcc`, `f1`}. Default `mcc`.
  - `BENCHMARK_LEVEL4CNN_SELECTION_METRIC` ∈ {`mcc`, `f1`}. Default herda THRESHOLD_METRIC.
  - Decoupling testado e refutado (lição 16/`v7+F_adapt_v2`).
- **Per-side adapter LR multipliers** (lição 19): `BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT` e `_LR_MULT_LIG`. Default herda `adapter_lr_mult` do yaml. Direção D = `_LR_MULT_LIG=5.0` (acoplado a Direção A).
- **Per-side adapter capacity** (Direção A, lição 19): `_ADAPTER_LAYERS_LIG`, `_ADAPTER_ATTN_HEADS_LIG`, idem `_PROT`. Default herda yaml.
- **BAN-residual** (lição 12 reformulada, pendente): `BENCHMARK_LEVEL4CNN_BAN_RESIDUAL=1` ativa caminho residual gateado por α=0 (identity-init).
- **Composite checkpoint criterion** (lição 14): `score = val_mcc - λ·val_loss` via `_SELECTION_LAMBDA_LOSS` (default 0 = pure val_mcc).
- **Numerical regime**: fp32 + AMP off + TF32 off (auto quando `no_amp=true + double=false`).
- **Calibração**: Platt on val (default `_PLATT=1`). Temperature opt-in (`_TEMPERATURE=1`).
- **Threshold selection** (4 modelos comparados): MCC-optimal on val (canônico DT-Kinase) ou F1-optimal val (DrugBAN/GraphBAN nativo). Para comparação justa, DT-Kinase usa F1 via `v7+F_adapt`.
- **`MultiTaskLoss`**: vestigial, NOT used in v7.
- **ESM**: `src/__init__.py` adds `llm/ESM/` to `sys.path`. Never `pip install esm`.
- **Dataset `all`**: H + NH combined; 386 099 post-filter samples. Embeddings loaded from both `benchmark_{human,non_human}_v2` dirs.
- **Seq limits**: protein MAX_SEQ_LEN=1024 (C-terminal truncation of long kinases, e.g. ULK1); ligand SMILES 512 tokens (rarely hit).
- **Non-determinism**: cuDNN non-deterministic reductions ON by default. σ over seeds captures init + reduction noise. Strict determinism via `BENCHMARK_LEVEL4CNN_DETERMINISTIC=1` (~20–30 % slower).
- **No k-fold**: fixed scaffold split + 5-seed variance estimation (3-seed para iteração rápida; 5-seed para canônico final).
- **Two-phase pipeline gotcha** (lição 7): `benchmark_comparison.json` "phase: train" e "phase: test" são modelos DIFERENTES. Para `level4_cnn` o train se faz sempre sobre `train_loader`; o `mode` apenas controla se `test_loader` é construído + se reporta val ou test.
- **LoRA cache override** (refutado mas infra existe): `BENCHMARK_LEVEL4CNN_PROTEIN_CACHE_OVERRIDE` e `_LIGAND_CACHE_OVERRIDE` redirecionam dataloaders para caches alternativos (LoRA-FT-ed). Lição 20 mostra que MLM offline não ajuda; reservado para LoRA end-to-end futuro.

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
