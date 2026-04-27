# CLAUDE.md

Guidance for Claude Code operating on this repository.

## Context files (versioned, sync across machines)

When resuming work after `/clear`, host change, or fresh repo clone,
these documents reconstitute the full project state. None are
gitignored; all sync via `git pull`.

| File | Purpose | Read when |
|---|---|---|
| `CLAUDE.md` | this file — overview, configs, env knobs, hosts, key dev notes | always (auto-loaded) |
| `docs/01-methodology/licoes_aprendidas.md` | optimization track narrative + **23 methodological lessons** + §9 operational snapshot + §10 future directions + §6.13 plateau analysis | when planning next experiment or interpreting a result |
| `docs/01-methodology/experiments_log.md` | persistent table of every benchmark run with full extracted metrics (MCC, F1, AUROC, etc.) per host + raw JSON yaml stanzas | when comparing configs, validating reproducibility, or cross-checking values |
| `docs/01-methodology/v8.md` | v8 multi-source POC architecture document (ChemBERTa/BioBERT/ADMET/ClassyFire injection); separate experimental track from main optimization | only when working on multi-source feature injection |
| `README.md` | repo-level onboarding (high-level setup, install) | new contributors |
| `configs/*.yaml` | training configurations (each carries inline docstring) | when running or comparing variants |
| `git log --oneline -30` | recent commits with detailed messages explaining each change | when reconstructing recent decisions |

**Update rule**: any new methodological insight goes to
`docs/01-methodology/licoes_aprendidas.md` (with explicit lesson number). Any new
multi-seed result goes to `docs/01-methodology/experiments_log.md` (table row + YAML
stanza). Any new config/env knob/runner script gets a row in the
`CLAUDE.md` tables below.

## Project overview

**semantic-screening** — framework for protein-ligand interaction prediction on kinases. Implements **DT-Kinase** (Level 4 CNN v7, CNN 2D + bi-modal cross-attention) and evaluates it against three baselines (**DrugBAN**, **GraphBAN**, **ConPLex**) under an identical protocol.

Scientific thesis: *semantic screening* — predicting bioactivity from 1D linear notations (amino acids, SMILES) without 3D structures or hand-crafted descriptors. Some models use PLMs (DT-Kinase, GraphBAN, ConPLex), one is from scratch (DrugBAN); the paradigm is defined by the input modality, not the encoder.

Repo: `gmmsb-lncc/semantic-screening` · Python 3.12 in `env/` · PyTorch 2.0+ · MIT.

## Active work (2026-04, 23 lessons; PLATEAU empírico atingido sobre v7+F)

1. **DT-Kinase optimization track** — push v7 MCC from baseline (~0.506 NH 5-seed) toward 0.55-0.60 target. Tracked in `docs/01-methodology/licoes_aprendidas.md` (sections §4-§14, **24 methodological lessons** documented). **STATUS PLATEAU (lição 21, §6.13)**: **15+ modificações incrementais testadas, nenhuma supera vanilla v7 reproducivelmente em 5-seed**. Espaço incremental empiricamente esgotado. **Lição 22 (§6.14)**: 2D RoPE per-modality regrediu −0.020 MCC — *category error* cross-modal. **Lição 23 (§6.12.1)**: BAN-residual + BAN_LR_MULT=5 regrediu −0.016 MCC — acoplamento multiplicativo zera ∇W. **Lição 24 (NOVO)**: v7+F LEGACY 3-seed (0.5266 ± 0.010) regrediu para 0.4923 ± 0.025 em 5-seed (z=0.55σ vs vanilla v7) — Tier F efeito nulo sob 5-seed; aplicação direta de Lição 3 ao próprio v7+F.
   - **Canônico MCC primário (final, pós-Lição 24)** — `configs/v7.yaml` vanilla v7: **0.506 ± 0.006** NH 5-seed (cap 5 da tese). Ckpt operacional p/ matriz cross-dataset + comitê inferência + ranking final. v7+F LEGACY ré-classificado como tentativa que falhou validação 5-seed.
   - **Canônico p/ comparação F1 baselines** — `v7+F_adapt` (matched THR/SEL=f1): 0.4929 ± 0.016, F1=0.787 (competitivo com DrugBAN/GraphBAN F1 nativo).
   - **A+D combo** — empate técnico ~0.530 (d03 single host). Claim de σ=0.004 **RETRATADA** após grid sweep d02 mostrar σ=0.037 mesma config (lição 19, retração).
   - **BAN-residual `v7_ban_res`** — REGREDIU −0.010 a −0.018 MCC, σ=0.045 (lição 19 reaplica: 3.93M params extras sem LR matching). §6.12.
   - **Refutados/empates (13 direções)** — Tier B, D vanilla SWA, E Mixup, §6.5 fixes, v7_asymF, v7_adapt_v2 (decoupled), v7_ban_F (Xavier W_ban), LoRA-MLM offline, λ=0.5 composite, A só, D só, BAN-residual, A+D combo.
   - **Direções restantes (não-incrementais, §6.13)** — encoder maior (ESM-35M/150M), treino prolongado (centenas epochs), LoRA end-to-end, multi-task RDKit properties, BAN-residual + LR boost atomicamente.
   - **Recomendação operacional**: tese fecha sobre vanilla v7 (0.506 ± 0.006 NH 5-seed) como ckpt canônico + v7+F_adapt como comparação F1. v7+F LEGACY tratado como caso de Lição 24 (tentativa promissora 3-seed que falha validação 5-seed; protocolo cinco-sementes é necessário p/ canonização).
2. **Cross-dataset evaluation matrix (3×3)** — completo. Off-diagonal v7=0.298, DrugBAN=0.348, GraphBAN=0.342, ConPLex=0.209. **Documentação completa migrada para `~/PhD/cross_matrix/`** (figures + Beamer source + Anexo A da tese).
3. **Interpretabilidade DT-Kinase** — `scripts/inference/explain.py` POC implementado: extrai per-residue + per-ligand-token attention via forward hooks (M_k pre-CNN + HierPool weights). Diferenciador metodológico vs ConPLex (sem atenção); paridade com DrugBAN/GraphBAN BAN attn mas com 3 níveis interpretáveis (raw + per-head + HierPool).
4. **Rerun v7 benchmark with Platt-on-val** (PR #206) — completed. All thesis Chapter 5 MCC numbers regenerated.
5. **Post-hoc statistics** — `scripts/thesis_followups/bootstrap_ci.py` on saved logits (CI 95% + paired Wilcoxon) replaces "x ± σ" in thesis tables 17 & 18.

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

Track: improve v7 toward MCC ≥ 0.55-0.60 on NH. Detailed log + lessons in `docs/01-methodology/licoes_aprendidas.md` (20 methodological lessons across §4-§11).

**Status flags resumidos**:
- **CONFIRMED (§6.9.1, lição 17)**: §6.5 EmbeddingAdapter fixes (pre-norm + LoRA gates + zero-init self-attn) **prejudicam** v7+F por −0.053 MCC, AUROC −0.033, σ ×5.6. Default flipped (commit `de2ef0e`): `BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY=1` é o default; v7+F reproduz 0.5266 ± 0.010 em d01.
- **NEW (§6.10, lição 19)**: capacity (Direção A) e LR (Direção D) sobre o mesmo módulo são **mutuamente dependentes**. A+D combo = empate técnico ~0.530. Devem ser tratadas como intervenção atômica. **Claim de redução de variância (σ=0.004) RETRATADO** após grid d02 produzir σ=0.037 mesma config — foi acidente single-host.
- **REFUTED (§6.11, lição 20)**: LoRA-MLM-offline em MoLFormer top-2 layers regrediu −0.025 MCC, AUROC −0.010. MLM ≠ tarefa downstream + corpus pequeno (5276 SMILES) → encoder literalmente pior. Pivotar para LoRA-end-to-end se houver retomada.
- **REFUTED (§6.12)**: BAN-residual (`v7_ban_res`, lição 12 reformulada com α-gate identity-init) regrediu −0.010 a −0.018 MCC, σ ×4.5. Identity-init OK mas Lição 19 reaplica: 3.93M params extras sem LR matching = sub-treinado. Direção corretiva: BAN_LR_MULT dedicado (não testado).
- **REFUTED (§6.14, lição 22)**: 2D RoPE per-modality (`v7_rope`) regrediu −0.020 MCC, AUROC quase igual (−0.002). Cross-modal *category error*: termo (i−j) de RoPE assume comparabilidade entre posições; em M_k[prot_pos, lig_pos] os eixos são entidades distintas em escalas diferentes — sinal posicional injetado é espúrio. Heurística: NÃO usar RoPE 1D direto em ambos eixos cross-modais.
- **PLATEAU (§6.13, lição 21)**: **14 modificações incrementais** sobre v7+F testadas, **nenhuma supera baseline reproducivelmente**. Espaço incremental empiricamente esgotado. Próximas direções devem ser não-incrementais: encoder maior, treino prolongado, LoRA end-to-end.

| Config | Tiers / Variante | Status (NH) | Notas |
|---|---|---|---|
| `configs/v7.yaml` | baseline | 0.486 (seed 42) / 0.506 (5-seed ref) | thesis reference |
| `configs/v7_plus.yaml` | A + C | **0.5143 ± 0.0079** (5-seed) | validado multi-seed |
| `configs/v7_plus_F.yaml` | A + C + F | 3-seed: 0.5266 ± 0.010; **5-seed: 0.4923 ± 0.025** (LEGACY default) | **REFUTADO em 5-seed (Lição 24)** — empate vs vanilla v7 |
| `configs/v7_plus_F_adapt.yaml` | A + C + F + §6.5 + matched F1 | 0.4929 ± 0.016 / F1=0.787 | **canônico p/ comparação baselines F1** |
| `v7+F + A+D combo` (env-only, sem yaml) | A + C + F + lig 2L/12h + lig_lr=5x | ~0.530 (empate, single host) | claim σ=0.004 RETRATADO (grid d02 σ=0.037); lição 19 |
| `configs/v7_ban_res.yaml` | A + C + F + BAN-residual α-gate | regrediu (~0.508 ± 0.045) | §6.12 — capacidade extra sem LR matching, lição 19 reaplica |
| `configs/v7_ban_res_lr.yaml` | A + C + F + BAN-residual + BAN_LR_MULT=5 | regrediu (0.511 ± 0.046, NH 2-seed d03) | §6.12.1 — LR-boost não recupera; gargalo é gate multiplicativo (α·W) |
| `configs/v7_plus_F_adv.yaml` | A + C + F + CDAN adversarial DA (corpus=all) | em execução d02 | requer batches H+NH para sinal não-zero |
| `configs/v7_plus_F_morgan.yaml` | A + C + F + Morgan FP topológico | regrediu (-0.085 MCC) | proxy GCN não recuperou DrugBAN signal — refutado d01 |
| `configs/v7_plus_F_coral.yaml` | A + C + F + CORAL covariance match (corpus=all) | em execução d01 | mecanismo ortogonal a CDAN, sem head adversarial |
| `configs/v7_rope.yaml` | A + C + F + 2D RoPE per-modality | regrediu (~0.506 adj, 0.497 ± 0.022 d02) | §6.14 — category error cross-modal, lição 22 |
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
- **BAN-residual** (lição 12 reformulada): `BENCHMARK_LEVEL4CNN_BAN_RESIDUAL=1`. ✗ REJEITADO em todas as parametrizações (vanilla §6.12 e LR-boosted §6.12.1).
- **CDAN adversarial DA**: `BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA=0.1`, requer `CORPUS=all`. ⏳ d02.
- **CORAL DA** (covariance match, sem GRL): `BENCHMARK_LEVEL4CNN_CORAL_LAMBDA=0.1`, requer `CORPUS=all`. ⏳ d01.
- **Morgan FP** (proxy GCN): `BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR=...`. ✗ REJEITADO (-0.085 MCC).

**Runners disponíveis** (todos em `scripts/v8/`):

| Script | Propósito |
|---|---|
| `run_v7_yaml.sh` | runner genérico, lê `V7_CONFIG` (default `configs/v7_plus_F.yaml` desde commit `4831032`), `CORPUS`, `SEEDS` env |
| `run_v7_plus_F_adapt.sh` | v7+F + §6.5 + matched F1 (THR=SEL=f1) |
| `run_v7_plus_F_adapt_v2.sh` | v7+F + §6.5 + THR=f1 SEL=mcc — **DESCARTADO** |
| `run_v7_ban_res.sh` | v7+F + BAN-residual α-gate (lição 12 reformulada) |
| `run_AD_grid_d02.sh` | grid 6 cells: lr_mult_lig {3,5,8} × layers_lig {2,3} |
| `run_v7_lora_d03.sh` | LoRA-MLM offline pipeline (3 stages) — **REFUTADO** |
| `run_v7_morgan_d01.sh` | v7+F + Morgan FP topológico — **REFUTADO** (-0.085 MCC) |
| `run_v7_adv_d02.sh` | v7+F + CDAN adversarial DA (corpus=all) — ⏳ d02 em execução |
| `run_v7_banres_lr_d03.sh` | v7+F + BAN-residual + BAN_LR_MULT=5 — **REFUTADO** Lição 23 |
| `run_v7_coral_d01.sh` | v7+F + CORAL covariance match (corpus=all) — ⏳ d01 em execução |
| `aggregate_AD_grid.py` | aggregator multi-corpus do grid sweep |
| `lora_finetune_molformer.py` | LoRA MLM trainer (standalone) |
| `recache_molformer_lora.py` | inference + cache embeddings via LoRA delta |
| `precompute_ligand_morgan.py` | precompute Morgan FP por chembl_id (RDKit) |
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

- **Production config (thesis baseline + canonical, post-Lição 24)**: `configs/v7.yaml` vanilla v7, **0.506 ± 0.006** NH 5-seed. v7+F LEGACY refutado em 5-seed (0.4923 ± 0.025), retido apenas no histórico do Apêndice F.
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
- **BAN-residual** (lição 12 reformulada, REFUTADO §6.12 + §6.12.1): `BENCHMARK_LEVEL4CNN_BAN_RESIDUAL=1` ativa caminho residual gateado por α=0. `BENCHMARK_LEVEL4CNN_BAN_LR_MULT` (default 1.0) aplica LR-boost dedicado em W_k+α_k.
- **CDAN adversarial DA** (em validação d02): `BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA` (default 0.0) ativa DomainAdversarialHead com Gradient Reversal. `_ADVERSARIAL_N_DOMAINS` (default 2). Requer `CORPUS=all` para batches mistos.
- **CORAL DA** (em validação d01): `BENCHMARK_LEVEL4CNN_CORAL_LAMBDA` (default 0.0) ativa loss `||Cov(f_dom0)-Cov(f_dom1)||²_F` sobre pooled features. Sem head adversarial, sem GRL. Requer `CORPUS=all`.
- **Morgan FP auxiliary** (REFUTADO -0.085 MCC): `BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR=<path>` aciona projeção `_MORGAN_BITS` (default 1024) → `_MORGAN_PROJ` (default 32) concatenada ao pooled vector.
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
