# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**semantic-screening** is an open-source framework for predicting protein-ligand interactions, focusing on kinase targets. It implements the **DT-Kinase** architecture (CNN 2D + Bi-modal Cross-Attention) and orchestrates baseline evaluations against state-of-the-art models (DrugBAN, GraphBAN, ConPLex).

The unifying scientific concept is **"semantic screening"**: predicting compound bioactivity by extracting meaning exclusively from the linear notation of sequences (amino acids for proteins, SMILES for ligands), without requiring 3D coordinates or heuristic descriptors. While DT-Kinase, GraphBAN, and ConPLex leverage pre-trained Language Models (PLMs), the framework also accommodates models trained entirely from scratch (such as DrugBAN), demonstrating that the semantic paradigm is defined by the input modality (1D primary sequences) rather than the specific encoding strategy.

**Repository**: gmmsb-lncc/semantic-screening | **License**: MIT | **Python**: 3.9+ (env uses 3.12) | **PyTorch**: 2.0+

## Current Objectives (active work)

Near-term priorities driving active commits:

1. **Rerun v7 benchmark with Platt-on-val calibration.** A recent fix (PR #206) moved the Platt scaling fit from train logits to validation logits, matching the thesis text and standard Platt (1999) methodology. All MCC numbers published in the thesis Chapter 5 will be regenerated. The canonical config is `configs/v7.yaml` (with `batch_size: 128` override of the committed 256).
2. **Complete the `All` dataset benchmark** (386,099 samples post-filter; Human + Non-Human combined). This is the last corpus not yet fully evaluated and is required before generating bootstrap CIs + paired Wilcoxon statistics over the four models.
3. **Post-hoc statistical analysis**: once the rerun is done, run `scripts/thesis_followups/bootstrap_ci.py` on the saved logits to produce the CI/p-value tables that replace the current "x ± σ" rows in thesis tables 17 and 18.

## PR workflow

**Open PRs against `cross_attention_lite`, not `main`.** The `main` branch is reserved for stable releases; `cross_attention_lite` is the active integration branch. Use `gh pr create --base cross_attention_lite`.

The pattern in recent history (PRs #199, #201, #204, #206) is one merge commit per topic, squashing from a task-specific working branch into `cross_attention_lite`.

## Related Repository: PhD Thesis

The sibling repository **`~/PhD`** (`/Users/sulfierry/PhD`) contains the LaTeX source for the doctoral thesis that formalizes this framework. It provides:

- **Academic narrative** (Chapters 1–6): problem formulation, literature review, data curation methodology, architectural specification (DT-Kinase Levels 1–4), experimental results, and conclusions.
- **Canonical definitions**: the "semantic screening" concept, scaffold split protocol, monotonic filter, multi-seed evaluation protocol (5 seeds), and MCC-optimal threshold calibration — all described with mathematical rigor.
- **Figures and tables** (`tex/`, `figures/`): TikZ bar charts, architecture diagrams, and comparative tables that reference results produced by this repository's `benchmark/` module.
- **Standardized model order**: DT-Kinase, ConPLex, DrugBAN, GraphBAN — used consistently in all legends, tables, and prose.

When making changes to this codebase that affect reported metrics, architectural claims, or evaluation protocols, the corresponding thesis chapters in `~/PhD/tex/` must be updated to maintain consistency.

## Environment Setup

```bash
# Option A: Conda
conda env create -f environment.yml && conda activate docktkinase

# Option B: venv (used in this repo)
python setup.py              # Creates env/, installs deps, downloads models
source activate_env.sh       # Activates venv + sets PYTHONPATH

# ESM-2 must be loaded from local repo (pip versions cause segfaults)
git clone https://github.com/facebookresearch/esm.git llm/ESM
```

The virtual environment lives in `env/`. Activate with `source env/bin/activate`.
*Note: Some baseline models (like ConPLex, DrugBAN) require their own isolated conda environments. See `setup_env.sh` inside their respective directories.*

## Running Tests

```bash
# Pytest (configured in pyproject.toml)
pytest                                    # All tests
pytest -m unit                            # Unit tests only
pytest -m "not slow"                      # Skip slow tests
pytest tests/classifier_test/             # Classifier module tests
pytest tests/test_cross_attention_model.py  # Single test file
```

Markers defined: `slow`, `integration`, `unit`, `regression`, `classifier`, `build`, `requires_gpu`, `requires_data`.

## Architecture: DT-Kinase (v7)

The production architecture is **Level 4 CNN v7**, a hierarchical model with 4 ablation levels:

| Level | Architecture | Input | Purpose |
|-------|-------------|-------|---------|
| 1 | KNN/MLP classifiers | Mean-pooled vectors (concat) | Baseline — no learned interaction |
| 2 | MLP on concatenated embeddings | Per-token matrices (mean-pooled) | Learned combination, no interaction |
| 3 | Bi-modal attention pooling | Per-token matrices | Selective aggregation, no 2D interaction |
| 4 (v7) | CNN 2D + cross-attention + attention pooling | Per-token matrices | Full model — positional interaction maps |

### Production Model: Level 4 CNN v7

- **Encoders**: ESM-2 8M (protein, frozen, 320-dim) + MoLFormer (ligand, frozen, 768-dim)
- **Adapters**: `EmbeddingAdapter` with 4-head self-attention + MLP bottleneck (prot=256, lig=512), output-layer zero-init → starts as identity
- **Projections**: Multi-head linear projections (K=8 heads, d_head=32)
- **Interaction maps**: `M_k = P_k L_k^T / √d_h`, scores preserved raw (no softmax marginalization)
- **CNN 2D**: 4 layers (8→32→64→64→64), dilation=2 only on layer 3, BatchNorm2d+GELU. No internal dropout.
- **Pooling**: Hierarchical attention (ligand axis → protein axis), single learnable query per stage.
- **Dropout**: p=0.35 applied to the pooled vector before the classifier (not inside CNN).
- **Output**: Binary classification (active/inactive, pChEMBL ≥ 6.0) via `Linear(64, 1)` + BCEWithLogitsLoss.
- **Loss**: Focal Loss (γ=2.0) with class weight `α_+ = clip(N_-/N_+, 1.0, 20.0)`.
- **Calibration**: Platt scaling (2-parameter logistic regression) fit on **validation** logits (changed from train in PR #206).
- **Threshold**: MCC-optimal on validation, 2-pass sweep (100 coarse + 100 fine ±0.05).
- **Config**: `configs/v7.yaml` (source of truth; B=128, dropout=0.35, λ=0.04, patience=15, float64).

### What is NOT the v7 benchmark

- `src/classifier/models/cross_attention_model.py` contains `CrossAttentionAffinityModel`, `MultiTaskLoss`, etc. — these are **exploratory variants**, not the thesis benchmark path.
- `src/attention_matrix/model.py` and `examples/crossattention_split_analysis/models/level5_lite/` are also experimental.
- The canonical v7 training and evaluation pipeline lives in **`benchmark/levels/level4_cnn.py`** and is entry-pointed via `run_from_config.py configs/v7.yaml`.

### Key Source Files

```
src/classifier/models/cross_attention_model.py   # Production model classes:
    CrossAttentionAffinityModel                  # Full Level 4 CNN v7
    MultiTaskLoss                                # Joint classification + regression
    CNNEncoder, CrossAttention, MultiTaskHead    # Sub-modules

benchmark/                                       # Orchestration layer:
    orchestrator.py     # BenchmarkOrchestrator — runs all levels × seeds
    levels/             # Level implementations (level1.py → level4_cnn.py)
    config.py           # Runtime config from v7.yaml
    metrics.py          # MCC, AUROC, AUPRC, F1, Precision, Recall
    splits.py           # Scaffold split loader
    visualization.py    # Result plots

scaffolds_splits/                                # Data curation module:
    splitter.py         # Bemis-Murcko scaffold split (80/10/10)
    monotonic.py        # Monotonic profile filter
    validation.py       # Split integrity checks
```

### Running the Benchmark

```bash
# Config-driven (production):
python3 run_from_config.py configs/v7.yaml --dataset non_human
python3 run_from_config.py configs/v7.yaml --dataset human
python3 run_from_config.py configs/v7.yaml --dataset all
python3 run_from_config.py configs/v7.yaml --dataset non_human --dry-run

# Shell orchestrator:
bash run_benchmark.sh
```

## Baseline Models Integration

Three SOTA baselines are evaluated under identical conditions. Each has its own directory and isolated environment:

| Model | Directory | Encoders | Publication |
|-------|-----------|----------|-------------|
| **DrugBAN** | `DrugBAN/` | CNN 1D + GCN (trained from scratch, **no PLMs**) | *Nature Machine Intelligence* |
| **GraphBAN** | `GraphBAN/` | ESM-1b + ChemBERTa + task-specific encoders | *Nature Communications* |
| **ConPLex** | `ConPLex/` | ProtBERT + Morgan fingerprints (contrastive) | *PNAS* |

**`KinBAN/`** contains an experimental fork (work in progress).

**Equitable Comparison Protocol** (formalized in the PhD thesis, Chapter 4):
- **Same data**: identical scaffold splits (train/val/test) for all models
- **Same seeds**: `[42, 123, 456, 789, 1024]` — 5 independent runs
- **Same threshold**: MCC-optimal on validation set (no test leakage)
- **Same metric**: MCC as primary, AUROC/AUPRC/F1 as secondary
- Baseline training scripts: `DrugBAN/run_dtkinase_drugban.sh`, `ConPLex/run_conplex_kinase_benchmark.sh`

## Data Layout

**Input datasets** (not in git, ~415 MB total):
- `tests/datasets/kinase_human_compounds.tsv`
- `tests/datasets/kinase_non_human_compounds.tsv`
- `tests/datasets/kinase_all_compounds.tsv`

**Kinase Scaffold Splits** (used for standard benchmarking):
- `DrugBAN/datasets/kinase/{non_human|human|all}/scaffold/{train|val|test}.csv`

**Pre-computed embeddings** are stored at:
`./results/protein_model_benchmark_{human|non_human}_v2/{embedding_name}/build/`
with subdirs: `protein_matrices/`, `ligand_matrices/`, `molformer_matrix/`, `attention_matrices/`.

## Data Curation: `scaffolds_splits/`

The production split module implements:
- **Bemis-Murcko scaffold split** (80/10/10) ensuring no scaffold leaks between partitions
- **Monotonic profile filter** removing trivially predictable kinases/compounds
- **Universal split**: single partition over the combined Human + Non-Human pool

## Legacy Split Module: `crossattention_split_analysis/`

Earlier exploration module with three split strategies (retained for reproducibility):
1. `new_compound_new_kinase` — both unseen (hardest)
2. `compound` — compound unseen, kinase may overlap
3. `random` — 80/10/10 random (allows leakage, baseline only)

**Affinity threshold**: pChEMBL ≥ 6.0 (IC₅₀ ≤ 1 μM) → active.

## Key Development Notes

- **Production config**: `configs/v7.yaml` — single source of truth for hyperparameters. Override only via env vars or CLI flags.
- **Seeds**: `[42, 123, 456, 789, 1024]` — multi-seed protocol, partitions fixed across seeds.
- **Model selection**: best validation MCC checkpoint (DT-Kinase); DrugBAN/GraphBAN use validation MCC; ConPLex uses validation MCC with threshold calibration (see `ConPLex/` scripts). All four models now use **MCC-optimal on val** for threshold selection (enforced for equitable comparison).
- **Calibration**: Platt scaling on val logits (default ON via `BENCHMARK_LEVEL4CNN_PLATT=1`); Temperature scaling also on val (opt-in via `BENCHMARK_LEVEL4CNN_TEMPERATURE=1`).
- **`MultiTaskLoss` is NOT used in v7.** Its presence in `src/classifier/` is vestigial from exploratory variants. The v7 benchmark uses pure `FocalLoss` → `BCEWithLogitsLoss`.
- **ESM loading**: `src/__init__.py` adds `llm/ESM/` to `sys.path`. **Never install ESM via pip** (causes segfaults).
- **Dataset `all`**: combines `human` + `non_human` by loading from both embedding directories. Post-filter total: 386,099 samples.
- **Embeddings are pre-computed**: ESM-2 and MoLFormer run once, stored as `.npy` matrices under `protein_matrices/` and `molformer_matrix/`, reused across all seeds and epochs.
- **Sequence limits**: MAX_SEQ_LEN=1024 for proteins (truncates C-terminal of atypically long kinases like ULK1); 512 for ligand SMILES tokens (rarely hit in practice).
- **Non-determinism**: cuDNN non-deterministic reductions are active by default. σ over seeds captures both seed-dependent initialization AND GPU reduction noise. Strict determinism via `BENCHMARK_LEVEL4CNN_DETERMINISTIC=1` (20–30% slower).
- **No cross-validation**: uses fixed scaffold split + 5-seed variance estimation (not k-fold).

## Thesis follow-up scripts

`scripts/thesis_followups/` contains non-architectural analyses for the upcoming thesis rerun:

| Script | Purpose |
|--------|---------|
| `run_platt_vs_temperature.sh` | Calibration comparison table (both now fit on val) |
| `run_pchembl_sensitivity.sh` | Activity threshold sweep τ ∈ {5.5, 6.0, 6.5, 7.0, 7.5} |
| `run_cross_species.sh` + `eval_checkpoint_on_dataset.py` | H→NH and NH→H transfer (thesis Objective 5) |
| `bootstrap_ci.py` | Post-hoc bootstrap CI + paired Wilcoxon on saved logits (no retraining) |

`scripts/count_v7_params.py` prints the trainable/frozen parameter breakdown for v7.

## Adding a New Protein Model

1. Add model name to choices in `scripts/run_complete_pipeline.py`
2. Add dimension mapping in `protein_dims` dict
3. Implement embedding strategy in `src/build/embeddings/strategies/`

---

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## Think in Code — MANDATORY

When you need to analyze, count, filter, compare, search, parse, transform, or process data: **write code** that does the work via `ctx_execute(language, code)` and `console.log()` only the answer. Do NOT read raw data into context to process mentally. Your role is to PROGRAM the analysis, not to COMPUTE it. Write robust, pure JavaScript — no npm dependencies, only Node.js built-ins (`fs`, `path`, `child_process`). Always use `try/catch`, handle `null`/`undefined`, and ensure compatibility with both Node.js and Bun. One script replaces ten tool calls and saves 100x context.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls. Each command: `{label: "descriptive header", command: "..."}`. Label becomes FTS5 chunk title — descriptive labels improve search.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
| `ctx purge` | Call the `ctx_purge` MCP tool with confirm: true. Warns before wiping the knowledge base. |

After /clear or /compact: knowledge base and session stats are preserved. Use `ctx purge` if you want to start fresh.
