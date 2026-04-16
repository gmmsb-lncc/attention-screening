# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**semantic-screening** is an open-source framework for predicting protein-ligand interactions, focusing on kinase targets. It implements the **DT-Kinase** architecture (CNN 2D + Bi-modal Cross-Attention) and orchestrates baseline evaluations against state-of-the-art models (DrugBAN, GraphBAN, ConPLex). 

The unifying scientific concept is **"semantic screening"**: predicting compound bioactivity by extracting meaning exclusively from the linear notation of sequences (amino acids for proteins, SMILES for ligands), without requiring 3D coordinates or heuristic descriptors. While DT-Kinase, GraphBAN, and ConPLex leverage pre-trained Language Models (PLMs), the framework also accommodates models trained entirely from scratch (such as DrugBAN), demonstrating that the semantic paradigm is defined by the input modality (1D primary sequences) rather than the specific encoding strategy.

**Repository**: gmmsb-lncc/semantic-screening | **License**: MIT | **Python**: 3.9+ (env uses 3.12) | **PyTorch**: 2.0+

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

## Architecture: DT-Kinase

### Parallel Pipelines

1. **Classical ML Pipeline** (`src/integrated_pipeline.py`): Generates mean-pooled vector embeddings → trains 10 models (XGBoost, LightGBM, CatBoost, RF, SVM, etc) for both classification and regression.
2. **DT-Kinase Deep Learning Pipeline** (`src/attention_matrix/`, `crossattention_split_analysis/`): Uses per-token matrix embeddings → CNN multi-scale encoders (kernels {3,5,7}) → bidirectional cross-attention → multi-task prediction (classification + regression jointly).

### Module Dependency Flow

```
Input TSV (seq_id, seq, chembl_id, smiles, pchembl_value)
    │
    ├─→ src/build/embeddings/strategies/     # ESM-2, ESM-C, SMI-TED, MoLFormer
    │
    ├─→ src/classifier/ + src/regression/    # Classical ML (uses vectors)
    │
    └─→ src/attention_matrix/model.py        # DT-Kinase (uses matrices)
            CrossAttentionModel              # Basic: single cross-attn layer
            ImprovedCrossAttentionModel      # Deep: multi-layer + FFN + GELU
        src/classifier/models/cross_attention_model.py
            CrossAttentionAffinityModel      # Full DT-Kinase with CNN encoders
            MultiTaskLoss                    # Joint classification + regression loss
```

## Baseline Models Integration

To rigorously evaluate DT-Kinase, the repository integrates several State-of-the-Art (SOTA) baseline models. Each resides in its own root directory with isolated environments to avoid dependency conflicts:

1. **DrugBAN** (`DrugBAN/`): Deep Bilinear Attention Network (MIT license).
2. **GraphBAN** (`GraphBAN/`): Graph-based Bilinear Attention Network (MIT license).
3. **ConPLex** (`ConPLex/`): Contrastive PLM-based exploration (structure-free, co-embedding with distance metrics) (MIT license).

**Evaluation Workflow**:
- We use unified **scaffold splits** (train/val/test) for fair comparison.
- Baseline datasets are usually located in `DrugBAN/datasets/kinase/{dataset}/scaffold/`.
- Training and evaluation are run via dedicated shell scripts and python wrappers in each baseline directory (e.g., `ConPLex/run_conplex_kinase_benchmark.sh`, `DrugBAN/run_dtkinase_drugban.sh`).

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

## Split Analysis Module (`crossattention_split_analysis/`)

This is the most actively developed module. Key files:

- `config.py` — `TrainingConfig` dataclass, `SUPPORTED_EMBEDDINGS`, `DATASET_PATHS`, `AVAILABLE_SCENARIOS`, thresholds
- `experiment.py` — `run_single_analysis()` (CLI entry), `run_crossattention_analysis()` (orchestrator), `run_scenario()` (single train/eval)
- `data/splits.py` — Three split strategies: `split_random`, `split_by_compound`, `split_new_compound_new_kinase`

**Three data split scenarios** (hardest → easiest):
1. `new_compound_new_kinase` — both compound AND kinase unseen in test (true generalization)
2. `compound` — compound unseen, kinase may overlap
3. `random` — random 80/10/10 split (baseline, allows data leakage)

**Affinity threshold**: pChEMBL >= 6.0 (IC50 <= 1000 nM) → active.

## Key Development Notes

- Default seeds for multi-seed experiments: `[42, 123, 456, 789, 1024]`
- Model selection (early stopping) uses **validation MCC** or **validation AUPR/AUROC** depending on the pipeline, not loss.
- `MultiTaskLoss` weights: classification=1.0, regression=0.5
- ESM loading: `src/__init__.py` adds `llm/ESM/` to `sys.path`. **Never install ESM via pip**.
- The `--use_attention` flag switches protein input from per-residue embeddings to attention matrices `[seq_len, seq_len]`.
- The dataset `all` combines `human` + `non_human` by loading from both embedding directories.

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
