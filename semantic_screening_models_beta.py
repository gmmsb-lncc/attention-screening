#!/usr/bin/env python3
"""Unified benchmark orchestrator for semantic-screening model comparison.

Coordinates the full pipeline:
  Step 0:  Verify / generate scaffold splits
  Step 0b: Verify / extract ligand vectors (if Level 2 requested)
  Step 1:  Level 1 — Fingerprint + KNN/MLP  (baseline)
  Step 2:  Level 2 — Embedding vectors + KNN/MLP
  Step 3:  Level 3 — Matrices + CNN+CrossAttention  (DT-Kinase)
  Step 4:  Comparative report and visualizations

Usage:
    python semantic_screening_models_beta.py --dataset non_human --embedding 8M
    python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 1,2
    python semantic_screening_models_beta.py --dataset non_human --embedding 8M --levels 3 --epochs 100
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_BASE_PATH = "./results/protein_model_benchmark_{dataset_type}_v2"
DEFAULT_SCAFFOLD_SPLIT_DIR = "scaffolds_splits/output"

SUPPORTED_EMBEDDINGS = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}

LEVEL_LABELS = {
    "level1_fp_knn": "Level 1 (FP+KNN)",
    "level1_fp_mlp": "Level 1 (FP+MLP)",
    "level2_emb_knn": "Level 2 (Emb+KNN)",
    "level2_emb_mlp": "Level 2 (Emb+MLP)",
    "level3_cnn": "Level 3 (CNN)",
    "level4_cnn_ca": "Level 4 (CNN+CA)",
    "level5_lite": "Level 5 (Lite)",
}

METRICS_ORDER = ["accuracy", "mcc", "f1", "precision", "recall", "auc"]

# Plotting palette (colorblind-friendly)
LEVEL_COLORS = {
    "level1_fp_knn": "#1b9e77",
    "level1_fp_mlp": "#66c2a5",
    "level2_emb_knn": "#7570b3",
    "level2_emb_mlp": "#a6a3d9",
    "level3_cnn": "#d95f02",
    "level4_cnn_ca": "#e7298a",
    "level5_lite": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class BenchmarkProgress:
    """Global progress tracker with nested step/substep display."""

    def __init__(self, levels: List[int], dataset: str, embedding: str):
        self.dataset = dataset
        self.embedding = embedding
        self.step_timings: Dict[str, float] = {}
        self._step_start: float = 0.0

        # Build ordered list of steps
        self.steps = ["Step 0: Scaffold Splits"]
        if 2 in levels:
            self.steps.append("Step 0b: Ligand Vectors")
        if 1 in levels:
            self.steps.append("Step 1: Level 1 (FP+KNN/MLP)")
        if 2 in levels:
            self.steps.append("Step 2: Level 2 (Emb+KNN/MLP)")
        if 3 in levels:
            self.steps.append("Step 3: Level 3 (CNN)")
        if 4 in levels:
            self.steps.append("Step 4: Level 4 (CNN+CA)")
        if 5 in levels:
            self.steps.append("Step 5: Level 5-Lite")
        self.steps.append("Report + Visualizations")

        self.total = len(self.steps)
        self.current_idx = 0
        self.global_bar = tqdm(
            total=self.total,
            desc=f"Benchmark {dataset}/{embedding}",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} steps "
                "[{elapsed}<{remaining}, {postfix}]"
            ),
            position=0,
            leave=True,
            colour="green",
        )
        self.global_bar.set_postfix_str(self.steps[0])

    def begin_step(self, step_name: str) -> None:
        """Mark the start of a step (prints banner + updates global bar)."""
        self._step_start = time.time()
        self.global_bar.set_postfix_str(step_name)
        # Print a visible banner below the bar
        tqdm.write("")
        tqdm.write("=" * 70)
        tqdm.write(f"[{self.current_idx}/{self.total}] {step_name}")
        tqdm.write("=" * 70)

    def end_step(self, step_name: str) -> None:
        """Mark step completion, advance global bar."""
        elapsed = time.time() - self._step_start
        self.step_timings[step_name] = elapsed
        self.current_idx += 1
        self.global_bar.update(1)
        mins, secs = divmod(int(elapsed), 60)
        tqdm.write(f"  -> {step_name} done in {mins}m{secs:02d}s")

    def close(self, total_elapsed: float) -> None:
        """Print final summary and close bars."""
        self.global_bar.set_postfix_str("COMPLETE")
        self.global_bar.close()
        tqdm.write("")
        tqdm.write("=" * 70)
        tqdm.write("BENCHMARK TIMING SUMMARY")
        tqdm.write("=" * 70)
        for name, secs in self.step_timings.items():
            m, s = divmod(int(secs), 60)
            h, m = divmod(m, 60)
            if h:
                tqdm.write(f"  {name:<42s}  {h}h{m:02d}m{s:02d}s")
            else:
                tqdm.write(f"  {name:<42s}  {m}m{s:02d}s")
        tqdm.write("-" * 70)
        h, rem = divmod(int(total_elapsed), 3600)
        m, s = divmod(rem, 60)
        tqdm.write(f"  {'TOTAL':<42s}  {h}h{m:02d}m{s:02d}s")
        tqdm.write("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified benchmark: scaffold split → models → comparative report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--dataset", required=True, choices=["human", "non_human", "all"],
                    help="Dataset to benchmark (all = human + non_human combined)")
    p.add_argument("--embedding", default="8M", choices=["8M", "150M", "650M"],
                    help="ESM-2 model shorthand (default: 8M)")

    # Level selection
    p.add_argument("--levels", default="1,2,3",
                    help="Comma-separated levels to run: 1=FP, 2=Emb, 3=CNN, 4=CNN+CA, 5=Level5-Lite "
                         "(default: 1,2,3)")

    # Output
    p.add_argument("--output_dir", default=None,
                    help="Root results dir (default: ./results/benchmark_{dataset}_{embedding})")
    p.add_argument("--scaffold_split_dir", default=DEFAULT_SCAFFOLD_SPLIT_DIR,
                    help=f"Scaffold split dir (default: {DEFAULT_SCAFFOLD_SPLIT_DIR})")

    # Reproducibility
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Seeds for multi-seed runs (default: from config.DEFAULT_SEEDS)")

    # Flags
    p.add_argument("--force", action="store_true",
                    help="Force recalculation of all levels")
    p.add_argument("--force_split", action="store_true",
                    help="Force regeneration of scaffold splits")

    # Level 3/4 hyperparameters
    p.add_argument("--epochs", type=int, default=500,
                    help="Max epochs for Level 3/4 (default: 500)")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for Level 3/4 (default: 32)")
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience (default: 5, 0 to disable)")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                    help="Learning rate for Level 3/4 (default: 1e-4)")

    # Debug
    p.add_argument("--debug", action="store_true",
                    help="Debug mode (verbose output)")

    return p


def parse_levels(levels_str: str) -> List[int]:
    """Parse '1,2,3' into [1, 2, 3]."""
    try:
        levels = sorted(set(int(x.strip()) for x in levels_str.split(",")))
        for lv in levels:
            if lv not in (1, 2, 3, 4, 5):
                raise ValueError(f"Invalid level: {lv}. Valid: 1,2,3,4,5")
        return levels
    except ValueError as e:
        print(f"ERROR: Invalid --levels value: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 0: Scaffold splits
# ---------------------------------------------------------------------------

def _exists_tsv_or_gz(path: str) -> str:
    """Check if a .tsv file exists, also checking .tsv.gz variant.

    Returns the path that exists (preferring uncompressed), or empty string.
    """
    if os.path.exists(path):
        return path
    gz = path + ".gz" if not path.endswith(".gz") else path
    if os.path.exists(gz):
        return gz
    return ""


def ensure_scaffold_splits(
    dataset: str,
    scaffold_split_dir: str,
    force_split: bool,
) -> bool:
    """Verify or generate scaffold splits. Returns True on success."""
    # For "all", we need both human and non_human splits
    datasets_to_check = ["human", "non_human"] if dataset == "all" else [dataset]

    scenario_dir = os.path.join(scaffold_split_dir, "scenarios", "Sc")
    all_found = True
    found_paths = []

    for ds in datasets_to_check:
        train = _exists_tsv_or_gz(os.path.join(scenario_dir, f"{ds}_train.tsv"))
        val = _exists_tsv_or_gz(os.path.join(scenario_dir, f"{ds}_val.tsv"))
        test = _exists_tsv_or_gz(os.path.join(scaffold_split_dir, f"{ds}_test.tsv"))
        if train and val and test:
            found_paths.extend([
                ("train", ds, train), ("val", ds, val), ("test", ds, test),
            ])
        else:
            all_found = False

    if all_found and not force_split:
        print(f"  [OK] Scaffold splits found:")
        for label, ds, path in found_paths:
            print(f"       {ds} {label}: {path}")
        return True

    reason = "--force_split requested" if force_split else "missing split files"
    print(f"  [{reason}] Generating scaffold splits...")

    cmd = [
        sys.executable, "scaffold_split.py",
        "--output-dir", scaffold_split_dir,
        "--scenarios", "Sc",
    ]
    print(f"  Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: scaffold_split.py failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print("  ERROR: scaffold_split.py not found")
        return False


# ---------------------------------------------------------------------------
# Step 0b: Ligand vectors
# ---------------------------------------------------------------------------

def _extract_ligand_vectors(
    matrix_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Mean-pool MoLFormer per-token matrices into ligand vectors.

    Reads {chembl_id}_matrix.npy or {chembl_id}_molformer_matrix.npy 
    (shape [n_tokens, 768]) and writes {chembl_id}_embedding.npy (shape [768]).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for files with both patterns
    matrix_files = sorted(matrix_dir.glob("*_matrix.npy"))
    molformer_files = sorted(matrix_dir.glob("*_molformer_matrix.npy"))
    
    # Combine and deduplicate (prefer _matrix.npy over _molformer_matrix.npy)
    all_files = {}
    for mf in matrix_files:
        chembl_id = mf.stem.replace("_matrix", "")
        all_files[chembl_id] = mf
    for mf in molformer_files:
        chembl_id = mf.name.replace("_molformer_matrix.npy", "")
        if chembl_id not in all_files:
            all_files[chembl_id] = mf
    
    matrix_files = sorted(all_files.values(), key=lambda x: x.name)
    
    if not matrix_files:
        print(f"  WARNING: no matrix files found in {matrix_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    processed = skipped = errors = 0
    for mf in matrix_files:
        # Extract chembl_id from filename (handle both patterns)
        if mf.name.endswith("_molformer_matrix.npy"):
            chembl_id = mf.name.replace("_molformer_matrix.npy", "")
        else:
            chembl_id = mf.stem.replace("_matrix", "")
        
        out_path = output_dir / f"{chembl_id}_embedding.npy"
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            mat = np.load(mf)
            if mat.ndim != 2:
                print(f"  WARNING: unexpected shape {mat.shape} for {mf.name}, skipping")
                errors += 1
                continue
            np.save(out_path, mat.mean(axis=0).astype(np.float32))
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {mf.name}: {e}")
            errors += 1
    return {"processed": processed, "skipped": skipped, "errors": errors}


def ensure_ligand_vectors(
    dataset: str,
    embedding_name: str,
    force: bool,
) -> bool:
    """Verify or extract ligand vectors. Returns True on success."""
    # For "all", process both human and non_human
    datasets_to_process = ["human", "non_human"] if dataset == "all" else [dataset]
    all_ok = True

    for ds in datasets_to_process:
        build_dir = Path(
            EMBEDDING_BASE_PATH.format(dataset_type=ds),
            embedding_name,
            "build",
        )
        molformer_dir = build_dir / "molformer_matrix"
        vector_dir = build_dir / "ligand_embeddings"

        if vector_dir.exists() and any(vector_dir.glob("*_embedding.npy")) and not force:
            n_files = len(list(vector_dir.glob("*_embedding.npy")))
            print(f"  [OK] Ligand vectors ({ds}): {vector_dir} ({n_files} files)")
            continue

        if not molformer_dir.exists():
            print(f"  WARNING: MoLFormer matrix dir not found ({ds}): {molformer_dir}")
            print(f"           Level 2 embedding features will not include ligand vectors.")
            all_ok = False
            continue

        print(f"  Extracting ligand vectors ({ds}) from {molformer_dir}...")

        stats = _extract_ligand_vectors(molformer_dir, vector_dir, force=force)
        print(
            f"  Done ({ds}): {stats['processed']} extracted, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        if stats["processed"] + stats["skipped"] == 0:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Step 1: Level 1 — Fingerprint baseline
# ---------------------------------------------------------------------------

def _run_level_multiseed(
    dataset: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
    feature_type: str,
    embedding_name: str = None,
) -> Optional[Dict]:
    """Run Level 1 or 2 across multiple seeds, aggregate mean+std.

    Each seed re-trains KNN/MLP with different random init. The scaffold
    splits are fixed (precomputed), so only model randomness varies.
    """
    from split_comparison_analysis import run_single_dataset

    seed_results_per_model: Dict[str, Dict[str, List[float]]] = {}

    for i, seed in enumerate(seeds):
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        tqdm.write(f"  Seed {i+1}/{len(seeds)}: {seed}")

        result = run_single_dataset(
            dataset_type=dataset,
            output_dir=seed_dir,
            force=force,
            seed=seed,
            scenarios=["scaffold"],
            scaffold_split_dir=scaffold_split_dir,
            feature_type=feature_type,
            embedding_name=embedding_name,
        )

        # If cached, load from disk
        if result is None:
            result = _load_split_comparison_results(seed_dir)

        if result is None:
            tqdm.write(f"    WARNING: seed {seed} returned no results.")
            continue

        # Find scaffold scenario
        sc_key = None
        for key in result:
            if "scaffold" in key.replace("\n", " ").lower():
                sc_key = key
                break
        if sc_key is None and result:
            sc_key = next(iter(result))
        if sc_key is None:
            continue

        sc = result[sc_key]
        for model in ["KNN", "MLP"]:
            if model not in sc:
                continue
            if model not in seed_results_per_model:
                seed_results_per_model[model] = {}
            for metric in METRICS_ORDER:
                val = sc[model].get(metric)
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    seed_results_per_model[model].setdefault(metric, []).append(float(val))

    if not seed_results_per_model:
        return None

    # Aggregate: build result dict matching the format expected by aggregate_benchmark_metrics
    scaffold_key = "Split by Scaffold"
    aggregated = {}
    for model, metrics_dict in seed_results_per_model.items():
        agg = {}
        for metric, values in metrics_dict.items():
            arr = np.array(values)
            agg[metric] = float(np.mean(arr))
            agg[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        agg["n_seeds"] = len(next(iter(metrics_dict.values())))
        aggregated[model] = agg

    # Save aggregated JSON
    os.makedirs(output_dir, exist_ok=True)
    agg_path = os.path.join(output_dir, "split_comparison_results.json")
    with open(agg_path, "w") as f:
        json.dump({
            "dataset": dataset,
            "feature_type": feature_type,
            "embedding_name": embedding_name,
            "seeds": seeds,
            "results": {scaffold_key: aggregated},
        }, f, indent=2)
    tqdm.write(f"  Aggregated results saved: {agg_path}")

    return {scaffold_key: aggregated}


def run_level1(
    dataset: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
) -> Optional[Dict]:
    """Run Level 1: Fingerprint + KNN/MLP (multi-seed). Returns results dict or None."""
    level_dir = os.path.join(output_dir, "level1_fingerprint", dataset)
    print(f"  Output: {level_dir}")

    return _run_level_multiseed(
        dataset=dataset,
        output_dir=level_dir,
        scaffold_split_dir=scaffold_split_dir,
        seeds=seeds,
        force=force,
        feature_type="fingerprint",
    )


# ---------------------------------------------------------------------------
# Step 2: Level 2 — Embedding vectors
# ---------------------------------------------------------------------------

def run_level2(
    dataset: str,
    embedding_name: str,
    embedding_short: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
) -> Optional[Dict]:
    """Run Level 2: Embedding vectors + KNN/MLP (multi-seed). Returns results dict or None."""
    level_dir = os.path.join(output_dir, f"level2_embedding_{embedding_short}", dataset)
    print(f"  Output: {level_dir}")

    return _run_level_multiseed(
        dataset=dataset,
        output_dir=level_dir,
        scaffold_split_dir=scaffold_split_dir,
        seeds=seeds,
        force=force,
        feature_type="embedding",
        embedding_name=embedding_name,
    )


def _load_split_comparison_results(level_dir: str) -> Optional[Dict]:
    """Load cached split_comparison_results.json and reconstruct results dict."""
    json_path = os.path.join(level_dir, "split_comparison_results.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("results")
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Step 3: Level 3 — CNN (optionally + CrossAttention)
# ---------------------------------------------------------------------------

def run_level3(
    dataset: str,
    embedding_name: str,
    embedding_short: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
    epochs: int,
    batch_size: int,
    patience: Optional[int],
    learning_rate: float,
    num_cross_attn_layers: int = 0,
) -> Optional[Dict]:
    """Run Level 3: CNN on embedding matrices (optionally + CrossAttention).

    Args:
        num_cross_attn_layers: 0 = CNN-only (default), >=1 = add cross-attention.
    """
    from crossattention_split_analysis.experiment import run_single_analysis

    if num_cross_attn_layers > 0:
        tag = "level4_cnn_ca"
    else:
        tag = "level3_cnn"
    level_dir = os.path.join(output_dir, f"{tag}_{embedding_short}")
    print(f"  Output: {level_dir}")
    print(f"  Cross-attention layers: {num_cross_attn_layers}"
          f" ({'CNN+CA' if num_cross_attn_layers > 0 else 'CNN-only'})")

    results = run_single_analysis(
        embedding_name=embedding_name,
        dataset_type=dataset,
        output_dir=level_dir,
        seeds=seeds,
        force=force,
        scenarios=["scaffold"],
        num_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_cross_attn_layers=num_cross_attn_layers,
        classification_only=True,
        use_molformer_ligand=True,
        scaffold_split_dir=scaffold_split_dir,
        model_variant="cnn_crossattn",
    )

    # If cached, try to load from disk
    if results is None:
        results = _load_crossattention_results(level_dir, dataset, embedding_short)

    return results


def _load_crossattention_results(
    level_dir: str,
    dataset: str,
    embedding_short: str,
) -> Optional[Dict]:
    """Load cached crossattention_analysis_results.json."""
    full_name = SUPPORTED_EMBEDDINGS.get(embedding_short, embedding_short)
    short_name = full_name.replace("esm2_", "").replace("_UR50D", "")
    prefix = f"{dataset}_molformer_{short_name}_"
    json_path = os.path.join(level_dir, f"{prefix}crossattention_analysis_results.json")

    if not os.path.exists(json_path):
        # Try finding any matching file
        candidates = glob.glob(os.path.join(level_dir, "*crossattention_analysis_results.json"))
        if candidates:
            json_path = candidates[0]
        else:
            return None

    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("model_results")
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Step 5: Level 5-Lite — Transformer + Cross-Attention
# ---------------------------------------------------------------------------

def run_level5_lite(
    dataset: str,
    embedding_name: str,
    embedding_short: str,
    output_dir: str,
    scaffold_split_dir: str,
    seeds: List[int],
    force: bool,
    epochs: int,
    batch_size: int,
    patience: Optional[int],
    learning_rate: float,
) -> Optional[Dict]:
    """Run Level 5-Lite: Transformer encoders + Bidirectional Cross-Attention.
    
    This level uses:
    - Pre-calculated ESM-2 protein matrices (per-residue)
    - Pre-calculated MoLFormer ligand matrices (per-token)
    - Transformer encoders for both modalities
    - Bidirectional cross-attention for interaction modeling
    - Attention pooling for sequence-to-vector aggregation
    
    Returns results dict or None.
    """
    from crossattention_split_analysis.experiment import run_single_analysis
    
    level_dir = os.path.join(output_dir, f"level5_lite_{embedding_short}")
    tqdm.write(f"  Output: {level_dir}")
    tqdm.write(f"  Architecture: Transformer + Cross-Attention (Level 5-Lite)")
    
    results = run_single_analysis(
        embedding_name=embedding_name,
        dataset_type=dataset,
        output_dir=level_dir,
        seeds=seeds,
        force=force,
        scenarios=["scaffold"],
        num_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=512,
        num_encoder_layers=2,
        num_cross_attn_layers=1,
        num_heads=8,
        dropout=0.1,
        classifier_dropout=0.3,
        classification_only=True,
        use_molformer_ligand=True,
        scaffold_split_dir=scaffold_split_dir,
        model_variant="level5_lite",
    )
    
    # If cached, try to load from disk
    if results is None:
        results = _load_crossattention_results(level_dir, dataset, embedding_short)
    
    return results


# ---------------------------------------------------------------------------
# Step 6: Aggregate metrics
# ---------------------------------------------------------------------------

def _extract_metric(results_dict: Dict, model_key: str, metric: str) -> Optional[float]:
    """Safely extract a metric value from a model results block."""
    if not results_dict or model_key not in results_dict:
        return None
    val = results_dict[model_key].get(metric)
    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return None


def _extract_metric_std(results_dict: Dict, model_key: str, metric: str) -> Optional[float]:
    """Extract std for a metric from multi-seed results."""
    if not results_dict or model_key not in results_dict:
        return None
    val = results_dict[model_key].get(f"{metric}_std")
    if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
        return float(val)
    return None


def _find_scaffold_scenario_key(results: Dict) -> Optional[str]:
    """Find the scaffold scenario key in results (may vary across modules)."""
    if not results:
        return None

    for key in results:
        normalized = key.replace("\n", " ").lower()
        if "scaffold" in normalized:
            return key

    # Fallback: return first key with warning
    if results:
        first_key = next(iter(results))
        warnings.warn(
            f"No 'scaffold' scenario found in results; falling back to '{first_key}'"
        )
        return first_key
    return None


def aggregate_benchmark_metrics(
    level1_results: Optional[Dict],
    level2_results: Optional[Dict],
    level3_results: Optional[Dict],
    level3_key: str = "level3_cnn",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Aggregate metrics from all levels into a unified dict.

    Args:
        level3_key: "level3_cnn" (CNN-only) or "level3_cnn_ca" (CNN+CrossAttention).

    Returns:
        {model_key: {metric: value, metric_std: value, ...}}
    """
    aggregated = {}

    # Level 1 — Fingerprint
    if level1_results:
        sc_key = _find_scaffold_scenario_key(level1_results)
        if sc_key and sc_key in level1_results:
            sc = level1_results[sc_key]
            for model_key, label_key in [("KNN", "level1_fp_knn"), ("MLP", "level1_fp_mlp")]:
                row = {}
                for m in METRICS_ORDER:
                    row[m] = _extract_metric(sc, model_key, m)
                    row[f"{m}_std"] = _extract_metric_std(sc, model_key, m)
                aggregated[label_key] = row

    # Level 2 — Embedding
    if level2_results:
        sc_key = _find_scaffold_scenario_key(level2_results)
        if sc_key and sc_key in level2_results:
            sc = level2_results[sc_key]
            for model_key, label_key in [("KNN", "level2_emb_knn"), ("MLP", "level2_emb_mlp")]:
                row = {}
                for m in METRICS_ORDER:
                    row[m] = _extract_metric(sc, model_key, m)
                    row[f"{m}_std"] = _extract_metric_std(sc, model_key, m)
                aggregated[label_key] = row

    # Level 3 — CNN (or CNN+CA)
    if level3_results:
        sc_key = _find_scaffold_scenario_key(level3_results)
        if sc_key:
            sc_data = level3_results[sc_key]
            # level3 results can be nested (scenario -> model -> metrics) or flat (scenario -> metrics)
            if isinstance(sc_data, dict):
                # Check if it looks like {metric: value} directly
                if "accuracy" in sc_data or "mcc" in sc_data:
                    metrics_block = sc_data
                else:
                    # Nested: find CNN+CrossAttn key
                    metrics_block = None
                    for k, v in sc_data.items():
                        if isinstance(v, dict) and ("mcc" in v or "accuracy" in v):
                            metrics_block = v
                            break
                    if metrics_block is None:
                        metrics_block = sc_data

                row = {}
                for m in METRICS_ORDER:
                    val = metrics_block.get(m)
                    if val is not None and isinstance(val, (int, float)):
                        row[m] = float(val) if not np.isnan(val) else None
                    else:
                        row[m] = None
                    std_val = metrics_block.get(f"{m}_std")
                    if std_val is not None and isinstance(std_val, (int, float)):
                        row[f"{m}_std"] = float(std_val) if not np.isnan(std_val) else None
                    else:
                        row[f"{m}_std"] = None
                aggregated[level3_key] = row

    return aggregated


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_comparison_table(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
) -> None:
    """Print a formatted comparison table to the terminal."""

    print("\n" + "=" * 90)
    print(f"BENCHMARK COMPARISON: {dataset} / ESM-2 {embedding_short} / Scaffold Split")
    print("=" * 90)

    header = f"{'Model':<22s}"
    for m in METRICS_ORDER:
        header += f"  {m.upper():>10s}"
    print(header)
    print("-" * 90)

    for model_key in [
        "level1_fp_knn", "level1_fp_mlp",
        "level2_emb_knn", "level2_emb_mlp",
        "level3_cnn", "level4_cnn_ca", "level5_lite",
    ]:
        if model_key not in aggregated:
            continue
        row = aggregated[model_key]
        label = LEVEL_LABELS[model_key]
        line = f"{label:<22s}"
        for m in METRICS_ORDER:
            val = row.get(m)
            std = row.get(f"{m}_std")
            if val is None:
                line += f"  {'N/A':>10s}"
            elif std is not None and std > 0:
                cell = f"{val:.3f}\u00b1{std:.3f}"
                line += f"  {cell:>10s}"
            else:
                line += f"  {val:>10.4f}"
        print(line)

    print("=" * 90)


# ---------------------------------------------------------------------------
# Step 4: Save JSON
# ---------------------------------------------------------------------------

def save_benchmark_json(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
    levels: List[int],
    seeds: List[int],
    elapsed_seconds: float,
) -> str:
    """Save benchmark_comparison.json and return its path."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "embedding": embedding_short,
            "embedding_full": SUPPORTED_EMBEDDINGS.get(embedding_short, embedding_short),
            "split": "scaffold",
            "levels_executed": levels,
            "seeds": seeds,
            "elapsed_seconds": round(elapsed_seconds, 1),
        },
        "results": {},
    }

    for model_key, row in aggregated.items():
        label = LEVEL_LABELS.get(model_key, model_key)
        entry = {"label": label}
        for m in METRICS_ORDER:
            val = row.get(m)
            entry[m] = round(val, 6) if val is not None else None
            std = row.get(f"{m}_std")
            entry[f"{m}_std"] = round(std, 6) if std is not None else None
        output["results"][model_key] = entry

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_comparison.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBenchmark JSON saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _available_models(aggregated: Dict) -> List[str]:
    """Return model keys that have at least one non-None metric."""
    order = ["level1_fp_knn", "level1_fp_mlp", "level2_emb_knn", "level2_emb_mlp",
             "level3_cnn", "level4_cnn_ca", "level5_lite"]
    available = []
    for k in order:
        if k in aggregated:
            row = aggregated[k]
            if any(row.get(m) is not None for m in METRICS_ORDER):
                available.append(k)
    return available


def plot_grouped_bar_chart(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> Optional[str]:
    """Grouped bar chart: metrics on x-axis, bars grouped by model."""
    models = _available_models(aggregated)
    if not models:
        return None

    n_metrics = len(METRICS_ORDER)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    all_vals = []
    for i, mk in enumerate(models):
        row = aggregated[mk]
        vals = [row.get(m) if row.get(m) is not None else np.nan for m in METRICS_ORDER]
        stds = [row.get(f"{m}_std") or 0.0 for m in METRICS_ORDER]
        all_vals.extend(v for v in vals if not np.isnan(v))
        offset = (i - n_models / 2 + 0.5) * bar_width

        bars = ax.bar(
            x + offset,
            [v if not np.isnan(v) else 0 for v in vals],
            bar_width * 0.9,
            yerr=stds if any(s > 0 for s in stds) else None,
            capsize=3,
            label=LEVEL_LABELS[mk],
            color=LEVEL_COLORS[mk],
            edgecolor="white",
            linewidth=0.5,
        )

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val != 0:
                y_pos = max(bar.get_height(), 0) + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER], fontsize=11)
    y_min = min(all_vals) if all_vals else 0
    ax.set_ylim(min(0, y_min - 0.05), 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Comparison — {dataset} / ESM-2 {embedding_short} / Scaffold Split",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "benchmark_grouped_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def plot_radar_chart(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> Optional[str]:
    """Radar (spider) chart comparing all models across metrics."""
    models = _available_models(aggregated)
    if not models:
        return None

    metrics = METRICS_ORDER
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    all_radar_vals = []
    for mk in models:
        row = aggregated[mk]
        vals = [row.get(m) if row.get(m) is not None else 0.0 for m in metrics]
        all_radar_vals.extend(vals)
        vals += vals[:1]  # close polygon
        ax.plot(angles, vals, "o-", linewidth=2, label=LEVEL_LABELS[mk],
                color=LEVEL_COLORS[mk], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=LEVEL_COLORS[mk])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    r_min = min(all_radar_vals) if all_radar_vals else 0
    ax.set_ylim(min(0, r_min - 0.05), 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.6)
    ax.set_title(
        f"Radar — {dataset} / ESM-2 {embedding_short} / Scaffold Split",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "benchmark_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def plot_metric_heatmap(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> Optional[str]:
    """Heatmap: models (rows) vs metrics (columns), color-coded by value."""
    models = _available_models(aggregated)
    if not models:
        return None

    n_models = len(models)
    n_metrics = len(METRICS_ORDER)
    matrix = np.full((n_models, n_metrics), np.nan)

    for i, mk in enumerate(models):
        row = aggregated[mk]
        for j, m in enumerate(METRICS_ORDER):
            val = row.get(m)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(10, max(3, n_models * 0.8 + 1.5)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(n_models):
        for j in range(n_metrics):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=11, color="gray")
            else:
                std = aggregated[models[i]].get(f"{METRICS_ORDER[j]}_std")
                txt = f"{val:.3f}"
                if std is not None and std > 0:
                    txt += f"\n±{std:.3f}"
                text_color = "white" if val < 0.4 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER], fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([LEVEL_LABELS[mk] for mk in models], fontsize=11)

    ax.set_title(
        f"Performance Heatmap — {dataset} / ESM-2 {embedding_short} / Scaffold Split",
        fontsize=13, fontweight="bold", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Score", fontsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "benchmark_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def plot_mcc_ranking(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> Optional[str]:
    """Horizontal bar chart ranking models by MCC (the primary selection metric)."""
    models = _available_models(aggregated)
    if not models:
        return None

    # Sort by MCC descending
    items = []
    for mk in models:
        mcc = aggregated[mk].get("mcc")
        std = aggregated[mk].get("mcc_std")
        if mcc is not None:
            items.append((mk, mcc, std or 0.0))
    if not items:
        return None

    items.sort(key=lambda x: x[1])  # ascending for horizontal bars (top = best)

    fig, ax = plt.subplots(figsize=(9, max(3, len(items) * 0.8 + 1)))

    y_pos = np.arange(len(items))
    mccs = [x[1] for x in items]
    stds = [x[2] for x in items]
    colors = [LEVEL_COLORS[x[0]] for x in items]
    labels = [LEVEL_LABELS[x[0]] for x in items]

    bars = ax.barh(y_pos, mccs, xerr=stds if any(s > 0 for s in stds) else None,
                   capsize=4, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Value labels
    for bar, mcc_val, std_val in zip(bars, mccs, stds):
        txt = f"{mcc_val:.3f}"
        if std_val > 0:
            txt += f" \u00b1 {std_val:.3f}"
        x_pos = max(bar.get_width(), 0) + 0.01
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                txt, ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("MCC", fontsize=12)
    x_min = min(mccs) if mccs else 0
    x_max = max(mccs) if mccs else 0
    ax.set_xlim(min(0, x_min - 0.05), max(x_max * 1.25, 0.1))
    ax.set_title(
        f"MCC Ranking — {dataset} / ESM-2 {embedding_short} / Scaffold Split",
        fontsize=13, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()
    path = os.path.join(output_dir, "benchmark_mcc_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def plot_level_comparison_strip(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> Optional[str]:
    """Strip chart showing each metric as a separate panel, with models on y-axis.

    This gives the clearest per-metric comparison with exact values visible.
    """
    models = _available_models(aggregated)
    if not models or len(METRICS_ORDER) == 0:
        return None

    n_metrics = len(METRICS_ORDER)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, max(3, len(models) * 0.7 + 1)),
                             sharey=True)
    if n_metrics == 1:
        axes = [axes]

    y_pos = np.arange(len(models))

    for j, metric in enumerate(METRICS_ORDER):
        ax = axes[j]
        vals = []
        stds = []
        colors = []
        for mk in models:
            v = aggregated[mk].get(metric)
            s = aggregated[mk].get(f"{metric}_std")
            vals.append(v if v is not None else np.nan)
            stds.append(s if s is not None else 0.0)
            colors.append(LEVEL_COLORS[mk])

        plot_vals = [v if not np.isnan(v) else 0.0 for v in vals]
        ax.barh(y_pos, plot_vals,
                xerr=stds if any(s > 0 for s in stds) else None,
                capsize=3, color=colors, edgecolor="white",
                linewidth=0.5, height=0.55)

        for i, (v, s) in enumerate(zip(vals, stds)):
            if not np.isnan(v) and v != 0:
                txt = f"{v:.3f}"
                ax.text(max(v, 0) + 0.005, i, txt, ha="left", va="center", fontsize=8)

        v_min = min((v for v in vals if not np.isnan(v)), default=0)
        ax.set_xlim(min(0, v_min - 0.05), 1.05)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([LEVEL_LABELS[mk] for mk in models], fontsize=10)

    fig.suptitle(
        f"Per-Metric Comparison — {dataset} / ESM-2 {embedding_short} / Scaffold Split",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, "benchmark_per_metric.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")
    return path


def generate_all_visualizations(
    aggregated: Dict[str, Dict],
    dataset: str,
    embedding_short: str,
    output_dir: str,
) -> List[str]:
    """Generate all benchmark visualizations. Returns list of saved paths."""
    paths = []

    print("\n[Visualizations]")

    for plot_fn in [
        plot_grouped_bar_chart,
        plot_radar_chart,
        plot_metric_heatmap,
        plot_mcc_ranking,
        plot_level_comparison_strip,
    ]:
        try:
            p = plot_fn(aggregated, dataset, embedding_short, output_dir)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  WARNING: {plot_fn.__name__} failed: {e}")

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    dataset = args.dataset
    embedding_short = args.embedding
    embedding_name = SUPPORTED_EMBEDDINGS[embedding_short]
    scaffold_split_dir = args.scaffold_split_dir

    # Seeds
    if args.seeds:
        seeds = args.seeds
    else:
        from crossattention_split_analysis.config import DEFAULT_SEEDS
        seeds = DEFAULT_SEEDS

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"./results/benchmark_{dataset}_{embedding_short}"

    # Patience
    patience = args.patience if args.patience > 0 else None

    force = args.force

    print("=" * 70)
    print("SEMANTIC SCREENING — UNIFIED BENCHMARK")
    print("=" * 70)
    print(f"  Dataset:          {dataset}")
    print(f"  Embedding:        {embedding_short} ({embedding_name})")
    print(f"  Levels:           {levels}")
    print(f"  Seeds:            {seeds}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Scaffold splits:  {scaffold_split_dir}")
    print(f"  Force:            {force}")
    if 3 in levels or 4 in levels:
        print(f"  DL epochs:        {args.epochs}")
        print(f"  DL batch_size:    {args.batch_size}")
        print(f"  DL patience:      {patience}")
        print(f"  DL learning_rate: {args.learning_rate}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # Initialize global progress tracker
    progress = BenchmarkProgress(levels, dataset, embedding_short)

    # -----------------------------------------------------------------------
    # Step 0: Scaffold splits
    # -----------------------------------------------------------------------
    step_name = "Step 0: Scaffold Splits"
    progress.begin_step(step_name)
    if not ensure_scaffold_splits(dataset, scaffold_split_dir, args.force_split):
        tqdm.write("FATAL: Cannot proceed without scaffold splits.")
        progress.global_bar.close()
        sys.exit(1)
    progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 0b: Ligand vectors (if level 2)
    # -----------------------------------------------------------------------
    if 2 in levels:
        step_name = "Step 0b: Ligand Vectors"
        progress.begin_step(step_name)
        ensure_ligand_vectors(dataset, embedding_name, force)
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 1: Level 1
    # -----------------------------------------------------------------------
    level1_results = None
    if 1 in levels:
        step_name = "Step 1: Level 1 (FP+KNN/MLP)"
        progress.begin_step(step_name)
        level1_results = run_level1(
            dataset=dataset,
            output_dir=output_dir,
            scaffold_split_dir=scaffold_split_dir,
            seeds=seeds,
            force=force,
        )
        if level1_results:
            tqdm.write("  Level 1 completed successfully.")
        else:
            tqdm.write("  WARNING: Level 1 returned no results.")
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 2: Level 2
    # -----------------------------------------------------------------------
    level2_results = None
    if 2 in levels:
        step_name = "Step 2: Level 2 (Emb+KNN/MLP)"
        progress.begin_step(step_name)
        level2_results = run_level2(
            dataset=dataset,
            embedding_name=embedding_name,
            embedding_short=embedding_short,
            output_dir=output_dir,
            scaffold_split_dir=scaffold_split_dir,
            seeds=seeds,
            force=force,
        )
        if level2_results:
            tqdm.write("  Level 2 completed successfully.")
        else:
            tqdm.write("  WARNING: Level 2 returned no results.")
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 3: Level 3 — CNN-only
    # -----------------------------------------------------------------------
    level3_results = None
    if 3 in levels:
        step_name = "Step 3: Level 3 (CNN)"
        progress.begin_step(step_name)
        tqdm.write(f"  Seeds to run: {seeds} ({len(seeds)} total)")
        tqdm.write(f"  Max epochs per seed: {args.epochs}, patience: {patience}")
        level3_results = run_level3(
            dataset=dataset,
            embedding_name=embedding_name,
            embedding_short=embedding_short,
            output_dir=output_dir,
            scaffold_split_dir=scaffold_split_dir,
            seeds=seeds,
            force=force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=patience,
            learning_rate=args.learning_rate,
            num_cross_attn_layers=0,
        )
        if level3_results:
            tqdm.write("  Level 3 completed successfully.")
        else:
            tqdm.write("  WARNING: Level 3 returned no results.")
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 4: Level 4 — CNN + CrossAttention
    # -----------------------------------------------------------------------
    level4_results = None
    if 4 in levels:
        step_name = "Step 4: Level 4 (CNN+CA)"
        progress.begin_step(step_name)
        tqdm.write(f"  Seeds to run: {seeds} ({len(seeds)} total)")
        tqdm.write(f"  Max epochs per seed: {args.epochs}, patience: {patience}")
        level4_results = run_level3(
            dataset=dataset,
            embedding_name=embedding_name,
            embedding_short=embedding_short,
            output_dir=output_dir,
            scaffold_split_dir=scaffold_split_dir,
            seeds=seeds,
            force=force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=patience,
            learning_rate=args.learning_rate,
            num_cross_attn_layers=2,
        )
        if level4_results:
            tqdm.write("  Level 4 completed successfully.")
        else:
            tqdm.write("  WARNING: Level 4 returned no results.")
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Step 5: Level 5-Lite — Transformer + Cross-Attention
    # -----------------------------------------------------------------------
    level5_results = None
    if 5 in levels:
        step_name = "Step 5: Level 5-Lite"
        progress.begin_step(step_name)
        tqdm.write(f"  Seeds to run: {seeds} ({len(seeds)} total)")
        tqdm.write(f"  Max epochs per seed: {args.epochs}, patience: {patience}")
        level5_results = run_level5_lite(
            dataset=dataset,
            embedding_name=embedding_name,
            embedding_short=embedding_short,
            output_dir=output_dir,
            scaffold_split_dir=scaffold_split_dir,
            seeds=seeds,
            force=force,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=patience,
            learning_rate=args.learning_rate,
        )
        if level5_results:
            tqdm.write("  Level 5-Lite completed successfully.")
        else:
            tqdm.write("  WARNING: Level 5-Lite returned no results.")
        progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Report: Comparative report + visualizations
    # -----------------------------------------------------------------------
    step_name = "Report + Visualizations"
    progress.begin_step(step_name)

    aggregated = aggregate_benchmark_metrics(
        level1_results, level2_results, level3_results, level3_key="level3_cnn",
    )
    # Merge Level 4 if present
    if level4_results:
        l4_agg = aggregate_benchmark_metrics(
            None, None, level4_results, level3_key="level4_cnn_ca",
        )
        aggregated.update(l4_agg)
    
    # Merge Level 5-Lite if present
    if level5_results:
        l5_agg = aggregate_benchmark_metrics(
            None, None, level5_results, level3_key="level5_lite",
        )
        aggregated.update(l5_agg)

    if not aggregated:
        tqdm.write("  No results to compare. At least one level must produce results.")
        progress.global_bar.close()
        sys.exit(1)

    # Terminal table
    print_comparison_table(aggregated, dataset, embedding_short)

    # Save JSON
    elapsed = time.time() - t_start
    json_path = save_benchmark_json(
        aggregated, dataset, embedding_short, output_dir, levels, seeds, elapsed,
    )

    # Visualizations
    viz_paths = generate_all_visualizations(aggregated, dataset, embedding_short, output_dir)
    progress.end_step(step_name)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - t_start
    progress.close(total_elapsed)

    tqdm.write("")
    tqdm.write(f"  Results: {json_path}")
    if viz_paths:
        tqdm.write(f"  Plots:   {len(viz_paths)} generated in {output_dir}/")
        for p in viz_paths:
            tqdm.write(f"           - {os.path.basename(p)}")

    return aggregated


if __name__ == "__main__":
    main()
