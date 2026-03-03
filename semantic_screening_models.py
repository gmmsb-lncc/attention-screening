#!/usr/bin/env python3
"""Unified benchmark for semantic-screening model comparison.

Pipeline:
  Step 0: Verify/generate scaffold splits
  Step 1: Level 1 — Fingerprint + KNN/MLP (baseline)
  Step 2: Level 2 — Embedding vectors + KNN/MLP
  Step 3: Level 3 — Transformer + Cross-Attention
  Report: Comparison table, JSON, and key visualizations

Usage:
    python semantic_screening_models.py --dataset human --embedding 8M --levels 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_BASE_PATH = "./results/protein_model_benchmark_{dataset_type}_v2"

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
    "level3_crossatt": "Level 3 (CrossAtt)",
}

METRICS_ORDER = ["accuracy", "mcc", "f1", "precision", "recall", "auc"]

LEVEL_COLORS = {
    "level1_fp_knn": "#1b9e77",
    "level1_fp_mlp": "#66c2a5",
    "level2_emb_knn": "#7570b3",
    "level2_emb_mlp": "#a6a3d9",
    "level3_crossatt": "#ff7f0e",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark: scaffold split → models → comparative report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["human", "non_human"],
                   help="Dataset to benchmark")
    p.add_argument("--embedding", default="8M", choices=["8M", "150M", "650M"],
                   help="ESM-2 model shorthand (default: 8M)")
    p.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3],
                   help="Levels to run: 1=FP, 2=Emb, 3=CrossAtt (default: 1 2 3)")
    p.add_argument("--output_dir", default=None,
                   help="Results dir (default: ./results/benchmark_{dataset}_{embedding})")
    p.add_argument("--scaffold_split_dir", default="scaffolds_splits/output",
                   help="Scaffold split directory")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds for multi-seed runs (default: [42, 123, 456, 789, 1024])")
    p.add_argument("--force", action="store_true",
                   help="Force recalculation")
    p.add_argument("--force_split", action="store_true",
                   help="Force regeneration of scaffold splits")
    p.add_argument("--epochs", type=int, default=300,
                   help="Max epochs for Level 3 (default: 300)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for Level 3 (default: 64)")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (default: 20)")
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="Learning rate for Level 3 (default: 5e-5)")
    return p


# ---------------------------------------------------------------------------
# Step 0: Scaffold splits
# ---------------------------------------------------------------------------

def _find_tsv(path: str) -> str:
    """Find .tsv or .tsv.gz file."""
    if os.path.exists(path):
        return path
    gz = path + ".gz" if not path.endswith(".gz") else path
    return gz if os.path.exists(gz) else ""


def ensure_scaffold_splits(dataset: str, scaffold_split_dir: str, force: bool) -> bool:
    """Verify or generate scaffold splits."""
    scenario_dir = os.path.join(scaffold_split_dir, "scenarios", "Sc")
    
    train = _find_tsv(os.path.join(scenario_dir, f"{dataset}_train.tsv"))
    val = _find_tsv(os.path.join(scenario_dir, f"{dataset}_val.tsv"))
    test = _find_tsv(os.path.join(scaffold_split_dir, f"{dataset}_test.tsv"))
    
    if train and val and test and not force:
        print(f"  [OK] Scaffold splits found for {dataset}")
        return True
    
    print(f"  Generating scaffold splits...")
    cmd = [sys.executable, "scaffold_split.py", "--output-dir", scaffold_split_dir, "--scenarios", "Sc"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: scaffold_split.py failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 0b: Ligand vectors (mean pooling)
# ---------------------------------------------------------------------------

def ensure_ligand_vectors(dataset: str, embedding_name: str, force: bool) -> bool:
    """Extract ligand vectors from MoLFormer matrices using mean pooling."""
    build_dir = Path(EMBEDDING_BASE_PATH.format(dataset_type=dataset), embedding_name, "build")
    molformer_dir = build_dir / "molformer_matrix"
    vector_dir = build_dir / "ligand_embeddings"
    
    if vector_dir.exists() and any(vector_dir.glob("*_embedding.npy")) and not force:
        print(f"  [OK] Ligand vectors: {vector_dir}")
        return True
    
    if not molformer_dir.exists():
        print(f"  WARNING: MoLFormer dir not found: {molformer_dir}")
        return False
    
    vector_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    
    for mf in molformer_dir.glob("*_matrix.npy"):
        chembl_id = mf.stem.replace("_matrix", "").replace("_molformer", "")
        out_path = vector_dir / f"{chembl_id}_embedding.npy"
        if out_path.exists() and not force:
            continue
        try:
            mat = np.load(mf)
            vec = mat.mean(axis=0).astype(np.float32)
            np.save(out_path, vec)
            processed += 1
        except Exception as e:
            print(f"  ERROR: {mf.name}: {e}")
    
    print(f"  Ligand vectors: {processed} extracted")
    return True


# ---------------------------------------------------------------------------
# Level 1 & 2: Classical ML
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
    """Run Level 1/2 across seeds and aggregate results."""
    from split_comparison_analysis import run_single_dataset
    
    seed_results: Dict[str, Dict[str, List[float]]] = {}
    
    for seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
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
        
        if result is None:
            result = _load_results(seed_dir)
        if result is None:
            continue
        
        # Find scaffold scenario
        sc_key = next((k for k in result if "scaffold" in k.lower()), next(iter(result), None))
        if not sc_key:
            continue
        
        sc = result[sc_key]
        for model in ["KNN", "MLP"]:
            if model not in sc:
                continue
            if model not in seed_results:
                seed_results[model] = {}
            for m in METRICS_ORDER:
                val = sc[model].get(m)
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    seed_results[model].setdefault(m, []).append(float(val))
    
    if not seed_results:
        return None
    
    # Aggregate
    aggregated = {}
    for model, metrics in seed_results.items():
        agg = {}
        for m, vals in metrics.items():
            arr = np.array(vals)
            agg[m] = float(np.mean(arr))
            agg[f"{m}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        aggregated[model] = agg
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_comparison_results.json"), "w") as f:
        json.dump({"results": {"Split by Scaffold": aggregated}}, f, indent=2)
    
    return {"Split by Scaffold": aggregated}


def _load_results(level_dir: str) -> Optional[Dict]:
    """Load cached results."""
    path = os.path.join(level_dir, "split_comparison_results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("results")
    except (json.JSONDecodeError, KeyError):
        return None


def run_level1(dataset: str, output_dir: str, scaffold_split_dir: str, seeds: List[int], force: bool) -> Optional[Dict]:
    """Level 1: Fingerprint + KNN/MLP."""
    level_dir = os.path.join(output_dir, "level1_fingerprint", dataset)
    print(f"  Level 1 output: {level_dir}")
    return _run_level_multiseed(dataset, level_dir, scaffold_split_dir, seeds, force, "fingerprint")


def run_level2(dataset: str, embedding_name: str, embedding_short: str, output_dir: str,
               scaffold_split_dir: str, seeds: List[int], force: bool) -> Optional[Dict]:
    """Level 2: Embedding vectors + KNN/MLP."""
    level_dir = os.path.join(output_dir, f"level2_embedding_{embedding_short}", dataset)
    print(f"  Level 2 output: {level_dir}")
    return _run_level_multiseed(dataset, level_dir, scaffold_split_dir, seeds, force, "embedding", embedding_name)


# ---------------------------------------------------------------------------
# Level 3: Transformer + Cross-Attention
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
    patience: int,
    learning_rate: float,
) -> Optional[Dict]:
    """Level 3: Transformer encoders + Cross-Attention."""
    from crossattention_split_analysis.experiment import run_single_analysis
    
    level_dir = os.path.join(output_dir, f"level3_crossatt_{embedding_short}")
    print(f"  Level 3 output: {level_dir}")
    
    results = run_single_analysis(
        embedding_name=embedding_name,
        dataset_type=dataset,
        output_dir=level_dir,
        seeds=seeds,
        force=force,
        scenarios=["scaffold"],
        num_epochs=epochs,
        patience=patience if patience > 0 else None,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=128,  # Reduced from 256 to combat overfitting (314K vs 972K params)
        num_cross_attn_layers=2,
        num_heads=4,  # Reduced from 8 for less memorization
        dropout=0.5,  # Increased from 0.4
        classifier_dropout=0.6,
        weight_decay=1e-3,  # Increased from 5e-4 for stronger L2 regularization
        label_smoothing=0.15,  # Increased from 0.1
        max_grad_norm=0.5,  # Reduced from 1.0 for more stable gradients
        classification_only=True,
        use_molformer_ligand=True,
        scaffold_split_dir=scaffold_split_dir,
        model_variant="level3_crossatt",
        optimize_threshold=False,
        fixed_threshold=0.5,
    )
    
    if results is None:
        results = _load_crossattention_results(level_dir, dataset, embedding_short)
    
    return results


def _load_crossattention_results(level_dir: str, dataset: str, embedding_short: str) -> Optional[Dict]:
    """Load cached crossattention results."""
    import glob
    candidates = glob.glob(os.path.join(level_dir, "*crossattention_analysis_results.json"))
    if not candidates:
        return None
    try:
        with open(candidates[0]) as f:
            return json.load(f).get("model_results")
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Aggregate & Report
# ---------------------------------------------------------------------------

def aggregate_metrics(level1: Optional[Dict], level2: Optional[Dict], level3: Optional[Dict]) -> Dict:
    """Aggregate metrics from all levels."""
    result = {}
    
    for level_results, prefix_map in [
        (level1, [("KNN", "level1_fp_knn"), ("MLP", "level1_fp_mlp")]),
        (level2, [("KNN", "level2_emb_knn"), ("MLP", "level2_emb_mlp")]),
    ]:
        if not level_results:
            continue
        sc_key = next((k for k in level_results if "scaffold" in k.lower()), next(iter(level_results), None))
        if not sc_key:
            continue
        sc = level_results[sc_key]
        for model, label in prefix_map:
            if model not in sc:
                continue
            row = {}
            for m in METRICS_ORDER:
                val = sc[model].get(m)
                row[m] = float(val) if val is not None and not np.isnan(val) else None
                std = sc[model].get(f"{m}_std")
                row[f"{m}_std"] = float(std) if std is not None and not np.isnan(std) else None
            result[label] = row
    
    # Level 3
    if level3:
        sc_key = next((k for k in level3 if "scaffold" in k.lower()), next(iter(level3), None))
        if sc_key:
            data = level3[sc_key]
            # Handle nested or flat structure
            if "accuracy" in data or "mcc" in data:
                metrics_block = data
            else:
                metrics_block = next((v for v in data.values() if isinstance(v, dict) and "mcc" in v), data)
            
            row = {}
            for m in METRICS_ORDER:
                val = metrics_block.get(m)
                row[m] = float(val) if val is not None and isinstance(val, (int, float)) and not np.isnan(val) else None
                std = metrics_block.get(f"{m}_std")
                row[f"{m}_std"] = float(std) if std is not None and isinstance(std, (int, float)) and not np.isnan(std) else None
            result["level3_crossatt"] = row
    
    return result


def print_table(aggregated: Dict, dataset: str, embedding: str) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {dataset} / ESM-2 {embedding} / Scaffold Split")
    print("=" * 80)
    
    header = f"{'Model':<22s}" + "".join(f"  {m.upper():>8s}" for m in METRICS_ORDER)
    print(header)
    print("-" * 80)
    
    for key in ["level1_fp_knn", "level1_fp_mlp", "level2_emb_knn", "level2_emb_mlp", "level3_crossatt"]:
        if key not in aggregated:
            continue
        row = aggregated[key]
        line = f"{LEVEL_LABELS[key]:<22s}"
        for m in METRICS_ORDER:
            val = row.get(m)
            std = row.get(f"{m}_std")
            if val is None:
                line += f"  {'N/A':>8s}"
            elif std and std > 0:
                line += f"  {val:.3f}±{std:.2f}"
            else:
                line += f"  {val:>8.4f}"
        print(line)
    print("=" * 80)


def save_json(aggregated: Dict, dataset: str, embedding: str, output_dir: str,
              levels: List[int], seeds: List[int], elapsed: float) -> str:
    """Save benchmark results to JSON."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "embedding": embedding,
            "embedding_full": SUPPORTED_EMBEDDINGS[embedding],
            "levels": levels,
            "seeds": seeds,
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": {k: {**{"label": LEVEL_LABELS.get(k, k)}, **v} for k, v in aggregated.items()},
    }
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_comparison.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Visualizations (minimal)
# ---------------------------------------------------------------------------

def plot_mcc_ranking(aggregated: Dict, dataset: str, embedding: str, output_dir: str) -> Optional[str]:
    """MCC ranking horizontal bar chart."""
    items = [(k, aggregated[k].get("mcc"), aggregated[k].get("mcc_std", 0)) 
             for k in aggregated if aggregated[k].get("mcc") is not None]
    if not items:
        return None
    
    items.sort(key=lambda x: x[1])
    
    fig, ax = plt.subplots(figsize=(9, len(items) * 0.8 + 1))
    y = np.arange(len(items))
    mccs = [x[1] for x in items]
    stds = [x[2] or 0 for x in items]
    colors = [LEVEL_COLORS.get(x[0], "#888") for x in items]
    labels = [LEVEL_LABELS.get(x[0], x[0]) for x in items]
    
    ax.barh(y, mccs, xerr=stds if any(stds) else None, capsize=4, color=colors, height=0.6)
    
    for i, (mcc, std) in enumerate(zip(mccs, stds)):
        txt = f"{mcc:.3f}" + (f" ± {std:.3f}" if std else "")
        ax.text(max(mcc, 0) + 0.01, i, txt, ha="left", va="center", fontsize=10)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("MCC")
    ax.set_title(f"MCC Ranking — {dataset} / ESM-2 {embedding}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    
    path = os.path.join(output_dir, "benchmark_mcc_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")
    return path


def plot_heatmap(aggregated: Dict, dataset: str, embedding: str, output_dir: str) -> Optional[str]:
    """Performance heatmap."""
    models = [k for k in ["level1_fp_knn", "level1_fp_mlp", "level2_emb_knn", "level2_emb_mlp", "level3_crossatt"]
              if k in aggregated]
    if not models:
        return None
    
    matrix = np.array([[aggregated[m].get(metric) or np.nan for metric in METRICS_ORDER] for m in models])
    
    fig, ax = plt.subplots(figsize=(10, len(models) * 0.8 + 1.5))
    im = ax.imshow(matrix, cmap=plt.cm.RdYlGn, aspect="auto", vmin=0, vmax=1)
    
    for i in range(len(models)):
        for j in range(len(METRICS_ORDER)):
            val = matrix[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "N/A"
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)
    
    ax.set_xticks(range(len(METRICS_ORDER)))
    ax.set_xticklabels([m.upper() for m in METRICS_ORDER])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([LEVEL_LABELS[m] for m in models])
    ax.set_title(f"Performance — {dataset} / ESM-2 {embedding}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    
    path = os.path.join(output_dir, "benchmark_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    
    dataset = args.dataset
    embedding = args.embedding
    embedding_name = SUPPORTED_EMBEDDINGS[embedding]
    levels = sorted(set(args.levels))
    
    # Validate levels
    for lv in levels:
        if lv not in (1, 2, 3):
            print(f"ERROR: Invalid level {lv}. Valid: 1, 2, 3")
            sys.exit(1)
    
    seeds = args.seeds or [42, 123, 456, 789, 1024]
    output_dir = args.output_dir or f"./results/benchmark_{dataset}_{embedding}"
    patience = args.patience if args.patience > 0 else None
    
    print("=" * 60)
    print("SEMANTIC SCREENING BENCHMARK")
    print("=" * 60)
    print(f"  Dataset:    {dataset}")
    print(f"  Embedding:  {embedding} ({embedding_name})")
    print(f"  Levels:     {levels}")
    print(f"  Seeds:      {seeds}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    
    # Step 0: Scaffold splits
    print("\n[Step 0] Scaffold Splits")
    if not ensure_scaffold_splits(dataset, args.scaffold_split_dir, args.force_split):
        print("FATAL: Cannot proceed without scaffold splits.")
        sys.exit(1)
    
    # Step 0b: Ligand vectors
    if 2 in levels:
        print("\n[Step 0b] Ligand Vectors")
        ensure_ligand_vectors(dataset, embedding_name, args.force)
    
    # Run levels
    level1_results = level2_results = level3_results = None
    
    if 1 in levels:
        print("\n[Step 1] Level 1: Fingerprint + KNN/MLP")
        level1_results = run_level1(dataset, output_dir, args.scaffold_split_dir, seeds, args.force)
        print("  Done." if level1_results else "  WARNING: No results.")
    
    if 2 in levels:
        print("\n[Step 2] Level 2: Embedding + KNN/MLP")
        level2_results = run_level2(dataset, embedding_name, embedding, output_dir, 
                                    args.scaffold_split_dir, seeds, args.force)
        print("  Done." if level2_results else "  WARNING: No results.")
    
    if 3 in levels:
        print("\n[Step 3] Level 3: Transformer + Cross-Attention")
        level3_results = run_level3(
            dataset, embedding_name, embedding, output_dir, args.scaffold_split_dir,
            seeds, args.force, args.epochs, args.batch_size, patience, args.learning_rate,
        )
        print("  Done." if level3_results else "  WARNING: No results.")
    
    # Report
    print("\n[Report]")
    aggregated = aggregate_metrics(level1_results, level2_results, level3_results)
    
    if not aggregated:
        print("  ERROR: No results to report.")
        sys.exit(1)
    
    print_table(aggregated, dataset, embedding)
    elapsed = time.time() - t_start
    save_json(aggregated, dataset, embedding, output_dir, levels, seeds, elapsed)
    
    # Visualizations
    print("\n[Visualizations]")
    plot_mcc_ranking(aggregated, dataset, embedding, output_dir)
    plot_heatmap(aggregated, dataset, embedding, output_dir)
    
    # Summary
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal time: {h}h{m:02d}m{s:02d}s")


if __name__ == "__main__":
    main()
