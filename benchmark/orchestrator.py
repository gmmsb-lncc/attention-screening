"""Benchmark orchestrator: coordinates the full pipeline.

This is the top-level entry point that ties together scaffold splits,
embedding extraction, fine-tuning, multi-level model training,
metric aggregation, reporting, and visualization.

Design:
  - Follows the **Facade** pattern: one simple ``run()`` method.
  - Each concern (splits, embeddings, levels, metrics, plots) lives
    in its own module — the orchestrator only coordinates.
  - Level runners are plugged in via the ``BaseLevelRunner`` interface
    (Open/Closed principle) so adding Level 5/6 requires zero changes here.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.finetuning import run_finetuning_pipeline
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.level1 import Level1Runner
from benchmark.levels.level1b import Level1bRunner
from benchmark.levels.level1c import Level1cRunner
from benchmark.levels.level2 import Level2Runner
from benchmark.levels.level3 import Level3Runner, Level3aRunner
from benchmark.levels.level4 import Level4Runner
from benchmark.levels.level5 import Level5Runner
from benchmark.levels.level5b import Level5bRunner
from benchmark.levels.level6a import Level6aRunner
from benchmark.levels.level6b import Level6bRunner
from benchmark.metrics import aggregate_benchmark_metrics
from benchmark.progress import BenchmarkProgress
from benchmark.reporting import print_comparison_table, save_benchmark_json
from benchmark.splits import ensure_scaffold_splits
from benchmark.visualization import generate_all


class BenchmarkOrchestrator:
    """Coordinates the end-to-end benchmark pipeline.

    Parameters
    ----------
    config:
        Immutable run configuration (from CLI or programmatic use).
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config
        self._level_results: Dict[str, Optional[Dict]] = {}
        self._finetuned_dirs: Dict[str, Dict[str, Optional[str]]] = {}
        self._t_start: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Dict]:
        """Execute the full benchmark and return aggregated metrics."""
        config = self._config
        self._print_banner()
        os.makedirs(config.resolved_output_dir, exist_ok=True)

        self._t_start = time.time()
        progress = BenchmarkProgress(config)

        # --- Step 0: Scaffold splits --------------------------------
        self._run_step(progress, "Step 0: Scaffold Splits", self._ensure_splits)

        # --- Fine-tuning (optional) ---------------------------------
        if config.finetune:
            self._run_step(
                progress,
                "Step FT: ESM-2 + MolFormer Fine-tuning",
                self._run_finetuning,
            )

        # --- Level runners ------------------------------------------
        runners = self._build_runners()

        for level, runner, step_name in runners:
            self._run_step(
                progress,
                step_name,
                lambda r=runner: self._run_level(r, level),
            )

        # --- Report + Visualization ---------------------------------
        self._run_step(progress, "Report + Visualizations", self._generate_report)

        total_elapsed = time.time() - self._t_start
        progress.close(total_elapsed)
        self._print_summary(total_elapsed)

        return self._aggregated

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _ensure_splits(self) -> None:
        if not ensure_scaffold_splits(self._config):
            tqdm.write("FATAL: Cannot proceed without scaffold splits.")
            sys.exit(1)

    def _run_finetuning(self) -> None:
        self._finetuned_dirs = run_finetuning_pipeline(self._config)

    def _run_level(self, runner: BaseLevelRunner, level: str) -> None:
        result = runner.run()
        self._level_results[level] = result
        if result:
            tqdm.write(f"  Level {level} completed successfully.")
        else:
            tqdm.write(f"  WARNING: Level {level} returned no results.")

    def _generate_report(self) -> None:
        config = self._config

        self._aggregated = aggregate_benchmark_metrics(
            level1a_results=self._level_results.get("1a"),
            level1b_results=self._level_results.get("1b"),
            level1c_results=self._level_results.get("1c"),
            level2_results=self._level_results.get("2"),
            level3_results=self._level_results.get("3"),
            level3a_results=self._level_results.get("3a"),
            level4_results=self._level_results.get("4"),
            level5_results=self._level_results.get("5a"),
            level5b_results=self._level_results.get("5b"),
            level6a_results=self._level_results.get("6a"),
            level6b_results=self._level_results.get("6b"),
        )

        if not self._aggregated:
            tqdm.write("  No results to compare. At least one level must produce results.")
            sys.exit(1)

        # Rigor gate: block partial comparisons when requested levels are missing.
        strict = os.getenv("BENCHMARK_STRICT_LEVEL_COMPLETENESS", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if strict:
            expected = {
                "1a": ["level1a_fp_knn", "level1a_fp_mlp"],
                "1b": ["level1b_ligmean_knn", "level1b_ligmean_mlp"],
                "1c": ["level1c_ligattn_knn", "level1c_ligattn_mlp"],
                "2": ["level2_meanpool_knn", "level2_meanpool_mlp"],
                "3": ["level3_attnpool_knn", "level3_attnpool_mlp"],
                "3a": ["level3a_attnpool_mlp"],
                "4": ["level4_crossatt_knn", "level4_crossatt_mlp"],
                "5a": ["level5_da_knn", "level5_da_mlp"],
                "5b": ["level5b_da_knn", "level5b_da_mlp"],
                "6a": ["level6a_ban_knn", "level6a_ban_mlp"],
                "6b": ["level6b_ban_knn", "level6b_ban_mlp"],
            }
            missing_models: list[str] = []
            for lv in config.levels:
                for key in expected.get(lv, []):
                    row = self._aggregated.get(key)
                    if not row or row.get("mcc") is None:
                        missing_models.append(key)
            if missing_models:
                tqdm.write("FATAL: Strict completeness gate failed.")
                tqdm.write(f"  Missing model rows: {missing_models}")
                tqdm.write("  Set BENCHMARK_STRICT_LEVEL_COMPLETENESS=0 to bypass.")
                sys.exit(1)

        print_comparison_table(self._aggregated, config)

        elapsed = time.time() - self._t_start
        self._json_path = save_benchmark_json(self._aggregated, config, elapsed)
        self._viz_paths = generate_all(self._aggregated, config)

    # ------------------------------------------------------------------
    # Runner construction
    # ------------------------------------------------------------------

    def _build_runners(
        self,
    ) -> List[tuple[str, BaseLevelRunner, str]]:
        """Instantiate level runners for the configured levels."""
        config = self._config
        runners: List[tuple[str, BaseLevelRunner, str]] = []

        if "1a" in config.levels:
            runners.append(("1a", Level1Runner(config), "Step 1a: L1a (FP+KNN/MLP)"))

        if "1b" in config.levels:
            runners.append(("1b", Level1bRunner(config), "Step 1b: L1b (LigMeanPool+KNN/MLP)"))

        if "1c" in config.levels:
            runners.append(("1c", Level1cRunner(config), "Step 1c: L1c (LigAttnPool+KNN/MLP)"))

        if "2" in config.levels:
            runners.append(("2", Level2Runner(config), "Step 2: L2 (MeanPool+KNN/MLP)"))

        if "3" in config.levels:
            runners.append(("3", Level3Runner(config), "Step 3: L3 (AttnPool+KNN/MLP)"))

        if "3a" in config.levels:
            runners.append(("3a", Level3aRunner(config), "Step 3a: L3a (AttnPool+MLP only)"))

        if "4" in config.levels:
            runners.append(("4", Level4Runner(config), "Step 4: L4 (CrossAttn+AttnPool+KNN/MLP)"))

        if "5a" in config.levels:
            runners.append(("5a", Level5Runner(config), "Step 5a: L5a (CrossAttn+AttnPool+GRL+KNN/MLP)"))

        if "5b" in config.levels:
            runners.append(("5b", Level5bRunner(config), "Step 5b: L5b (AttnPool+GRL+KNN/MLP)"))

        if "6a" in config.levels:
            runners.append(("6a", Level6aRunner(config), "Step 6a: L6a (CrossAttn+BAN+GRL+KNN/MLP)"))

        if "6b" in config.levels:
            runners.append(("6b", Level6bRunner(config), "Step 6b: L6b (AttnPool+BAN+GRL+KNN/MLP)"))

        return runners

    # ------------------------------------------------------------------
    # Fine-tuned embedding resolution (reserved for future use)
    # ------------------------------------------------------------------

    def _resolve_finetuned_dirs(self) -> tuple[Optional[str], Optional[str]]:
        """Resolve fine-tuned embedding directories when ``--use_finetuned``."""
        config = self._config
        if not config.use_finetuned:
            return None, None

        if config.dataset == "all":
            tqdm.write("  NOTE: --use_finetuned with --dataset all — checking per-dataset paths.")
            all_exist = True
            for ds in ("human", "non_human"):
                ds_output = config.resolved_output_dir.replace("benchmark_all", f"benchmark_{ds}")
                base = os.path.join(ds_output, "finetuned_embeddings", ds)
                if not self._check_finetuned_dir(base, ds):
                    all_exist = False
            if not all_exist:
                tqdm.write("    Falling back to vanilla embeddings.\n")
            return None, None  # per-dataset paths handled in Level2Runner for "all"

        base = os.path.join(config.resolved_output_dir, "finetuned_embeddings", config.dataset)
        protein_dir = os.path.join(base, "protein_embeddings")
        ligand_dir = os.path.join(base, "ligand_embeddings")

        protein_ok = os.path.isdir(protein_dir) and os.listdir(protein_dir)
        ligand_ok = os.path.isdir(ligand_dir) and os.listdir(ligand_dir)

        if protein_ok and ligand_ok:
            tqdm.write(f"\n  Using fine-tuned embeddings from: {base}")
            return protein_dir, ligand_dir

        tqdm.write(f"\n  WARNING: Fine-tuned embeddings not found at {base}")
        tqdm.write("    Run with --finetune first. Falling back to vanilla embeddings.\n")
        return None, None

    @staticmethod
    def _check_finetuned_dir(base: str, dataset: str) -> bool:
        protein_dir = os.path.join(base, "protein_matrices")
        ligand_dir = os.path.join(base, "ligand_matrices")
        p_ok = os.path.isdir(protein_dir) and os.listdir(protein_dir)
        l_ok = os.path.isdir(ligand_dir) and os.listdir(ligand_dir)
        if p_ok and l_ok:
            n_p = len([f for f in os.listdir(protein_dir) if f.endswith(".npy")])
            n_l = len([f for f in os.listdir(ligand_dir) if f.endswith(".npy")])
            tqdm.write(f"    OK {dataset}: {n_p} protein matrices, {n_l} ligand matrices")
            return True
        tqdm.write(f"    MISSING {dataset}: {base}")
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _run_step(
        progress: BenchmarkProgress,
        step_name: str,
        fn: object,
    ) -> None:
        """Run a pipeline step with progress tracking."""
        progress.begin_step(step_name)
        fn()
        progress.end_step(step_name)

    def _print_banner(self) -> None:
        config = self._config
        print("=" * 70)
        print("SEMANTIC SCREENING — UNIFIED BENCHMARK")
        print("=" * 70)
        print(f"  Dataset:          {config.dataset}")
        print(f"  Mode:             {config.mode} ({'fit=train, eval=val' if config.mode == 'train' else 'fit=val, eval=test'})")
        print(f"  Embedding:        {config.embedding} ({config.embedding_name})")
        print(f"  Levels:           {config.levels}")
        print(f"  Seeds:            {config.resolved_seeds}")
        print(f"  Output dir:       {config.resolved_output_dir}")
        print(f"  Scaffold splits:  {config.scaffold_split_dir}")
        print(f"  Force:            {config.force}")
        if any(lv in config.levels for lv in ("1c", "3", "4")):
            print(f"  DL epochs:        {config.epochs}")
            print(f"  DL batch_size:    {config.batch_size}")
            print(f"  DL patience:      {config.resolved_patience}")
            print(f"  DL learning_rate: {config.learning_rate}")
        print("=" * 70)

    def _print_summary(self, total_elapsed: float) -> None:
        json_path = getattr(self, "_json_path", "")
        viz_paths = getattr(self, "_viz_paths", [])
        tqdm.write("")
        if json_path:
            tqdm.write(f"  Results: {json_path}")
        if viz_paths:
            tqdm.write(f"  Plots:   {len(viz_paths)} generated in {self._config.resolved_output_dir}/")
            for p in viz_paths:
                tqdm.write(f"           - {os.path.basename(p)}")
