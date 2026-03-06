#!/usr/bin/env python3
"""
Hyperparameter sweep for CrossAttention with MCC-first model selection.

Protocol:
1. Screen a grid of configurations using a validation-driven objective.
2. Rank by MCC (primary), accuracy (secondary), loss (tertiary).
3. Optionally rerun top-K with a larger seed set for robust selection.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from crossattention_split_analysis.experiment import run_single_analysis


DEFAULT_SCREEN_SEEDS = [42, 123]
DEFAULT_FINAL_SEEDS = [42, 123, 456, 789, 1024]


@dataclass(frozen=True)
class SweepConfig:
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    dropout: float
    num_cnn_layers: int
    num_cross_attn_layers: int
    num_heads: int
    ff_dim: int
    batch_size: int

    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "num_cnn_layers": self.num_cnn_layers,
            "num_cross_attn_layers": self.num_cross_attn_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "batch_size": self.batch_size,
        }


def _iter_product(space: Dict[str, Iterable]) -> Iterable[Dict]:
    """Simple cartesian product helper for dict-based grids."""
    keys = list(space.keys())
    values = [list(space[k]) for k in keys]
    if not keys:
        yield {}
        return

    def _recurse(idx: int, current: Dict):
        if idx == len(keys):
            yield current.copy()
            return
        key = keys[idx]
        for value in values[idx]:
            current[key] = value
            yield from _recurse(idx + 1, current)

    yield from _recurse(0, {})


def build_search_space(profile: str) -> List[SweepConfig]:
    """
    Build preset search spaces.

    quick: 12 configs
    standard: 36 configs
    aggressive: 72 configs
    """
    if profile == "quick":
        # Quick profile now includes batch-size variation with LR scaling.
        # (batch, lr) pairs: (16, 5e-5), (32, 1e-4), (64, 2e-4)
        quick_pairs = [
            (16, 5e-5),
            (32, 1e-4),
            (64, 2e-4),
        ]
        configs: List[SweepConfig] = []
        for hidden_dim in [192, 256]:
            num_heads = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 1
            ff_dim = int(hidden_dim * 4)
            for dropout in [0.1, 0.2]:
                for batch_size, learning_rate in quick_pairs:
                    configs.append(
                        SweepConfig(
                            hidden_dim=hidden_dim,
                            learning_rate=float(learning_rate),
                            weight_decay=0.01,
                            dropout=float(dropout),
                            num_cnn_layers=3,
                            num_cross_attn_layers=2,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            batch_size=int(batch_size),
                        )
                    )
        return configs
    elif profile == "standard":
        base = {
            "hidden_dim": [224, 256, 320],
            "learning_rate": [5e-5, 1e-4, 2e-4],
            "weight_decay": [1e-3, 1e-2],
            "dropout": [0.05, 0.1],
            "num_cnn_layers": [3],
            "num_cross_attn_layers": [2],
            "batch_size": [32],
        }
    elif profile == "aggressive":
        base = {
            "hidden_dim": [224, 256, 320],
            "learning_rate": [5e-5, 1e-4, 2e-4],
            "weight_decay": [1e-3, 1e-2],
            "dropout": [0.05, 0.1],
            "num_cnn_layers": [2, 3],
            "num_cross_attn_layers": [2, 3],
            "batch_size": [32],
        }
    else:
        raise ValueError(f"Unsupported profile: {profile!r}")

    configs: List[SweepConfig] = []
    for raw in _iter_product(base):
        hidden_dim = int(raw["hidden_dim"])
        # Keep attention heads valid for hidden_dim.
        num_heads = 8 if hidden_dim % 8 == 0 else 4 if hidden_dim % 4 == 0 else 1
        ff_dim = int(hidden_dim * 4)
        cfg = SweepConfig(
            hidden_dim=hidden_dim,
            learning_rate=float(raw["learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            dropout=float(raw["dropout"]),
            num_cnn_layers=int(raw["num_cnn_layers"]),
            num_cross_attn_layers=int(raw["num_cross_attn_layers"]),
            num_heads=num_heads,
            ff_dim=ff_dim,
            batch_size=int(raw["batch_size"]),
        )
        configs.append(cfg)
    return configs


def _extract_metrics(results: Dict) -> Dict[str, float]:
    """Extract flat metric dictionary from run_single_analysis return object."""
    if not results:
        return {}

    scenario_key = next(iter(results.keys()))
    scenario_block = results.get(scenario_key, {})
    model_block = scenario_block.get("CNN+CrossAttn", {})

    wanted = [
        "accuracy",
        "mcc",
        "auc",
        "f1",
        "loss",
        "decision_threshold",
        "threshold_optimized_score",
        "n_params",
        "accuracy_std",
        "mcc_std",
        "auc_std",
        "f1_std",
        "loss_std",
    ]
    out = {}
    for key in wanted:
        if key in model_block:
            out[key] = model_block[key]
    return out


def _extract_metrics_from_saved_json(trial_dir: Path) -> Dict[str, float]:
    """Load metrics from an existing result JSON inside a trial directory."""
    candidates = sorted(trial_dir.glob("*crossattention_analysis_results.json"))
    if not candidates:
        return {}
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    model_results = payload.get("model_results", {})
    if not model_results:
        return {}
    first_block = next(iter(model_results.values()))
    if not isinstance(first_block, dict):
        return {}

    wanted = [
        "accuracy",
        "mcc",
        "auc",
        "f1",
        "loss",
        "decision_threshold",
        "threshold_optimized_score",
        "n_params",
        "accuracy_std",
        "mcc_std",
        "auc_std",
        "f1_std",
        "loss_std",
    ]
    out = {}
    for key in wanted:
        if key in first_block:
            out[key] = first_block[key]
    return out


def _as_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def _rank_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Rank trials by MCC desc, accuracy desc, loss asc."""
    ranked = df.copy()
    ranked["sort_mcc"] = ranked["mcc"].apply(lambda x: _as_float(x, -1e9))
    ranked["sort_accuracy"] = ranked["accuracy"].apply(lambda x: _as_float(x, -1e9))
    ranked["sort_loss"] = ranked["loss"].apply(lambda x: _as_float(x, 1e9))
    ranked = ranked.sort_values(
        by=["sort_mcc", "sort_accuracy", "sort_loss"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked.drop(columns=["sort_mcc", "sort_accuracy", "sort_loss"])


def _build_main_command(
    args: argparse.Namespace,
    best_cfg: SweepConfig,
    output_dir: str,
) -> str:
    cmd = [
        "python legacy/crossattention_split_analysis_main.py",
        f"--dataset {args.dataset}",
        f"--embedding {args.embedding}",
        "--scenarios scaffold",
        f"--scaffold_split_dir {args.scaffold_split_dir}",
        f"--epochs {args.epochs}",
        f"--batch_size {best_cfg.batch_size}",
        f"--learning_rate {best_cfg.learning_rate}",
        f"--weight_decay {best_cfg.weight_decay}",
        f"--hidden_dim {best_cfg.hidden_dim}",
        f"--num_cnn_layers {best_cfg.num_cnn_layers}",
        f"--num_cross_attn_layers {best_cfg.num_cross_attn_layers}",
        f"--num_heads {best_cfg.num_heads}",
        f"--ff_dim {best_cfg.ff_dim}",
        f"--dropout {best_cfg.dropout}",
        f"--max_grad_norm {args.max_grad_norm}",
        f"--classification_weight {args.classification_weight}",
        f"--regression_weight {args.regression_weight}",
        f"--threshold_metric {args.threshold_metric}",
        f"--output_dir {output_dir}",
    ]
    if args.external_test_mode:
        cmd.append("--external_test_mode")
    if args.molformer_ligand:
        cmd.append("--molformer_ligand")
    if args.no_early_stopping:
        cmd.append("--no-early-stopping")
    else:
        cmd.append(f"--patience {args.patience}")
    if args.disable_threshold_optimization:
        cmd.append("--disable-threshold-optimization")
        cmd.append(f"--fixed_threshold {args.fixed_threshold}")
    return " \\\n  ".join(cmd)


def _parse_pruning_epochs(raw: str, final_epochs: int) -> List[int]:
    """
    Parse pruning schedule and guarantee final epoch budget is included.

    Example:
      raw='80,200' and final_epochs=500 -> [80, 200, 500]
      raw='' and final_epochs=500 -> [500]
    """
    if final_epochs <= 0:
        raise ValueError("--epochs must be > 0")

    parsed = []
    tokens = [t.strip() for t in raw.split(",")] if raw else []
    for token in tokens:
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Invalid pruning epoch budget: {value} (must be > 0)")
        if value < final_epochs:
            parsed.append(value)
    stages = sorted(set(parsed))
    stages.append(final_epochs)
    return stages


def _stage_patience(
    base_patience: Optional[int],
    stage_epochs: int,
    no_early_stopping: bool,
) -> Optional[int]:
    """Compute patience for a pruning stage while keeping behavior stable."""
    if no_early_stopping or base_patience is None:
        return None
    stage_cap = max(5, int(0.3 * stage_epochs))
    return min(base_patience, stage_cap)


def _run_phase(
    phase_name: str,
    configs: List[Tuple[int, SweepConfig]],
    seeds: List[int],
    args: argparse.Namespace,
    phase_dir: Path,
    num_epochs: int,
    patience: Optional[int],
    stage_index: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    for idx, cfg in configs:
        trial_dir = phase_dir / f"trial_{idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{phase_name}] trial {idx:03d} | hidden={cfg.hidden_dim} "
            f"lr={cfg.learning_rate:.1e} wd={cfg.weight_decay:.1e} dropout={cfg.dropout}"
        )
        start = time.time()
        status = "ok"
        error_message = None
        metrics = {}
        cache_hit = False
        try:
            result = run_single_analysis(
                embedding_name=args.embedding,
                dataset_type=args.dataset,
                output_dir=str(trial_dir),
                seeds=seeds,
                force=args.force,
                use_attention=False,
                scenarios=["scaffold"],
                num_epochs=num_epochs,
                patience=patience,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                hidden_dim=cfg.hidden_dim,
                num_cnn_layers=cfg.num_cnn_layers,
                num_cross_attn_layers=cfg.num_cross_attn_layers,
                num_heads=cfg.num_heads,
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
                max_grad_norm=args.max_grad_norm,
                classification_weight=args.classification_weight,
                regression_weight=args.regression_weight,
                optimize_threshold=not args.disable_threshold_optimization,
                threshold_metric=args.threshold_metric,
                fixed_threshold=args.fixed_threshold,
                use_molformer_ligand=args.molformer_ligand,
                scaffold_split_dir=args.scaffold_split_dir,
                external_test_mode=args.external_test_mode,
            )
            if result is None:
                cached = _extract_metrics_from_saved_json(trial_dir)
                if cached:
                    metrics = cached
                    cache_hit = True
                else:
                    status = "failed"
                    error_message = "run_single_analysis returned None"
            else:
                metrics = _extract_metrics(result)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = str(exc)

        runtime_sec = time.time() - start
        row = {
            "phase": phase_name,
            "trial_id": idx,
            "status": status,
            "runtime_sec": runtime_sec,
            "output_dir": str(trial_dir),
            "n_seeds": len(seeds),
            "num_epochs": num_epochs,
            "patience": patience,
            "stage_index": stage_index,
            "cache_hit": cache_hit,
            "error": error_message,
            **cfg.to_dict(),
            **metrics,
        }
        rows.append(row)

        if status == "ok":
            print(
                f"  -> MCC={_as_float(row.get('mcc'), float('nan')):.4f}, "
                f"Acc={_as_float(row.get('accuracy'), float('nan')):.4f}, "
                f"Loss={_as_float(row.get('loss'), float('nan')):.4f}"
            )
        else:
            print(f"  -> failed: {error_message}")

    return pd.DataFrame(rows)


def _run_screen_with_pruning(
    indexed_configs: List[Tuple[int, SweepConfig]],
    args: argparse.Namespace,
    run_dir: Path,
    screen_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run screening phase with optional Successive Halving pruning.

    Returns:
        df_screen_all: concatenated results for all stages
        df_last_stage_ranked: ranked successful trials from last executed stage
    """
    if args.pruning_mode == "none":
        stage_epochs = [args.epochs]
    else:
        stage_epochs = _parse_pruning_epochs(args.prune_epochs, args.epochs)

    active = list(indexed_configs)
    all_stage_frames: List[pd.DataFrame] = []
    last_stage_ranked = pd.DataFrame()
    id_to_cfg = {idx: cfg for idx, cfg in indexed_configs}

    print(f"\nPruning mode: {args.pruning_mode}")
    print(f"Pruning stages (epochs): {stage_epochs}")

    for stage_idx, stage_ep in enumerate(stage_epochs, start=1):
        if not active:
            break

        stage_patience = _stage_patience(
            base_patience=args.patience,
            stage_epochs=stage_ep,
            no_early_stopping=args.no_early_stopping,
        )
        stage_name = f"screen_stage{stage_idx:02d}_e{stage_ep}"
        stage_dir = screen_dir / f"stage_{stage_idx:02d}_e{stage_ep}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[{stage_name}] active_trials={len(active)} "
            f"(epochs={stage_ep}, patience={'None' if stage_patience is None else stage_patience})"
        )
        df_stage = _run_phase(
            phase_name=stage_name,
            configs=active,
            seeds=args.screen_seeds,
            args=args,
            phase_dir=stage_dir,
            num_epochs=stage_ep,
            patience=stage_patience,
            stage_index=stage_idx,
        )
        all_stage_frames.append(df_stage)

        successful_stage = df_stage[df_stage["status"] == "ok"].copy()
        if successful_stage.empty:
            print(f"[{stage_name}] no successful trials.")
            break

        ranked_stage = _rank_trials(successful_stage)
        ranked_stage.to_csv(run_dir / f"screen_stage_{stage_idx:02d}_ranked.tsv", sep="\t", index=False)
        last_stage_ranked = ranked_stage

        is_last_stage = (stage_idx == len(stage_epochs))
        if is_last_stage:
            break

        n_keep = max(
            args.prune_min_keep,
            int(math.ceil(len(ranked_stage) * args.prune_keep_ratio)),
            min(args.top_k, len(ranked_stage)),
        )
        n_keep = min(n_keep, len(ranked_stage))

        survivor_ids = [int(x) for x in ranked_stage.head(n_keep)["trial_id"].tolist()]
        print(
            f"[{stage_name}] keeping {n_keep}/{len(ranked_stage)} trials "
            f"for next stage (ratio={args.prune_keep_ratio})"
        )
        active = [(trial_id, id_to_cfg[trial_id]) for trial_id in survivor_ids]

    if all_stage_frames:
        df_screen_all = pd.concat(all_stage_frames, axis=0, ignore_index=True)
    else:
        df_screen_all = pd.DataFrame()
    return df_screen_all, last_stage_ranked


def main():
    parser = argparse.ArgumentParser(
        description="CrossAttention hyperparameter sweep (MCC-first selection)."
    )
    parser.add_argument("--dataset", "-d", choices=["human", "non_human", "all"], required=True)
    parser.add_argument("--embedding", "-e", choices=["8M", "150M", "650M"], default="8M")
    parser.add_argument("--profile", choices=["quick", "standard", "aggressive"], default="quick")
    parser.add_argument("--max_trials", type=int, default=None, help="Optional cap on number of trials.")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K configs for final rerun.")
    parser.add_argument(
        "--pruning_mode",
        choices=["none", "halving"],
        default="halving",
        help="Trial pruning strategy during screening (default: halving).",
    )
    parser.add_argument(
        "--prune_epochs",
        type=str,
        default="80,200",
        help="Comma-separated early stage epoch budgets for pruning; final --epochs is always added.",
    )
    parser.add_argument(
        "--prune_keep_ratio",
        type=float,
        default=0.5,
        help="Fraction of trials kept between pruning stages (default: 0.5).",
    )
    parser.add_argument(
        "--prune_min_keep",
        type=int,
        default=3,
        help="Minimum number of trials kept between pruning stages (default: 3).",
    )

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-early-stopping", action="store_true")

    parser.add_argument("--screen_seeds", nargs="+", type=int, default=DEFAULT_SCREEN_SEEDS)
    parser.add_argument("--final_seeds", nargs="+", type=int, default=DEFAULT_FINAL_SEEDS)

    parser.add_argument("--classification_weight", type=float, default=1.0)
    parser.add_argument("--regression_weight", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--threshold_metric", choices=["mcc", "f1", "balanced_accuracy"], default="mcc")
    parser.add_argument("--disable-threshold-optimization", action="store_true")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)

    parser.add_argument("--scaffold_split_dir", type=str, default="scaffolds_splits/output")
    parser.add_argument("--external_test_mode", action="store_true", default=True)
    parser.add_argument("--molformer_ligand", action="store_true", default=True)
    parser.add_argument("--smited_ligand", action="store_true", help="Use SMI-TED instead of MoLFormer ligand matrices.")

    parser.add_argument("--output_dir", "-o", type=str, default="results/crossattention_hparam_sweep")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.smited_ligand:
        args.molformer_ligand = False

    if args.max_trials is not None and args.max_trials <= 0:
        raise ValueError("--max_trials must be > 0 when provided")
    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")
    if not (0.0 < args.prune_keep_ratio <= 1.0):
        raise ValueError("--prune_keep_ratio must be in (0, 1]")
    if args.prune_min_keep <= 0:
        raise ValueError("--prune_min_keep must be > 0")

    configs = build_search_space(args.profile)
    if args.max_trials is not None:
        configs = configs[: args.max_trials]
    indexed_configs = [(i + 1, cfg) for i, cfg in enumerate(configs)]
    stage_epochs_preview = [args.epochs] if args.pruning_mode == "none" else _parse_pruning_epochs(args.prune_epochs, args.epochs)

    print("=" * 80)
    print("CROSSATTENTION HYPERPARAMETER SWEEP (MCC-FIRST)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Embedding: {args.embedding}")
    print(f"Profile: {args.profile}")
    print(f"Trials: {len(indexed_configs)}")
    print(f"Screen seeds: {args.screen_seeds}")
    print(f"Final seeds: {args.final_seeds}")
    print(f"Threshold metric: {args.threshold_metric}")
    print(f"Pruning mode: {args.pruning_mode}")
    print(f"Pruning stages (epochs): {stage_epochs_preview}")
    print(f"External test mode: {args.external_test_mode}")
    print(f"MoLFormer ligand: {args.molformer_ligand}")
    print("=" * 80)

    if args.dry_run:
        preview = pd.DataFrame(
            [{"trial_id": idx, **cfg.to_dict()} for idx, cfg in indexed_configs]
        )
        print(preview.to_string(index=False))
        return

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.dataset}_{args.embedding}_{args.profile}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    screen_dir = run_dir / "screen"
    final_dir = run_dir / "final"
    screen_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    df_screen_all, df_screen_last_stage_ranked = _run_screen_with_pruning(
        indexed_configs=indexed_configs,
        args=args,
        run_dir=run_dir,
        screen_dir=screen_dir,
    )
    df_screen_all.to_csv(run_dir / "screen_results.tsv", sep="\t", index=False)

    if df_screen_last_stage_ranked.empty or "status" not in df_screen_last_stage_ranked.columns:
        print("\nNo successful trial in screening phase.")
        return

    successful_screen = df_screen_last_stage_ranked[df_screen_last_stage_ranked["status"] == "ok"].copy()
    if successful_screen.empty:
        print("\nNo successful trial in screening phase.")
        return

    top_rows = successful_screen.head(args.top_k).copy()
    top_ids = [int(x) for x in top_rows["trial_id"].tolist()]
    config_map = {idx: cfg for idx, cfg in indexed_configs}
    top_configs = [(idx, config_map[idx]) for idx in top_ids]

    rerun_final = args.final_seeds != args.screen_seeds
    if rerun_final:
        final_patience = _stage_patience(
            base_patience=args.patience,
            stage_epochs=args.epochs,
            no_early_stopping=args.no_early_stopping,
        )
        df_final = _run_phase(
            phase_name="final",
            configs=top_configs,
            seeds=args.final_seeds,
            args=args,
            phase_dir=final_dir,
            num_epochs=args.epochs,
            patience=final_patience,
            stage_index=None,
        )
        df_final_ranked = _rank_trials(df_final)
        df_final_ranked.to_csv(run_dir / "final_results.tsv", sep="\t", index=False)
        successful_final = df_final_ranked[df_final_ranked["status"] == "ok"].copy()
        winner_row = successful_final.iloc[0] if not successful_final.empty else top_rows.iloc[0]
    else:
        df_final_ranked = pd.DataFrame()
        winner_row = top_rows.iloc[0]

    winner_id = int(winner_row["trial_id"])
    winner_cfg = config_map[winner_id]
    best_output_dir = str((final_dir / f"trial_{winner_id:03d}") if rerun_final else (screen_dir / f"trial_{winner_id:03d}"))
    best_cmd = _build_main_command(args, winner_cfg, best_output_dir)

    summary = {
        "dataset": args.dataset,
        "embedding": args.embedding,
        "profile": args.profile,
        "selection_rule": "maximize MCC, tie-break by accuracy, then minimize loss",
        "threshold_metric": args.threshold_metric,
        "pruning": {
            "mode": args.pruning_mode,
            "keep_ratio": args.prune_keep_ratio,
            "min_keep": args.prune_min_keep,
            "stage_epochs": stage_epochs_preview,
        },
        "winner_trial_id": winner_id,
        "winner_metrics": {
            "mcc": winner_row.get("mcc"),
            "accuracy": winner_row.get("accuracy"),
            "loss": winner_row.get("loss"),
            "decision_threshold": winner_row.get("decision_threshold"),
            "mcc_std": winner_row.get("mcc_std"),
            "accuracy_std": winner_row.get("accuracy_std"),
        },
        "winner_config": winner_cfg.to_dict(),
        "recommended_command": best_cmd,
    }

    with open(run_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(run_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("CrossAttention Hyperparameter Sweep Artifacts\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write("Selection rule: maximize MCC, tie-break by accuracy, then minimize loss.\n\n")
        f.write(f"Pruning mode: {args.pruning_mode}\n")
        f.write(f"Pruning stages (epochs): {stage_epochs_preview}\n\n")
        f.write("Recommended command:\n")
        f.write(best_cmd + "\n")

    print("\n" + "=" * 80)
    print("SWEEP FINISHED")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Best trial: {winner_id:03d}")
    print(
        f"Best metrics: MCC={_as_float(winner_row.get('mcc'), float('nan')):.4f}, "
        f"Acc={_as_float(winner_row.get('accuracy'), float('nan')):.4f}, "
        f"Loss={_as_float(winner_row.get('loss'), float('nan')):.4f}"
    )
    print(f"Saved: {run_dir / 'best_config.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
