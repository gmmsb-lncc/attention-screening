#!/usr/bin/env python3
"""Build fixed-test scaffold splits and scenario-specific train/val partitions."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from scaffolds_splits.data_io import load_dataset
from scaffolds_splits.scaffold_utils import (
    UNKNOWN_SCAFFOLD,
    attach_scaffolds,
    build_compound_scaffold_table,
    scaffold_stats,
)
from scaffolds_splits.scenario_splitter import (
    SCENARIO_NAMES,
    SCENARIO_ORDER,
    ScenarioSplitConfig,
    build_split_distribution_row,
    split_train_val_by_scenario,
    validate_scenario_split,
)
from scaffolds_splits.selection import UniversalSelectionConfig, select_universal_test_scaffolds
from scaffolds_splits.splitter import ValidationSelectionConfig, select_test_scaffolds
from scaffolds_splits.validation import validate_universal_test_scaffolds
from scaffolds_splits.writer import (
    write_combined_test,
    write_distribution_summary,
    write_manifest,
    write_scenario_splits,
    write_universal_scaffolds,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-test scaffold split + scenario train/val generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--human-input",
        default="tests/datasets/kinase_human_compounds.tsv",
        help="Path to human TSV dataset",
    )
    parser.add_argument(
        "--non-human-input",
        default="tests/datasets/kinase_non_human_compounds.tsv",
        help="Path to non-human TSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="scaffolds_splits/output",
        help="Directory to store split outputs and manifest",
    )
    parser.add_argument(
        "--threshold-pchembl",
        type=float,
        default=6.0,
        help="Binary label threshold: label=1 if pChEMBL >= threshold",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--target-test-frac",
        type=float,
        default=0.10,
        help="Target test fraction on unique compounds",
    )
    parser.add_argument(
        "--target-val-frac",
        type=float,
        default=0.10,
        help="Target validation fraction on total rows (reported at dataset level)",
    )
    parser.add_argument(
        "--test-mode",
        choices=["per_dataset", "shared_scaffold"],
        default="shared_scaffold",
        help=(
            "Test selection strategy. "
            "per_dataset: each dataset gets its own scaffold test set near target fraction. "
            "shared_scaffold: both datasets use exactly the same scaffold test set."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default="S1,S2,S3,S4,Sc",
        help="Comma-separated scenario list. Supported: S1,S2,S3,S4,Sc or 'all'",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=64,
        help="Random restarts for test scaffold optimization",
    )
    parser.add_argument(
        "--scenario-restarts",
        type=int,
        default=16,
        help="Random restarts for grouped scenario train/val splitting",
    )
    parser.add_argument(
        "--s4-restarts",
        type=int,
        default=192,
        help="Random restarts for S4 (compound+kinase disjoint) splitting",
    )
    parser.add_argument(
        "--class-penalty",
        type=float,
        default=10.0,
        help="Penalty when class support constraints are violated",
    )
    parser.add_argument(
        "--class-rate-weight",
        type=float,
        default=2.0,
        help="Weight for class-rate preservation (train/val vs pool) in scenario splitting",
    )
    parser.add_argument(
        "--weight-human",
        type=float,
        default=8.0,
        help="Weight for human test-fraction deviation in shared_scaffold mode",
    )
    parser.add_argument(
        "--weight-non-human",
        type=float,
        default=1.0,
        help="Weight for non-human test-fraction deviation in shared_scaffold mode",
    )
    parser.add_argument(
        "--weight-ratio",
        type=float,
        default=0.1,
        help="Weight for human/non-human proportionality term in shared_scaffold mode",
    )
    parser.add_argument(
        "--keep-monotonic-kinases",
        action="store_true",
        help="Keep monotonic kinases (default removes them)",
    )
    parser.add_argument(
        "--keep-monotonic-compounds",
        action="store_true",
        help="Keep monotonic compounds (default removes them)",
    )
    parser.add_argument(
        "--max-rows-human",
        type=int,
        default=None,
        help="Debug-only: limit rows loaded from human input",
    )
    parser.add_argument(
        "--max-rows-non-human",
        type=int,
        default=None,
        help="Debug-only: limit rows loaded from non-human input",
    )
    # Backward-compatible no-op: combined file is always written now.
    parser.add_argument(
        "--write-combined-test",
        action="store_true",
        help="Deprecated: universal_test.tsv is always written",
    )
    return parser


def _parse_scenarios(raw: str) -> List[str]:
    aliases = {
        "s1": "S1",
        "random": "S1",
        "s2": "S2",
        "compound": "S2",
        "s3": "S3",
        "kinase": "S3",
        "s4": "S4",
        "new_compound_new_kinase": "S4",
        "sc": "Sc",
        "scaffold": "Sc",
    }

    txt = raw.strip()
    if txt.lower() == "all":
        return list(SCENARIO_ORDER)

    items = [x.strip() for x in txt.split(",") if x.strip()]
    if not items:
        raise ValueError("--scenarios produced an empty list")

    out: List[str] = []
    for item in items:
        key = item.lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown scenario '{item}'. Supported: {list(SCENARIO_ORDER)} or aliases random/compound/kinase/scaffold"
            )
        canonical = aliases[key]
        if canonical not in out:
            out.append(canonical)

    return out


def _count_unique_compounds_for_scaffolds(stats_df, scaffolds: Set[str]) -> int:
    if not scaffolds:
        return 0
    return int(stats_df[stats_df["scaffold"].isin(scaffolds)]["unique_compounds"].sum())


def _class_summary(df) -> Dict[str, float]:
    rows = int(len(df))
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return {
        "rows": rows,
        "pos_rows": pos,
        "neg_rows": neg,
        "pos_pct": 100.0 * pos / max(rows, 1),
        "neg_pct": 100.0 * neg / max(rows, 1),
        "unique_compounds": int(df["chembl_id"].nunique()),
        "unique_kinases": int(df["target_kinase"].nunique()) if "target_kinase" in df.columns else 0,
        "unique_scaffolds": int(df["scaffold"].nunique()) if "scaffold" in df.columns else 0,
    }


def _validate_fixed_test(dataset_name: str, test_df) -> None:
    if test_df.empty:
        raise ValueError(f"{dataset_name}: fixed test split is empty")

    labels = set(test_df["label"].astype(int).unique().tolist())
    if not {0, 1}.issubset(labels):
        raise ValueError(
            f"{dataset_name}: fixed test split must contain labels 0 and 1; got {sorted(labels)}"
        )


def _compute_val_fraction_in_pool(total_rows: int, test_rows: int, target_val_fraction_total: float) -> float:
    remaining = total_rows - test_rows
    if remaining <= 0:
        raise ValueError("No rows left after fixed test allocation")

    frac = target_val_fraction_total * total_rows / remaining
    return float(min(0.49, max(0.01, frac)))


def _write_top_level_from_canonical(
    out_dir: Path,
    dataset_name: str,
    canonical_train_path: str,
    canonical_val_path: str,
    test_df,
) -> Dict[str, str]:
    train_path = out_dir / f"{dataset_name}_train.tsv"
    val_path = out_dir / f"{dataset_name}_val.tsv"
    test_path = out_dir / f"{dataset_name}_test.tsv"

    shutil.copyfile(canonical_train_path, train_path)
    shutil.copyfile(canonical_val_path, val_path)
    test_df.to_csv(test_path, sep="\t", index=False)

    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


def _run_dataset_scenarios(
    dataset_name: str,
    full_df,
    test_df,
    pool_df,
    scenarios: List[str],
    args,
    out_dir: Path,
    seed_offset: int,
) -> Dict[str, object]:
    total_rows = int(len(full_df))
    test_rows = int(len(test_df))
    val_fraction_pool = _compute_val_fraction_in_pool(
        total_rows=total_rows,
        test_rows=test_rows,
        target_val_fraction_total=args.target_val_frac,
    )

    scenario_outputs: Dict[str, Dict[str, str]] = {}
    scenario_summaries: Dict[str, Dict[str, object]] = {}
    distribution_rows: List[Dict[str, float]] = []
    text_lines: List[str] = []

    test_stats = _class_summary(test_df)

    for i, scenario_code in enumerate(scenarios):
        split_cfg = ScenarioSplitConfig(
            val_fraction_in_pool=val_fraction_pool,
            seed=args.seed + seed_offset + i * 101,
            restarts=args.scenario_restarts,
            s4_restarts=args.s4_restarts,
            class_penalty=args.class_penalty,
            class_rate_weight=args.class_rate_weight,
        )
        split_result = split_train_val_by_scenario(
            remainder_df=pool_df,
            scenario_code=scenario_code,
            config=split_cfg,
        )

        validate_scenario_split(scenario_code, split_result.train_df, split_result.val_df)

        paths = write_scenario_splits(
            scenario_code=scenario_code,
            dataset_name=dataset_name,
            train_df=split_result.train_df,
            val_df=split_result.val_df,
            dropped_df=split_result.dropped_df,
            output_dir=str(out_dir),
        )
        scenario_outputs[scenario_code] = paths

        train_stats = _class_summary(split_result.train_df)
        val_stats = _class_summary(split_result.val_df)
        dropped_rows = int(len(split_result.dropped_df))

        scenario_summaries[scenario_code] = {
            "scenario_name": SCENARIO_NAMES[scenario_code],
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "dropped_rows": dropped_rows,
            "split_metrics": split_result.metrics,
            "target_val_fraction_in_pool": val_fraction_pool,
        }

        distribution_rows.append(
            build_split_distribution_row(
                dataset_name=dataset_name,
                scenario_code=scenario_code,
                split_name="train",
                split_df=split_result.train_df,
                total_rows_dataset=total_rows,
                dropped_rows=dropped_rows,
            )
        )
        distribution_rows.append(
            build_split_distribution_row(
                dataset_name=dataset_name,
                scenario_code=scenario_code,
                split_name="val",
                split_df=split_result.val_df,
                total_rows_dataset=total_rows,
                dropped_rows=dropped_rows,
            )
        )
        distribution_rows.append(
            build_split_distribution_row(
                dataset_name=dataset_name,
                scenario_code=scenario_code,
                split_name="test",
                split_df=test_df,
                total_rows_dataset=total_rows,
                dropped_rows=dropped_rows,
            )
        )

        line = (
            f"[{dataset_name}][{scenario_code}] "
            f"Train: +{train_stats['pos_pct']:.2f}% / -{train_stats['neg_pct']:.2f}% | "
            f"Val: +{val_stats['pos_pct']:.2f}% / -{val_stats['neg_pct']:.2f}% | "
            f"Test: +{test_stats['pos_pct']:.2f}% / -{test_stats['neg_pct']:.2f}% | "
            f"dropped={dropped_rows:,}"
        )
        print(line)
        text_lines.append(line)

    canonical = "Sc" if "Sc" in scenarios else scenarios[0]
    canonical_paths = scenario_outputs[canonical]
    top_level_paths = _write_top_level_from_canonical(
        out_dir=out_dir,
        dataset_name=dataset_name,
        canonical_train_path=canonical_paths["train"],
        canonical_val_path=canonical_paths["val"],
        test_df=test_df,
    )

    return {
        "dataset_name": dataset_name,
        "scenario_outputs": scenario_outputs,
        "scenario_summaries": scenario_summaries,
        "distribution_rows": distribution_rows,
        "summary_lines": text_lines,
        "top_level_paths": top_level_paths,
        "canonical_scenario": canonical,
        "target_val_fraction_in_pool": val_fraction_pool,
    }


def main() -> None:
    args = _build_parser().parse_args()
    scenarios = _parse_scenarios(args.scenarios)

    print("=" * 80)
    print("FIXED TEST + SCENARIO TRAIN/VAL SPLIT")
    print("=" * 80)
    print(f"Human input:      {args.human_input}")
    print(f"Non-human input:  {args.non_human_input}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Test mode:        {args.test_mode}")
    print(f"Scenarios:        {scenarios}")
    print(f"Monotonic filters: kinases={not args.keep_monotonic_kinases}, compounds={not args.keep_monotonic_compounds}")
    print("=" * 80)

    human_df = load_dataset(
        args.human_input,
        dataset_name="human",
        threshold_pchembl=args.threshold_pchembl,
        max_rows=args.max_rows_human,
        remove_monotonic_kinases=not args.keep_monotonic_kinases,
        remove_monotonic_compounds=not args.keep_monotonic_compounds,
    )
    non_human_df = load_dataset(
        args.non_human_input,
        dataset_name="non_human",
        threshold_pchembl=args.threshold_pchembl,
        max_rows=args.max_rows_non_human,
        remove_monotonic_kinases=not args.keep_monotonic_kinases,
        remove_monotonic_compounds=not args.keep_monotonic_compounds,
    )

    # Build scaffold annotations.
    human_compounds = build_compound_scaffold_table(human_df, dataset_name="human")
    non_human_compounds = build_compound_scaffold_table(non_human_df, dataset_name="non_human")

    human_df = attach_scaffolds(human_df, human_compounds)
    non_human_df = attach_scaffolds(non_human_df, non_human_compounds)

    human_stats = scaffold_stats(human_df)
    non_human_stats = scaffold_stats(non_human_df)

    total_unique_h = int(human_df["chembl_id"].nunique())
    total_unique_n = int(non_human_df["chembl_id"].nunique())

    # Select fixed test scaffolds.
    test_cfg_h = ValidationSelectionConfig(
        target_val_fraction=args.target_test_frac,
        seed=args.seed + 11,
        restarts=args.restarts,
        class_penalty=args.class_penalty,
    )
    test_cfg_n = ValidationSelectionConfig(
        target_val_fraction=args.target_test_frac,
        seed=args.seed + 22,
        restarts=args.restarts,
        class_penalty=args.class_penalty,
    )

    if args.test_mode == "shared_scaffold":
        selection_cfg = UniversalSelectionConfig(
            target_test_fraction=args.target_test_frac,
            seed=args.seed,
            restarts=args.restarts,
            weight_human=args.weight_human,
            weight_non_human=args.weight_non_human,
            weight_ratio=args.weight_ratio,
            class_penalty=args.class_penalty,
        )
        shared = select_universal_test_scaffolds(
            human_stats=human_stats,
            non_human_stats=non_human_stats,
            total_unique_human=total_unique_h,
            total_unique_non_human=total_unique_n,
            config=selection_cfg,
            unknown_scaffold=UNKNOWN_SCAFFOLD,
        )
        human_test_scaffolds = set(shared.test_scaffolds)
        non_human_test_scaffolds = set(shared.test_scaffolds)

        test_selection_metrics = {
            "mode": "shared_scaffold",
            **shared.metrics,
        }
        universal_scaffold_payload = {
            "mode": "shared_scaffold",
            "n_scaffolds": len(shared.test_scaffolds),
            "shared_scaffolds": sorted(shared.test_scaffolds),
        }
        print(
            "Shared test scaffolds selected: "
            f"n={len(shared.test_scaffolds)} | "
            f"human_test={shared.metrics['test_fraction_human']:.4f} | "
            f"non_human_test={shared.metrics['test_fraction_non_human']:.4f}"
        )
    else:
        human_test_scaffolds = set(
            select_test_scaffolds(
                stats_df=human_stats,
                total_unique_compounds=total_unique_h,
                config=test_cfg_h,
                excluded_scaffolds={UNKNOWN_SCAFFOLD},
            )
        )
        non_human_test_scaffolds = set(
            select_test_scaffolds(
                stats_df=non_human_stats,
                total_unique_compounds=total_unique_n,
                config=test_cfg_n,
                excluded_scaffolds={UNKNOWN_SCAFFOLD},
            )
        )

        h_test_comp = _count_unique_compounds_for_scaffolds(human_stats, human_test_scaffolds)
        n_test_comp = _count_unique_compounds_for_scaffolds(non_human_stats, non_human_test_scaffolds)
        h_frac = h_test_comp / max(total_unique_h, 1)
        n_frac = n_test_comp / max(total_unique_n, 1)

        test_selection_metrics = {
            "mode": "per_dataset",
            "target_test_fraction": args.target_test_frac,
            "human_test_fraction": h_frac,
            "non_human_test_fraction": n_frac,
            "human_test_compounds": h_test_comp,
            "non_human_test_compounds": n_test_comp,
            "human_test_scaffolds": len(human_test_scaffolds),
            "non_human_test_scaffolds": len(non_human_test_scaffolds),
            "intersection_scaffolds": len(human_test_scaffolds & non_human_test_scaffolds),
            "union_scaffolds": len(human_test_scaffolds | non_human_test_scaffolds),
        }
        universal_scaffold_payload = {
            "mode": "per_dataset",
            "target_test_fraction": args.target_test_frac,
            "human": {
                "n_scaffolds": len(human_test_scaffolds),
                "scaffolds": sorted(human_test_scaffolds),
            },
            "non_human": {
                "n_scaffolds": len(non_human_test_scaffolds),
                "scaffolds": sorted(non_human_test_scaffolds),
            },
            "intersection_scaffolds": sorted(human_test_scaffolds & non_human_test_scaffolds),
            "union_scaffolds": sorted(human_test_scaffolds | non_human_test_scaffolds),
        }
        print(
            "Per-dataset test scaffolds selected: "
            f"human={len(human_test_scaffolds)} ({h_frac:.4f}) | "
            f"non_human={len(non_human_test_scaffolds)} ({n_frac:.4f})"
        )

    # Materialize fixed test + remainder pools.
    human_test_mask = human_df["scaffold"].isin(human_test_scaffolds)
    non_human_test_mask = non_human_df["scaffold"].isin(non_human_test_scaffolds)

    human_test_df = human_df.loc[human_test_mask].copy().reset_index(drop=True)
    non_human_test_df = non_human_df.loc[non_human_test_mask].copy().reset_index(drop=True)

    human_pool_df = human_df.loc[~human_test_mask].copy().reset_index(drop=True)
    non_human_pool_df = non_human_df.loc[~non_human_test_mask].copy().reset_index(drop=True)

    _validate_fixed_test("human", human_test_df)
    _validate_fixed_test("non_human", non_human_test_df)
    if args.test_mode == "shared_scaffold":
        validate_universal_test_scaffolds(
            human_test_df["scaffold"].astype(str).unique().tolist(),
            non_human_test_df["scaffold"].astype(str).unique().tolist(),
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaffold_path = write_universal_scaffolds(universal_scaffold_payload, str(out_dir))

    # Generate per-scenario train/val for each dataset using the same fixed test.
    print("\nClass distribution by scenario:")
    human_run = _run_dataset_scenarios(
        dataset_name="human",
        full_df=human_df,
        test_df=human_test_df,
        pool_df=human_pool_df,
        scenarios=scenarios,
        args=args,
        out_dir=out_dir,
        seed_offset=100_000,
    )
    non_human_run = _run_dataset_scenarios(
        dataset_name="non_human",
        full_df=non_human_df,
        test_df=non_human_test_df,
        pool_df=non_human_pool_df,
        scenarios=scenarios,
        args=args,
        out_dir=out_dir,
        seed_offset=200_000,
    )

    # Backward-compatible top-level train/val/test files use canonical scenario.
    human_top_paths = human_run["top_level_paths"]
    non_human_top_paths = non_human_run["top_level_paths"]

    combined_test_path = write_combined_test(
        human_test_df=human_test_df,
        non_human_test_df=non_human_test_df,
        output_dir=str(out_dir),
    )

    distribution_rows = list(human_run["distribution_rows"]) + list(non_human_run["distribution_rows"])
    distribution_path = write_distribution_summary(distribution_rows, str(out_dir))

    text_report_path = out_dir / "split_class_distribution_report.txt"
    text_lines = list(human_run["summary_lines"]) + list(non_human_run["summary_lines"])
    text_report_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "human": args.human_input,
            "non_human": args.non_human_input,
        },
        "config": {
            "threshold_pchembl": args.threshold_pchembl,
            "seed": args.seed,
            "target_test_fraction": args.target_test_frac,
            "target_val_fraction": args.target_val_frac,
            "test_mode": args.test_mode,
            "scenarios": scenarios,
            "restarts": args.restarts,
            "scenario_restarts": args.scenario_restarts,
            "s4_restarts": args.s4_restarts,
            "class_penalty": args.class_penalty,
            "class_rate_weight": args.class_rate_weight,
            "weight_human": args.weight_human,
            "weight_non_human": args.weight_non_human,
            "weight_ratio": args.weight_ratio,
            "remove_monotonic_kinases": not args.keep_monotonic_kinases,
            "remove_monotonic_compounds": not args.keep_monotonic_compounds,
            "max_rows_human": args.max_rows_human,
            "max_rows_non_human": args.max_rows_non_human,
            "val_fraction_in_pool_policy": "row_based_relative_to_remaining_pool",
        },
        "test_selection": test_selection_metrics,
        "scenario_summaries": {
            "human": human_run["scenario_summaries"],
            "non_human": non_human_run["scenario_summaries"],
        },
        "canonical_scenario": {
            "human": human_run["canonical_scenario"],
            "non_human": non_human_run["canonical_scenario"],
        },
        "outputs": {
            "human": human_top_paths,
            "non_human": non_human_top_paths,
            "scenario_outputs": {
                "human": human_run["scenario_outputs"],
                "non_human": non_human_run["scenario_outputs"],
            },
            "universal_scaffolds": scaffold_path,
            "combined_test": combined_test_path,
            "class_distribution_summary": distribution_path,
            "class_distribution_report": str(text_report_path),
        },
    }

    manifest_path = write_manifest(manifest, str(out_dir))

    print("\nExecution finished successfully.")
    print(f"Manifest: {manifest_path}")
    print(f"Universal scaffold payload: {scaffold_path}")
    print(f"Combined test: {combined_test_path}")
    print(f"Class distribution summary: {distribution_path}")
    print(f"Class distribution report: {text_report_path}")


if __name__ == "__main__":
    main()
