#!/usr/bin/env python3
"""Run DT-Kinase benchmark from a YAML config file.

Usage:
    python run_from_config.py configs/non_human_v7.yaml
    python run_from_config.py configs/human_v7.yaml --dry-run

This script:
  1. Reads the YAML config (lightweight parser, no PyYAML dependency)
  2. Maps each field to the corresponding environment variable
  3. Invokes run_benchmark.sh with those env vars exported
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# ======================================================================
# Lightweight YAML parser (flat key-value + one nesting level)
# No external dependencies required.
# ======================================================================

def _parse_yaml(path: Path) -> dict:
    """Parse a simple YAML file into a nested dict (max 2 levels deep).

    Supports:
      - key: value       → top-level scalar
      - key:             → start of nested section
      -   subkey: value  → nested scalar
      - # comments       → ignored
      - boolean: true/false/yes/no → Python bool
      - numbers: 0.35, 500 → Python float/int
    """
    result: dict = {}
    current_section: str | None = None

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip()

            # Skip empty lines and comments
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            # Detect indentation level
            indent = len(line) - len(stripped)

            # Parse key: value
            match = re.match(r'^(\w[\w_]*)\s*:\s*(.*)', stripped)
            if not match:
                continue

            key = match.group(1)
            val_str = match.group(2).strip()

            # Remove inline comments
            if val_str and "#" in val_str:
                val_str = val_str[:val_str.index("#")].strip()

            # Convert value
            val = _convert_value(val_str) if val_str else None

            if indent == 0:
                # Top-level key
                if val is None:
                    # Section header
                    result[key] = {}
                    current_section = key
                else:
                    result[key] = val
                    current_section = None
            elif indent > 0 and current_section is not None:
                # Nested key
                section = result.setdefault(current_section, {})
                if val is None:
                    # Sub-section (2nd nesting level)
                    section[key] = {}
                    # Track sub-section for next lines
                    result[f"_subsection"] = (current_section, key)
                else:
                    # Check if we're in a sub-section
                    sub = result.get("_subsection")
                    if sub and indent >= 4 and isinstance(
                        section.get(sub[1]), dict
                    ):
                        section[sub[1]][key] = val
                    else:
                        section[key] = val

    result.pop("_subsection", None)
    return result


def _convert_value(s: str):
    """Convert a YAML value string to a Python type."""
    if not s:
        return None
    lower = s.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ======================================================================
# Config → Environment Variables mapping
# ======================================================================

def _bool_to_env(val) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, str):
        return "1" if val.lower() in ("true", "1", "yes") else "0"
    return str(val)


def build_env(cfg: dict) -> dict[str, str]:
    """Map YAML config to environment variables."""
    env: dict[str, str] = {}

    # --- Top-level ---
    env["DATASET"] = str(cfg.get("dataset", "non_human"))
    env["EMBEDDING"] = str(cfg.get("embedding", "8M"))
    env["LEVELS_CSV"] = str(cfg.get("levels", "4cnn"))
    env["EPOCHS"] = str(cfg.get("epochs", 500))
    env["MODEL_SELECTION_METRIC"] = str(cfg.get("model_selection_metric", "mcc"))
    env["OUTPUT_ROOT"] = str(cfg.get("output_root", "results/benchmark"))

    # --- Warnings ---
    if cfg.get("suppress_warnings", True):
        env["PYTHONWARNINGS"] = "ignore::UserWarning"

    # --- Level 4 CNN ---
    cnn = cfg.get("level4_cnn", {})
    if not isinstance(cnn, dict):
        cnn = {}

    _scalar_map = {
        "variant":       "BENCHMARK_LEVEL4CNN_VARIANT",
        "num_heads":     "BENCHMARK_LEVEL4CNN_NUM_HEADS",
        "head_dim":      "BENCHMARK_LEVEL4CNN_HEAD_DIM",
        "channels":      "BENCHMARK_LEVEL4CNN_CHANNELS",
        "batch_size":    "BENCHMARK_LEVEL4CNN_BATCH_SIZE",
        "dropout":       "BENCHMARK_LEVEL4CNN_DROPOUT",
        "weight_decay":  "BENCHMARK_LEVEL4CNN_WEIGHT_DECAY",
        "patience":      "BENCHMARK_LEVEL4CNN_PATIENCE",
        "focal_gamma":   "BENCHMARK_LEVEL4CNN_FOCAL_GAMMA",
        "contrastive_weight": "BENCHMARK_LEVEL4CNN_CONTRASTIVE_WEIGHT",
        "contrastive_dim":    "BENCHMARK_LEVEL4CNN_CONTRASTIVE_DIM",
        "label_smooth":  "BENCHMARK_LEVEL4CNN_LABEL_SMOOTH",
        "mixup_alpha":   "BENCHMARK_LEVEL4CNN_MIXUP_ALPHA",
        "pool_num_heads": "BENCHMARK_LEVEL4CNN_POOL_HEADS",
        "train_to_zero_thr":  "BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR",
        "checkpoint_every":   "BENCHMARK_LEVEL4CNN_CHECKPOINT_EVERY",
    }
    for key, envvar in _scalar_map.items():
        if key in cnn:
            env[envvar] = str(cnn[key])

    _bool_map = {
        "double":        "BENCHMARK_LEVEL4CNN_DOUBLE",
        "no_amp":        "BENCHMARK_LEVEL4CNN_NO_AMP",
        "deterministic": "BENCHMARK_LEVEL4CNN_DETERMINISTIC",
        "focal":         "BENCHMARK_LEVEL4CNN_FOCAL",
        "mlp_head":      "BENCHMARK_LEVEL4CNN_MLP_HEAD",
        "cosine_sim":    "BENCHMARK_LEVEL4CNN_COSINE_SIM",
        "cosine_feat":   "BENCHMARK_LEVEL4CNN_COSINE_FEAT",
        "train_to_zero": "BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO",
    }
    for key, envvar in _bool_map.items():
        if key in cnn:
            env[envvar] = _bool_to_env(cnn[key])

    # --- Adapter ---
    adapter = cnn.get("adapter", {})
    if not isinstance(adapter, dict):
        adapter = {}
    if adapter.get("enabled", False):
        env["BENCHMARK_LEVEL4CNN_ADAPTER"] = "1"
        _adapter_map = {
            "layers":    "BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS",
            "lr_mult":   "BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT",
            "prot_dim":  "BENCHMARK_LEVEL4CNN_ADAPTER_PROT_DIM",
            "lig_dim":   "BENCHMARK_LEVEL4CNN_ADAPTER_LIG_DIM",
        }
        for key, envvar in _adapter_map.items():
            if key in adapter:
                env[envvar] = str(adapter[key])
        if "self_attn" in adapter:
            env["BENCHMARK_LEVEL4CNN_ADAPTER_SELF_ATTN"] = _bool_to_env(
                adapter["self_attn"]
            )
    else:
        env["BENCHMARK_LEVEL4CNN_ADAPTER"] = "0"

    # --- Calibration ---
    cal = cnn.get("calibration", {})
    if not isinstance(cal, dict):
        cal = {}
    env["BENCHMARK_LEVEL4CNN_PLATT"] = _bool_to_env(cal.get("platt", True))
    env["BENCHMARK_LEVEL4CNN_TEMPERATURE"] = _bool_to_env(
        cal.get("temperature", False)
    )

    # --- Rigor ---
    rigor = cfg.get("rigor", {})
    if not isinstance(rigor, dict):
        rigor = {}
    env["BENCHMARK_ENFORCE_RIGOR"] = _bool_to_env(rigor.get("enforce", True))
    env["BENCHMARK_REQUIRE_TRAIN_SELECTION"] = _bool_to_env(
        rigor.get("require_train_selection", True)
    )
    env["BENCHMARK_STRICT_LEVEL_COMPLETENESS"] = _bool_to_env(
        rigor.get("strict_level_completeness", True)
    )

    # --- Gate ---
    gate = cfg.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}
    if "target_test_mcc" in gate:
        env["TARGET_TEST_MCC"] = str(gate["target_test_mcc"])
    if "max_test_mcc_std" in gate:
        env["MAX_TEST_MCC_STD"] = str(gate["max_test_mcc_std"])

    return env


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DT-Kinase benchmark from YAML config"
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to YAML config file (e.g. configs/v7.yaml)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["non_human", "human", "all"],
        help="Override dataset (non_human | human | all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the environment variables and command without executing"
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _parse_yaml(args.config)

    # CLI --dataset overrides YAML default
    if args.dataset:
        cfg["dataset"] = args.dataset

    # Resolve {dataset} and {embedding} placeholders in output_root
    dataset = str(cfg.get("dataset", "non_human"))
    embedding = str(cfg.get("embedding", "8M"))
    output_root = str(cfg.get("output_root", "results/benchmark"))
    output_root = output_root.replace("{dataset}", dataset)
    output_root = output_root.replace("{embedding}", embedding)
    cfg["output_root"] = output_root

    env_vars = build_env(cfg)

    # Build full environment (inherit current + overlay config)
    full_env = {**os.environ, **env_vars}

    # Print config summary
    print("=" * 64)
    print(f" DT-Kinase Benchmark | config: {args.config.name}")
    print(f" Dataset: {dataset} | Embedding: {embedding}")
    print("=" * 64)
    for key in sorted(env_vars):
        print(f"  {key}={env_vars[key]}")
    print("=" * 64)

    if args.dry_run:
        print("\n[DRY RUN] Would execute: bash run_benchmark.sh")
        print("Exiting without running.")
        return

    # Execute run_benchmark.sh with the configured environment
    script = Path(__file__).parent / "run_benchmark.sh"
    if not script.exists():
        print(f"Error: run_benchmark.sh not found at {script}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        ["bash", str(script)],
        env=full_env,
        cwd=str(Path(__file__).parent),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

