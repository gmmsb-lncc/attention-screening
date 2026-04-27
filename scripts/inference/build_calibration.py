"""Build level4_cnn_calibration.json sidecar next to a v7+F checkpoint.

Wraps scripts/thesis_followups/eval_checkpoint_on_dataset.py: runs the
canonical Platt + MCC-threshold fitting on the training corpus val set,
then extracts (a, b, threshold) into a small sidecar JSON consumed by
scripts/inference/models/dtkinase_score.py at inference time.

Usage:
    python scripts/inference/build_calibration.py \\
        --checkpoint results/benchmark_plusF_all_8M/test/level4_cnn_8M/all/seed_42/level4_cnn_model.pt \\
        --train-corpus all \\
        --seed 42

Writes:
    <ckpt_dir>/level4_cnn_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "thesis_followups" / "eval_checkpoint_on_dataset.py"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build inference calibration sidecar.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-corpus", choices=["human", "non_human", "all"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/v7_plus_F.yaml")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    # Run eval_checkpoint_on_dataset.py to produce a metrics JSON; we just
    # need (platt_a, platt_b, threshold) from it. Use train_corpus on both
    # sides so the fitted Platt + threshold come from the canonical val set.
    env = os.environ.copy()
    env.setdefault("BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY", "1")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--checkpoint", str(args.checkpoint),
            "--train-corpus", args.train_corpus,
            "--eval-dataset", args.train_corpus,
            "--split", "val",
            "--config", args.config,
            "--seed", str(args.seed),
            "--output", str(tmp_path),
        ]
        subprocess.run(cmd, check=True, env=env)
        metrics = json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

    sidecar = {
        "platt_a":   float(metrics["platt_a"]),
        "platt_b":   float(metrics["platt_b"]),
        "threshold": float(metrics["threshold"]),
        "corpus":    args.train_corpus,
        "n_val":     int(metrics["n_samples"]),
        "source_run": str(args.checkpoint),
        "config":    args.config,
        "seed":      args.seed,
    }
    out_path = args.checkpoint.parent / "level4_cnn_calibration.json"
    with open(out_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    print(f"wrote {out_path}")
    print(f"  platt_a={sidecar['platt_a']:.4f} "
          f"platt_b={sidecar['platt_b']:.4f} "
          f"thr={sidecar['threshold']:.3f}")


if __name__ == "__main__":
    main()
