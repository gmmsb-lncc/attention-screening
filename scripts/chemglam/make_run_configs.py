#!/usr/bin/env python3
"""Materialize train/validation/test ChemGLaM configs for one seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("configs/chemglam_universal.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    run_name = f"chemglam_universal_seed{args.seed}"
    checkpoint = str(Path("logs") / run_name / "best_checkpoint.ckpt")
    args.output.mkdir(parents=True, exist_ok=True)

    train = dict(base)
    train.update(seed=args.seed, experiment_name=run_name, checkpoint_path=None)

    def prediction(split: str) -> dict:
        config = dict(train)
        config.update(
            experiment_name=f"{run_name}_{split}",
            cache_dir=f"chemglam_universal_{split}",
            dataset_csv_path=f"data/chemglam/universal/{split}.csv",
            split_json_path=None,
            checkpoint_path=checkpoint,
            deterministic_eval=True,
        )
        return config

    for name, config in (("train", train), ("val", prediction("val")), ("test", prediction("test"))):
        (args.output / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
