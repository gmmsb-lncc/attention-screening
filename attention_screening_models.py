#!/usr/bin/env python3
"""Unified benchmark orchestrator for attention-screening model comparison.

Coordinates the full pipeline:
  Step 0:  Verify / generate scaffold splits
  Step FT: ESM-2 + MolFormer Fine-tuning (if --finetune flag set)
  Step 1a: Level 1a — Fingerprint + KNN/MLP (compound-only baseline)
  Step 1b: Level 1b — Ligand MoLFormer mean pooling + KNN/MLP
  Step 1c: Level 1c — Ligand MoLFormer attention pooling + KNN/MLP
  Step 2:  Level 2  — Protein+Ligand mean pooling + KNN/MLP
  Step 3:  Level 3  — Protein+Ligand attention pooling + KNN/MLP
  Step 4:  Level 4  — Protein+Ligand cross-attention + KNN/MLP
  Note: Levels after 4 (5a, 5b, 6a, 6b) are obsolete.
  Report:  Comparative report and visualizations

Monotonic complexity hierarchy:
  1a (FP) < 1b (lig mean) < 1c (lig attn) < 2 (prot+lig mean)
  < 3 (prot+lig attn)

Evaluation protocol:
  - All active levels use the **exact same** canonical KNN and MLP classifiers
    (benchmark.classifiers), ensuring the only variable is the representation.
  - Classifiers are trained on **validation-split** features and evaluated
    on the **test** split, eliminating train-set optimism for levels with
    learned feature extractors (1c, 3).

Usage:
  python attention_screening_models.py --dataset human --embedding 8M --levels 1a 1b 1c 2 3 4
  python attention_screening_models.py --dataset human --embedding 8M --levels 1a 2 3 4 --finetune
  python attention_screening_models.py --dataset human --embedding 8M --levels 1b 1c 2 3 4 --se3 --se3-features-dir path/to/se3_features
"""

from benchmark.cli import build_parser, config_from_args
from benchmark.orchestrator import BenchmarkOrchestrator


def main() -> None:
    """Parse CLI arguments, build config, and run the benchmark."""
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    orchestrator = BenchmarkOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
