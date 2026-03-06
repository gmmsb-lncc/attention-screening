#!/usr/bin/env python3
"""Unified benchmark orchestrator for semantic-screening model comparison.

Coordinates the full pipeline:
  Step 0:  Verify / generate scaffold splits
  Step 0b: Verify / extract ligand vectors (if Level 2 requested)
  Step FT: ESM-2 + MolFormer Fine-tuning (if --finetune flag set)
  Step 1:  Level 1 — Fingerprint + KNN/MLP  (baseline)
  Step 2:  Level 2 — Embedding vectors + KNN/MLP (with Attention Pooling)
  Step 3:  Level 3 — Embedding matrices + Mean Pooling + KNN/MLP
  Step 4:  Level 4 — Transformer + Cross-Attention + KNN/MLP
  Report:  Comparative report and visualizations

Usage:
    python semantic_screening_models.py --dataset human --embedding 8M --levels 1 2 3 4
    python semantic_screening_models.py --dataset human --embedding 8M --levels 1 2 3 --finetune
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
