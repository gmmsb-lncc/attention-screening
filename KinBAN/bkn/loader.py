"""BKN model loader — imports bundled GraphBAN source from ban-kinase-network/src/."""
from __future__ import annotations

import sys
from pathlib import Path

from .constants import BKN_SRC


def _ensure_src_on_path(src_dir: Path = BKN_SRC) -> None:
    """Add the bundled src/ directory to sys.path (idempotent)."""
    if not src_dir.exists():
        raise RuntimeError(
            f"BKN src/ not found at {src_dir}\n"
            "Run: python setup_bkn.py  (fetches GraphBAN source files)"
        )
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def setup_bkn_imports() -> dict:
    """Import GraphBAN modules from the bundled src/ directory.

    Returns a dict of module-level objects required by the training pipeline.
    Adds ban-kinase-network/src/ to sys.path as a side effect (idempotent).

    The trainer.py inside src/ has already been patched for 768-d drug
    embeddings (MoLFormer) — no exec() hackery needed.
    """
    _ensure_src_on_path(BKN_SRC)

    try:
        from configs import get_cfg_defaults
        from dataloader import DTIDataset, DTIDataset2, MultiDataLoader
        from domain_adaptator import Discriminator
        from models import GraphBAN, binary_cross_entropy, cross_entropy_logits
        from trainer import Trainer
        from utils import graph_collate_func, graph_collate_func2, mkdir, set_seed

        return {
            "get_cfg_defaults": get_cfg_defaults,
            "DTIDataset": DTIDataset,
            "DTIDataset2": DTIDataset2,
            "MultiDataLoader": MultiDataLoader,
            "GraphBAN": GraphBAN,
            "binary_cross_entropy": binary_cross_entropy,
            "cross_entropy_logits": cross_entropy_logits,
            "Trainer": Trainer,
            "graph_collate_func": graph_collate_func,
            "graph_collate_func2": graph_collate_func2,
            "mkdir": mkdir,
            "set_seed": set_seed,
            "Discriminator": Discriminator,
        }
    except ImportError as e:
        print(f"ERROR: BKN src/ import failed: {e}")
        print(f"Expected modules in: {BKN_SRC}")
        sys.exit(1)
