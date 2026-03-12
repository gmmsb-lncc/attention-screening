"""BKN package constants — embedding dimensions, seeds, and project paths."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------

DRUG_EMB_DIM: int = 768       # MoLFormer-XL hidden size
ESM2_EMB_DIM: int = 1280      # ESM-2 650M hidden size
ESM2_MAX_SEQ_LEN: int = 1022  # ESM-2 maximum input sequence length
MOLFORMER_MAX_LEN: int = 202  # MoLFormer tokeniser max_length

# HuggingFace / ESM model identifiers
ESM2_MODEL_NAME: str = "esm2_t33_650M_UR50D"
MOLFORMER_MODEL_NAME: str = "ibm/MoLFormer-XL-both-10pct"

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

CANONICAL_SEEDS: list[int] = [42, 123, 456, 789, 1024]
DEFAULT_TEACHER_EPOCHS: int = 10

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# ban-kinase-network/bkn/constants.py → .parent.parent = ban-kinase-network/
SCRIPT_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = SCRIPT_DIR.parent

# GraphBAN source files bundled inside this project (MIT licence)
BKN_SRC: Path = SCRIPT_DIR / "src"

# Legacy aliases kept for any code that still references these names
GRAPHBAN_DIR: Path = SCRIPT_DIR  # BKN is self-contained — no external GraphBAN needed
GRAPHBAN_INDUCTIVE: Path = BKN_SRC
