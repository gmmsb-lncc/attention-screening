"""Teacher GAE embedding generation (delegates to bundled teacher_gae.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .constants import BKN_SRC, DEFAULT_TEACHER_EPOCHS


def generate_teacher_embeddings(
    train_csv: Path,
    seed: int,
    output_parquet: Path,
    epochs: int = DEFAULT_TEACHER_EPOCHS,
) -> None:
    """Generate teacher GAE embeddings for a single seed.

    Delegates to the bundled teacher_gae.py script via subprocess.
    The teacher uses only graph structure (not ESM/MoLFormer features),
    so it is model-agnostic and unchanged from the GraphBAN original.

    Args:
        train_csv:       Path to the training CSV.
        seed:            Random seed for reproducibility.
        output_parquet:  Output path for the teacher embedding parquet file.
        epochs:          Number of GAE training epochs.
    """
    script = BKN_SRC / "teacher_gae.py"
    if not script.exists():
        raise FileNotFoundError(
            f"teacher_gae.py not found at {script}\n"
            "Make sure ban-kinase-network/src/ was populated correctly."
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--train_path", str(train_csv),
        "--seed", str(seed),
        "--teacher_path", str(output_parquet),
        "--epoch", str(epochs),
    ]
    print(f"\n  Generating teacher embeddings (seed={seed}, epochs={epochs})...")
    result = subprocess.run(cmd, capture_output=False, check=False, cwd=str(BKN_SRC))
    if result.returncode != 0:
        raise RuntimeError(
            f"teacher_gae.py failed (exit {result.returncode})"
        )
    print(f"    Teacher embeddings saved: {output_parquet}")
