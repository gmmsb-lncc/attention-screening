"""Ligand vector extraction using attention pooling.

Reads per-token MoLFormer matrices and writes fixed-size vector
embeddings so that Level 2 can consume them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.config import EMBEDDING_BASE_PATH, BenchmarkConfig


class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence aggregation.

    Uses a learnable query vector to compute attention weights over
    sequence positions, then performs a weighted sum.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.attention = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool a variable-length sequence into a fixed-size vector.

        Parameters
        ----------
        x: ``[seq_len, dim]``
        mask: ``[seq_len]`` bool — ``True`` for valid positions.

        Returns
        -------
        ``[dim]`` pooled vector.
        """
        scores = self.attention(x).squeeze(-1)  # [seq_len]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=0)  # [seq_len]
        return (x * weights.unsqueeze(-1)).sum(dim=0)  # [dim]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_ligand_vectors(
    matrix_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Extract fixed-size ligand vectors from MoLFormer matrices.

    Reads ``{chembl_id}_matrix.npy`` or ``{chembl_id}_molformer_matrix.npy``
    (shape ``[n_tokens, 768]``) and writes ``{chembl_id}_embedding.npy``
    (shape ``[768]``).

    Returns a dict with keys ``processed``, ``skipped``, ``errors``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect matrix files, deduplicating across naming patterns.
    all_files: dict[str, Path] = {}
    for mf in sorted(matrix_dir.glob("*_matrix.npy")):
        chembl_id = mf.stem.replace("_matrix", "")
        all_files[chembl_id] = mf
    for mf in sorted(matrix_dir.glob("*_molformer_matrix.npy")):
        chembl_id = mf.name.replace("_molformer_matrix.npy", "")
        if chembl_id not in all_files:
            all_files[chembl_id] = mf

    matrix_files = sorted(all_files.values(), key=lambda p: p.name)

    if not matrix_files:
        print(f"  WARNING: no matrix files found in {matrix_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_mat = np.load(matrix_files[0])
    pooling_model = AttentionPooling(sample_mat.shape[1]).to(device)
    pooling_model.eval()

    processed = skipped = errors = 0
    with torch.no_grad():
        for mf in matrix_files:
            chembl_id = _chembl_id_from_filename(mf)
            out_path = output_dir / f"{chembl_id}_embedding.npy"

            if out_path.exists() and not force:
                skipped += 1
                continue

            try:
                mat = np.load(mf)
                if mat.ndim != 2:
                    print(f"  WARNING: unexpected shape {mat.shape} for {mf.name}, skipping")
                    errors += 1
                    continue

                mat_tensor = torch.from_numpy(mat).float().to(device)
                mask = torch.ones(mat_tensor.shape[0], dtype=torch.bool, device=device)
                pooled = pooling_model(mat_tensor, mask)

                np.save(out_path, pooled.cpu().numpy().astype(np.float32))
                processed += 1
            except Exception as exc:
                print(f"  ERROR processing {mf.name}: {exc}")
                errors += 1

    return {"processed": processed, "skipped": skipped, "errors": errors}


def _chembl_id_from_filename(path: Path) -> str:
    """Extract ChEMBL ID from either naming convention."""
    if path.name.endswith("_molformer_matrix.npy"):
        return path.name.replace("_molformer_matrix.npy", "")
    return path.stem.replace("_matrix", "")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_ligand_vectors(config: BenchmarkConfig) -> bool:
    """Verify or extract ligand vectors using attention pooling.

    Returns ``True`` when all requested datasets have vectors available.
    """
    all_ok = True

    for ds in config.datasets_to_process():
        build_dir = Path(
            EMBEDDING_BASE_PATH.format(dataset_type=ds),
            config.embedding_name,
            "build",
        )
        molformer_dir = build_dir / "molformer_matrix"
        vector_dir = build_dir / "ligand_embeddings"

        if (
            vector_dir.exists()
            and any(vector_dir.glob("*_embedding.npy"))
            and not config.force
        ):
            n_files = len(list(vector_dir.glob("*_embedding.npy")))
            print(f"  [OK] Ligand vectors ({ds}): {vector_dir} ({n_files} files)")
            continue

        if not molformer_dir.exists():
            print(f"  WARNING: MoLFormer matrix dir not found ({ds}): {molformer_dir}")
            print("           Level 2 embedding features will not include ligand vectors.")
            all_ok = False
            continue

        print(f"  Extracting ligand vectors ({ds}) from {molformer_dir}...")
        stats = _extract_ligand_vectors(molformer_dir, vector_dir, force=config.force)
        print(
            f"  Done ({ds}): {stats['processed']} extracted, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )
        if stats["processed"] + stats["skipped"] == 0:
            all_ok = False

    return all_ok
