#!/usr/bin/env python3
"""Extract ligand embedding vectors from MoLFormer per-token matrices via mean pooling.

Reads {chembl_id}_matrix.npy files from molformer_matrix/ and saves
{chembl_id}_embedding.npy vectors (shape [768]) into ligand_embeddings/.

Usage:
    python scripts/extract_ligand_vectors.py \
        --embedding-dir results/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build

    # Process all ESM models at once (matrices are identical, vectors shared via symlink):
    python scripts/extract_ligand_vectors.py --all-models --dataset non_human
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def extract_vectors(
    matrix_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Mean-pool MoLFormer matrices into ligand vectors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_files = sorted(matrix_dir.glob("*_matrix.npy"))
    if not matrix_files:
        print(f"  WARNING: no matrix files found in {matrix_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    processed = 0
    skipped = 0
    errors = 0

    for mf in matrix_files:
        chembl_id = mf.stem.replace("_matrix", "")
        out_path = output_dir / f"{chembl_id}_embedding.npy"

        if out_path.exists() and not force:
            skipped += 1
            continue

        try:
            mat = np.load(mf)  # [n_tokens, 768]
            if mat.ndim != 2:
                print(f"  WARNING: unexpected shape {mat.shape} for {mf.name}, skipping")
                errors += 1
                continue

            vec = mat.mean(axis=0).astype(np.float32)  # [768]
            np.save(out_path, vec)
            processed += 1
        except Exception as e:
            print(f"  ERROR processing {mf.name}: {e}")
            errors += 1

    return {"processed": processed, "skipped": skipped, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Extract ligand vectors from MoLFormer matrices via mean pooling"
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        help="Path to embedding build dir (e.g. results/.../esm2_.../build)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Process all ESM models for the given dataset",
    )
    parser.add_argument(
        "--dataset",
        choices=["human", "non_human"],
        default="non_human",
        help="Dataset to process when using --all-models",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing vectors")
    args = parser.parse_args()

    if args.all_models:
        base = Path(f"results/protein_model_benchmark_{args.dataset}_v2")
        model_dirs = sorted(base.glob("esm2_*/build"))
    elif args.embedding_dir:
        model_dirs = [Path(args.embedding_dir)]
    else:
        parser.error("Provide --embedding-dir or --all-models")
        return

    for build_dir in model_dirs:
        matrix_dir = build_dir / "molformer_matrix"
        output_dir = build_dir / "ligand_embeddings"

        if not matrix_dir.exists():
            print(f"Skipping {build_dir}: no molformer_matrix/ directory")
            continue

        model_name = build_dir.parent.name
        print(f"Processing {model_name} ...")
        print(f"  Source:  {matrix_dir}")
        print(f"  Output:  {output_dir}")

        stats = extract_vectors(matrix_dir, output_dir, force=args.force)
        print(
            f"  Done: {stats['processed']} extracted, "
            f"{stats['skipped']} skipped, {stats['errors']} errors"
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
