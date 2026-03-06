#!/usr/bin/env python3
"""Diagnostic script to identify and fix ligand vector extraction issues.

This script checks:
1. If molformer_matrix directories exist and contain valid .npy files
2. If ligand_embeddings directories exist and contain valid vectors
3. File permissions
4. Shape and dtype of matrix files
5. Mismatch between matrix and embedding counts

Usage:
    python scripts/diagnose_ligand_vectors.py --dataset non_human
    python scripts/diagnose_ligand_vectors.py --dataset all --fix
"""

import argparse
import os
import stat
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Configuration
EMBEDDING_BASE_PATH = "./results/protein_model_benchmark_{dataset_type}_v2"
SUPPORTED_EMBEDDINGS = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
    "35M": "esm2_t12_35M_UR50D",
    "3B": "esm2_t36_3B_UR50D",
}


def check_directory_permissions(path: Path) -> Dict[str, any]:
    """Check if directory exists and has proper permissions."""
    result = {
        "exists": path.exists(),
        "is_dir": False,
        "readable": False,
        "writable": False,
        "file_count": 0,
        "error": None,
    }

    if not result["exists"]:
        result["error"] = f"Directory does not exist: {path}"
        return result

    result["is_dir"] = path.is_dir()

    try:
        result["readable"] = os.access(path, os.R_OK)
        result["writable"] = os.access(path, os.W_OK)
        result["file_count"] = len(list(path.glob("*.npy")))
    except Exception as e:
        result["error"] = str(e)

    return result


def validate_matrix_file(filepath: Path) -> Dict[str, any]:
    """Validate a matrix .npy file."""
    result = {
        "valid": False,
        "shape": None,
        "ndim": None,
        "dtype": None,
        "error": None,
    }

    try:
        mat = np.load(filepath)
        result["shape"] = mat.shape
        result["ndim"] = mat.ndim
        result["dtype"] = str(mat.dtype)

        # Check if it's a valid 2D matrix with 768 features (MoLFormer hidden size)
        if mat.ndim == 2 and mat.shape[1] == 768:
            result["valid"] = True
        elif mat.ndim == 2:
            result["error"] = f"Unexpected feature dimension: {mat.shape[1]} (expected 768)"
        else:
            result["error"] = f"Unexpected dimensions: {mat.ndim}D (expected 2D)"

    except Exception as e:
        result["error"] = f"Failed to load: {str(e)}"

    return result


def validate_vector_file(filepath: Path) -> Dict[str, any]:
    """Validate a vector .npy file."""
    result = {
        "valid": False,
        "shape": None,
        "ndim": None,
        "dtype": None,
        "error": None,
    }

    try:
        vec = np.load(filepath)
        result["shape"] = vec.shape
        result["ndim"] = vec.ndim
        result["dtype"] = str(vec.dtype)

        # Check if it's a valid 1D vector with 768 features
        if vec.ndim == 1 and vec.shape[0] == 768:
            result["valid"] = True
        elif vec.ndim == 1:
            result["error"] = f"Unexpected dimension: {vec.shape[0]} (expected 768)"
        else:
            result["error"] = f"Unexpected dimensions: {vec.ndim}D (expected 1D)"

    except Exception as e:
        result["error"] = f"Failed to load: {str(e)}"

    return result


def extract_vectors_from_matrices(
    matrix_dir: Path,
    output_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """Extract ligand vectors from matrices via mean pooling.

    Returns:
        Tuple of (processed, skipped, errors)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for files with both patterns
    matrix_files = sorted(matrix_dir.glob("*_matrix.npy"))
    molformer_files = sorted(matrix_dir.glob("*_molformer_matrix.npy"))
    
    # Combine and deduplicate (prefer _matrix.npy over _molformer_matrix.npy)
    all_files = {}
    for mf in matrix_files:
        chembl_id = mf.stem.replace("_matrix", "")
        all_files[chembl_id] = mf
    for mf in molformer_files:
        chembl_id = mf.name.replace("_molformer_matrix.npy", "")
        if chembl_id not in all_files:
            all_files[chembl_id] = mf
    
    matrix_files = sorted(all_files.values(), key=lambda x: x.name)

    if not matrix_files:
        print(f"  WARNING: No matrix files found in {matrix_dir}")
        return 0, 0, 0

    processed = 0
    skipped = 0
    errors = 0

    for mf in matrix_files:
        # Extract chembl_id from filename (handle both patterns)
        if mf.name.endswith("_molformer_matrix.npy"):
            chembl_id = mf.name.replace("_molformer_matrix.npy", "")
        else:
            chembl_id = mf.stem.replace("_matrix", "")

        out_path = output_dir / f"{chembl_id}_embedding.npy"

        if out_path.exists() and not force:
            skipped += 1
            continue

        if dry_run:
            processed += 1
            continue

        try:
            mat = np.load(mf)

            if mat.ndim != 2:
                if verbose:
                    print(f"  WARNING: {mf.name} has {mat.ndim}D shape {mat.shape}, skipping")
                errors += 1
                continue

            # Mean pooling over sequence length
            vec = mat.mean(axis=0).astype(np.float32)
            np.save(out_path, vec)
            processed += 1

        except Exception as e:
            if verbose:
                print(f"  ERROR processing {mf.name}: {e}")
            errors += 1

    return processed, skipped, errors


def diagnose_dataset(dataset_type: str, fix: bool = False, dry_run: bool = False):
    """Diagnose ligand vector extraction for a dataset."""

    print("\n" + "=" * 70)
    print(f"DIAGNOSING: {dataset_type.upper()}")
    print("=" * 70)

    issues_found = []
    total_matrices = 0
    total_vectors = 0

    # Check each protein model
    base_path = Path(EMBEDDING_BASE_PATH.format(dataset_type=dataset_type))

    if not base_path.exists():
        print(f"  ERROR: Base path does not exist: {base_path}")
        return issues_found

    for emb_short, emb_name in SUPPORTED_EMBEDDINGS.items():
        model_dir = base_path / emb_name
        if not model_dir.exists():
            continue

        build_dir = model_dir / "build"
        if not build_dir.exists():
            continue

        print(f"\n--- {emb_name} ({emb_short}) ---")

        # Check directories
        molformer_dir = build_dir / "molformer_matrix"
        ligand_dir = build_dir / "ligand_matrices"
        vector_dir = build_dir / "ligand_embeddings"

        # Check molformer_matrix
        molformer_info = check_directory_permissions(molformer_dir)
        print(f"  molformer_matrix/:")
        if molformer_info["exists"]:
            print(f"    ✓ Exists, {molformer_info['file_count']} files")
            if not molformer_info["readable"]:
                print(f"    ✗ NOT READABLE!")
                issues_found.append(f"{emb_name}: molformer_matrix not readable")
        else:
            print(f"    ✗ Does not exist")

        # Check ligand_matrices (legacy location)
        ligand_info = check_directory_permissions(ligand_dir)
        if ligand_info["exists"]:
            print(f"  ligand_matrices/: (legacy)")
            print(f"    ✓ Exists, {ligand_info['file_count']} files")

        # Check ligand_embeddings
        vector_info = check_directory_permissions(vector_dir)
        print(f"  ligand_embeddings/:")
        if vector_info["exists"]:
            print(f"    ✓ Exists, {vector_info['file_count']} files")
        else:
            print(f"    ✗ Does not exist")

        # Validate sample files
        if molformer_info["exists"] and molformer_info["file_count"] > 0:
            sample_files = list(molformer_dir.glob("*_matrix.npy"))[:3]
            print(f"  Validating sample matrices:")
            for sf in sample_files:
                validation = validate_matrix_file(sf)
                status = "✓" if validation["valid"] else "✗"
                print(f"    {status} {sf.name}: shape={validation['shape']}, dtype={validation['dtype']}")
                if not validation["valid"]:
                    print(f"      Error: {validation['error']}")
                    issues_found.append(f"{emb_name}: Invalid matrix {sf.name}")

        # Count totals
        total_matrices += molformer_info["file_count"]
        total_vectors += vector_info["file_count"]

        # Check if extraction is needed
        if molformer_info["file_count"] > 0:
            if vector_info["file_count"] == 0:
                print(f"\n  ⚠ ACTION NEEDED: {molformer_info['file_count']} matrices but 0 vectors!")
                if fix:
                    print(f"  → Extracting vectors...")
                    proc, skip, err = extract_vectors_from_matrices(
                        molformer_dir, vector_dir, force=False, dry_run=dry_run
                    )
                    if dry_run:
                        print(f"  [DRY RUN] Would extract {proc} vectors")
                    else:
                        print(f"  → Extracted {proc} vectors, skipped {skip}, errors {err}")
            elif vector_info["file_count"] < molformer_info["file_count"]:
                missing = molformer_info["file_count"] - vector_info["file_count"]
                print(f"\n  ⚠ MISMATCH: {missing} matrices without corresponding vectors!")
                if fix:
                    print(f"  → Extracting missing vectors...")
                    proc, skip, err = extract_vectors_from_matrices(
                        molformer_dir, vector_dir, force=False, dry_run=dry_run
                    )
                    if dry_run:
                        print(f"  [DRY RUN] Would extract {proc} vectors")
                    else:
                        print(f"  → Extracted {proc} vectors, skipped {skip}, errors {err}")
            else:
                print(f"\n  ✓ Matrices and vectors are in sync")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY for {dataset_type.upper()}:")
    print(f"  Total matrix files: {total_matrices}")
    print(f"  Total vector files: {total_vectors}")
    print(f"  Issues found: {len(issues_found)}")

    if issues_found:
        print("\n  Issues:")
        for issue in issues_found:
            print(f"    - {issue}")
    else:
        print("  ✓ No issues found!")

    return issues_found


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and fix ligand vector extraction issues"
    )
    parser.add_argument(
        "--dataset",
        choices=["human", "non_human", "all"],
        default="non_human",
        help="Dataset to diagnose (default: non_human)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix issues by extracting missing vectors"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LIGAND VECTOR EXTRACTION DIAGNOSTIC")
    print("=" * 70)

    all_issues = []

    if args.dataset == "all":
        for ds in ["human", "non_human"]:
            issues = diagnose_dataset(ds, fix=args.fix, dry_run=args.dry_run)
            all_issues.extend(issues)
    else:
        all_issues = diagnose_dataset(args.dataset, fix=args.fix, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if all_issues:
        print(f"\n⚠ Total issues found: {len(all_issues)}")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1
    else:
        print("\n✓ All checks passed! Ligand vectors are properly extracted.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
