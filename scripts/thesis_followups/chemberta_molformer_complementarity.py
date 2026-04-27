#!/usr/bin/env python3
"""Measure complementarity between MoLFormer and ChemBERTa ligand encoders.

Uses Centered Kernel Alignment (Kornblith et al., ICML 2019) to compare
the two representation spaces independently of their dimensionalities
(MoLFormer 768-d vs ChemBERTa 384-d).

Linear CKA in [0, 1]:
  > 0.90  highly redundant       → distillation likely no-op, skip
  0.70-0.90  meaningful overlap  → distillation may give marginal gain
  0.50-0.70  substantial complementarity → distillation likely helps
  < 0.50  very different         → caution, may force wrong direction

Both caches must exist:
    results/protein_model_benchmark_<corpus>_v2/8M/build/molformer_matrix/<id>.npy
    data/embeddings/v8/chemberta_<corpus>/<id>.npy

Each .npy is per-token (L, D); script mean-pools to a global (D,) vector.

Usage:
    python3 scripts/thesis_followups/chemberta_molformer_complementarity.py \\
        [--corpus non_human] [--n-samples 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between (n, d_x) and (n, d_y). Returns scalar in [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    if xx == 0 or yy == 0:
        return 0.0
    return float(xy / (xx * yy))


def load_global_pool(cache_dir: Path, n_max: int) -> tuple[np.ndarray, list[str]]:
    """Mean-pool per-token (L, D) → global (D,). Return (N, D) stack + IDs."""
    files = sorted(cache_dir.glob("*.npy"))[:n_max]
    arrs, ids = [], []
    for f in files:
        x = np.load(f, mmap_mode="r")
        if x.ndim == 1:
            arrs.append(x.astype(np.float64))
        elif x.ndim == 2:
            arrs.append(x.astype(np.float64).mean(axis=0))
        else:
            continue
        ids.append(f.stem)
    if not arrs:
        return np.empty((0, 0)), []
    return np.stack(arrs), ids


def cosine_per_pair(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. X and Y must have same n; can have diff d.
    Project to min-dim by truncation for fair comparison (rough heuristic).
    """
    d = min(X.shape[1], Y.shape[1])
    Xn = X[:, :d] / (np.linalg.norm(X[:, :d], axis=1, keepdims=True) + 1e-12)
    Yn = Y[:, :d] / (np.linalg.norm(Y[:, :d], axis=1, keepdims=True) + 1e-12)
    return (Xn * Yn).sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="non_human", choices=["human", "non_human", "all"])
    ap.add_argument("--n-samples", type=int, default=500)
    args = ap.parse_args()

    mol_dir = REPO / f"results/protein_model_benchmark_{args.corpus}_v2/8M/build/molformer_matrix"
    cb_dir = REPO / f"data/embeddings/v8/chemberta_{args.corpus}"

    if not mol_dir.exists():
        print(f"[fatal] MoLFormer cache missing: {mol_dir}", file=sys.stderr)
        return 2
    if not cb_dir.exists():
        print(f"[fatal] ChemBERTa cache missing: {cb_dir}", file=sys.stderr)
        return 2

    print(f"Loading global vectors (max {args.n_samples * 2} files each)...")
    M, mol_ids = load_global_pool(mol_dir, n_max=args.n_samples * 2)
    C, cb_ids = load_global_pool(cb_dir, n_max=args.n_samples * 2)
    print(f"  MoLFormer: shape={M.shape}, n_files={len(mol_ids)}")
    print(f"  ChemBERTa: shape={C.shape}, n_files={len(cb_ids)}")

    if M.size == 0 or C.size == 0:
        print("[fatal] one of the caches is empty", file=sys.stderr)
        return 2

    mol_idx = {i: k for k, i in enumerate(mol_ids)}
    cb_idx = {i: k for k, i in enumerate(cb_ids)}
    common = sorted(set(mol_ids) & set(cb_ids))[: args.n_samples]
    print(f"  intersection: {len(common)} common ligand IDs")

    if len(common) < 50:
        print(f"[warn] only {len(common)} common IDs; CKA will be noisy", file=sys.stderr)

    M_a = np.stack([M[mol_idx[i]] for i in common])
    C_a = np.stack([C[cb_idx[i]] for i in common])

    print(f"\nMoLFormer global stats: mean_norm={np.linalg.norm(M_a, axis=1).mean():.3f}")
    print(f"ChemBERTa global stats: mean_norm={np.linalg.norm(C_a, axis=1).mean():.3f}")

    cka = linear_cka(M_a, C_a)
    print(f"\n{'='*60}")
    print(f"  Linear CKA(MoLFormer, ChemBERTa) = {cka:.4f}")
    print(f"  on n={len(common)} ligantes do corpus {args.corpus}")
    print(f"{'='*60}\n")

    cos_truncated = cosine_per_pair(M_a, C_a)
    print(f"Per-pair cosine (truncated to min-dim={min(M_a.shape[1], C_a.shape[1])}):")
    print(f"  mean={cos_truncated.mean():.4f}  median={np.median(cos_truncated):.4f}")
    print(f"  min ={cos_truncated.min():.4f}  max   ={cos_truncated.max():.4f}")
    print(f"  std ={cos_truncated.std():.4f}")
    print()

    print("Interpretation:")
    if cka > 0.90:
        print(f"  CKA={cka:.3f} > 0.90 → HIGHLY REDUNDANT")
        print("  Recommendation: SKIP distillation. MoLFormer and ChemBERTa")
        print("  encode essentially the same information.")
        verdict = "skip"
    elif cka > 0.70:
        print(f"  CKA={cka:.3f} ∈ (0.70, 0.90] → MEANINGFUL OVERLAP")
        print("  Recommendation: distillation likely gives marginal gain")
        print("  (ΔMCC ~ 0.005-0.010). Consider only if budget allows.")
        verdict = "marginal"
    elif cka > 0.50:
        print(f"  CKA={cka:.3f} ∈ (0.50, 0.70] → SUBSTANTIAL COMPLEMENTARITY")
        print("  Recommendation: distillation likely helps")
        print("  (ΔMCC ~ 0.010-0.025). Worth implementing.")
        verdict = "implement"
    else:
        print(f"  CKA={cka:.3f} ≤ 0.50 → VERY DIFFERENT")
        print("  Recommendation: caution. Spaces too different — forcing")
        print("  alignment may pull MoLFormer features off-manifold.")
        print("  Consider matching specific properties instead of raw CLS.")
        verdict = "caution"

    print(f"\n[verdict] {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
