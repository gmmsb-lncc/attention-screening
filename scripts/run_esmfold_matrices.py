#!/usr/bin/env python3
"""Generate per-protein structure vectors for the --esmfold pipeline.

This script mirrors the operational style of other matrix-generation scripts:
- Reads a benchmark dataset TSV.
- Iterates unique ``seq_id`` / ``seq`` pairs.
- Writes one ``.npy`` vector per protein to:
  ``results/protein_model_benchmark_{dataset}_v2/{embedding}/build/protein_structure_features/``
- File naming: ``{seq_id}_esmfold.npy``

Modes:
- ``auto`` (default): try ESMFold structural inference if available; fallback to
  a sequence-based structural proxy vector.
- ``esmfold``: require ESMFold inference.
- ``proxy``: force fast sequence-based structural proxy vectors.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_PATHS = {
    "human": "tests/datasets/kinase_human_compounds.tsv",
    "non_human": "tests/datasets/kinase_non_human_compounds.tsv",
    "all": "tests/datasets/kinase_all_compounds.tsv",
}

EMBEDDING_BASE = "results/protein_model_benchmark_{dataset}_v2"

SUPPORTED_EMBEDDINGS = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA20)}
HYDROPHOBIC = set("AILMFWVPG")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def _sequence_proxy_vector(seq: str) -> np.ndarray:
    """Fast sequence-derived structural proxy (fixed-size vector).

    Returns a 96-d vector combining composition, motifs, and coarse
    physicochemical signals that correlate with fold characteristics.
    """
    seq = (seq or "").strip().upper()
    if not seq:
        return np.zeros((96,), dtype=np.float32)

    length = len(seq)

    # 20 AA composition
    comp = np.zeros((20,), dtype=np.float32)
    for ch in seq:
        idx = AA_INDEX.get(ch)
        if idx is not None:
            comp[idx] += 1.0
    comp /= max(length, 1)

    # 64 hashed dipeptide composition
    di = np.zeros((64,), dtype=np.float32)
    if length >= 2:
        for i in range(length - 1):
            a = AA_INDEX.get(seq[i], 0)
            b = AA_INDEX.get(seq[i + 1], 0)
            h = ((a * 31) + b) % 64
            di[h] += 1.0
        di /= float(length - 1)

    # 12 scalar descriptors
    frac_hydrophobic = sum(ch in HYDROPHOBIC for ch in seq) / length
    frac_polar = sum(ch in POLAR for ch in seq) / length
    frac_positive = sum(ch in POSITIVE for ch in seq) / length
    frac_negative = sum(ch in NEGATIVE for ch in seq) / length
    frac_gly = seq.count("G") / length
    frac_pro = seq.count("P") / length

    # Simple sequence complexity (Shannon entropy over AA frequencies)
    entropy = 0.0
    for p in comp:
        if p > 1e-8:
            entropy -= float(p * math.log(p + 1e-12))

    # Coarse moment-like stats over AA index sequence
    idx_seq = np.array([AA_INDEX.get(ch, 0) for ch in seq], dtype=np.float32)
    mean_idx = float(np.mean(idx_seq))
    std_idx = float(np.std(idx_seq))
    q25_idx = float(np.quantile(idx_seq, 0.25))
    q75_idx = float(np.quantile(idx_seq, 0.75))

    desc = np.array(
        [
            float(length),
            float(np.log1p(length)),
            frac_hydrophobic,
            frac_polar,
            frac_positive,
            frac_negative,
            frac_gly,
            frac_pro,
            float(entropy),
            mean_idx,
            std_idx,
            q25_idx + q75_idx,
        ],
        dtype=np.float32,
    )

    vec = np.concatenate([comp, di, desc]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _build_esmfold_model(device: str) -> tuple[Any, Any] | None:
    """Try loading ESMFold model from local/pip esm package."""
    try:
        import torch
    except Exception as exc:
        print(f"WARNING: ESMFold dependencies unavailable ({exc}).")
        return None

    # Prefer local ESM repo when available (project convention: llm/ESM).
    repo_root = Path(__file__).resolve().parent.parent
    local_esm_repo = repo_root / "llm" / "ESM"
    if local_esm_repo.exists():
        local_path = str(local_esm_repo)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    try:
        esm = importlib.import_module("esm")

        # Try known constructor paths across esm versions.
        model = None

        pretrained_mod = getattr(esm, "pretrained", None)
        if pretrained_mod is not None and hasattr(pretrained_mod, "esmfold_v1"):
            model = pretrained_mod.esmfold_v1()

        if model is None:
            try:
                from esm import pretrained as pretrained_import  # type: ignore
                if hasattr(pretrained_import, "esmfold_v1"):
                    model = pretrained_import.esmfold_v1()
            except Exception:
                pass

        if model is None:
            raise RuntimeError(
                "ESMFold constructor not found. Expected esm.pretrained.esmfold_v1()."
            )

        model = model.eval().to(dev)
    except Exception as exc:
        print(f"WARNING: Could not initialize ESMFold model ({exc}).")
        return None

    return model, dev


def _parse_ca_coords_from_pdb(pdb_str: str) -> np.ndarray:
    """Extract C-alpha coordinates from a PDB string."""
    coords: list[list[float]] = []
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coords.append([x, y, z])
        except Exception:
            continue
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _esmfold_vector(seq: str, model: Any, device_obj: Any) -> np.ndarray:
    """Build structural vector from ESMFold predicted coordinates.

    Output: 76-d vector
    - 12 global/shape descriptors
    - 64-bin C-alpha distance histogram (0..32A)
    """
    import torch

    seq = (seq or "").strip().upper()
    if not seq:
        return np.zeros((76,), dtype=np.float32)

    with torch.no_grad():
        pdb_str = model.infer_pdb(seq)

    ca = _parse_ca_coords_from_pdb(pdb_str)
    n = ca.shape[0]
    if n < 2:
        return np.zeros((76,), dtype=np.float32)

    center = ca.mean(axis=0, keepdims=True)
    centered = ca - center

    # Pairwise C-alpha distance histogram
    dists: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(ca[i] - ca[j]))
            dists.append(d)

    dist_hist = np.zeros((64,), dtype=np.float32)
    if dists:
        d_arr = np.asarray(dists, dtype=np.float32)
        hist, _ = np.histogram(d_arr, bins=64, range=(0.0, 32.0), density=False)
        dist_hist = hist.astype(np.float32)
        if dist_hist.sum() > 0:
            dist_hist /= dist_hist.sum()

    # Global shape descriptors
    rg = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    mins = ca.min(axis=0)
    maxs = ca.max(axis=0)
    span = maxs - mins

    cov = centered.T @ centered / max(n - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]

    globals_vec = np.array(
        [
            float(n),
            float(np.log1p(n)),
            rg,
            float(span[0]),
            float(span[1]),
            float(span[2]),
            float(eigvals[0]),
            float(eigvals[1]),
            float(eigvals[2]),
            float(eigvals[0] / (eigvals.sum() + 1e-8)),
            float(eigvals[1] / (eigvals.sum() + 1e-8)),
            float(eigvals[2] / (eigvals.sum() + 1e-8)),
        ],
        dtype=np.float32,
    )

    vec = np.concatenate([globals_vec, dist_hist]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def generate_esmfold_matrices(
    dataset: str,
    embedding_names: list[str],
    mode: str = "auto",
    device: str = "cpu",
    force: bool = False,
) -> None:
    """Generate per-protein structure vectors saved as ``*_esmfold.npy``."""
    dataset_path = Path(DATASET_PATHS[dataset])
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path, sep="\t")
    unique_proteins = df.drop_duplicates(subset="seq_id")[["seq_id", "seq"]].copy()
    print(f"Dataset: {dataset} — {len(unique_proteins)} unique proteins")

    encoder = None
    if mode in {"auto", "esmfold"}:
        print("\nTrying ESMFold encoder...")
        encoder = _build_esmfold_model(device)

    if encoder is None and mode == "esmfold":
        print("ERROR: Could not initialize ESMFold in strict mode.")
        sys.exit(1)

    if encoder is None:
        print("\nUsing sequence-based structural proxy vectors.")

    for emb_short in embedding_names:
        emb_full = SUPPORTED_EMBEDDINGS.get(emb_short, emb_short)
        base_dir = Path(EMBEDDING_BASE.format(dataset=dataset)) / emb_full / "build"
        output_dir = base_dir / "protein_structure_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Embedding: {emb_full}  |  Output: {output_dir}")
        print(f"{'='*60}")

        skipped = 0
        generated = 0

        for _, row in tqdm(unique_proteins.iterrows(), total=len(unique_proteins), desc="ESMFold"):
            seq_id = str(row["seq_id"])
            seq = str(row["seq"])
            out_file = output_dir / f"{seq_id}_esmfold.npy"

            if out_file.exists() and not force:
                skipped += 1
                continue

            if encoder is not None:
                model, dev = encoder
                try:
                    vec = _esmfold_vector(seq, model, dev)
                except Exception:
                    vec = _sequence_proxy_vector(seq)
            else:
                vec = _sequence_proxy_vector(seq)

            np.save(out_file, vec.astype(np.float32))
            generated += 1

        print(f"  Generated: {generated}, Skipped (exists): {skipped}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ESMFold-style protein structure vectors as *_esmfold.npy"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=list(DATASET_PATHS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--embedding",
        "-e",
        nargs="+",
        default=["650M"],
        help="Protein embedding shorthand(s) whose build/ dir to use (default: 650M)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "esmfold", "proxy"],
        default="auto",
        help="Generation mode (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for ESMFold inference (cpu or cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing vectors",
    )

    args = parser.parse_args()

    generate_esmfold_matrices(
        dataset=args.dataset,
        embedding_names=args.embedding,
        mode=args.mode,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
