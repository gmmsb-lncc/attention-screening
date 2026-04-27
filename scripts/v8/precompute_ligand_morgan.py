"""Pre-compute Morgan fingerprint topological encoding per ligand.

Morgan fingerprints (Rogers & Hahn 2010) capture circular substructure
presence as fixed-length bit vectors. They serve as a low-cost proxy
for the topological inductive bias provided by GCN-based encoders
(DrugBAN, GraphBAN), with the advantage of being deterministic, fast
to compute, and requiring no training.

For each unique chembl_id appearing in scaffolds_splits/output/*.tsv,
this script:
  1. Reads the SMILES string
  2. Generates Morgan FP (radius=2, n_bits=1024) via RDKit
  3. Saves as .npy at the configured cache directory

The DT-Kinase model can then load this FP and concatenate it to the
HierPool output as an additional feature channel (analogous to the
cosine_sim auxiliary feature).

Usage:
    python3 scripts/v8/precompute_ligand_morgan.py \
        --splits-dir scaffolds_splits/output \
        --out-dir results/morgan_fp \
        --n-bits 1024 --radius 2

Output:
    results/morgan_fp/<chembl_id>_morgan.npy   (1024-bit float32 array)
    results/morgan_fp/_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def collect_unique_smiles(splits_dir: Path) -> dict[str, str]:
    """Return {chembl_id: smiles} aggregated across all split TSVs."""
    out: dict[str, str] = {}
    patterns = ["*_train.tsv", "*_val.tsv", "*_test.tsv"]
    files = []
    for pat in patterns:
        files.extend(splits_dir.rglob(pat))
    print(f"[info] scanning {len(files)} TSV files under {splits_dir}")
    for tsv in files:
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"[warn] {tsv}: {e}")
            continue
        smi_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        if smi_col not in df.columns or "chembl_id" not in df.columns:
            continue
        df = df[["chembl_id", smi_col]].drop_duplicates()
        for cid, sm in zip(df["chembl_id"], df[smi_col]):
            if cid not in out and isinstance(sm, str) and len(sm) > 0:
                out[cid] = sm
    return out


def morgan_fp(smiles: str, n_bits: int, radius: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def main():
    if not HAS_RDKIT:
        raise SystemExit("[fatal] rdkit not installed. pip install rdkit")
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument("--radius", type=int, default=2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    smiles_map = collect_unique_smiles(args.splits_dir)
    print(f"[info] unique chembl_ids: {len(smiles_map)}")

    n_done, n_skip, n_fail = 0, 0, 0
    for cid, sm in tqdm(smiles_map.items(), desc="morgan FP"):
        out_path = args.out_dir / f"{cid}_morgan.npy"
        if out_path.exists():
            n_skip += 1
            continue
        fp = morgan_fp(sm, n_bits=args.n_bits, radius=args.radius)
        if fp is None:
            n_fail += 1
            continue
        np.save(out_path, fp)
        n_done += 1

    summary = {
        "splits_dir": str(args.splits_dir),
        "out_dir": str(args.out_dir),
        "n_bits": args.n_bits,
        "radius": args.radius,
        "n_total": len(smiles_map),
        "n_done": n_done,
        "n_skipped": n_skip,
        "n_failed": n_fail,
    }
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {n_done} written, {n_skip} skipped, {n_fail} failed → {args.out_dir}")


if __name__ == "__main__":
    main()
