#!/usr/bin/env python3
"""Build ConPLex per-rep calibration sidecars for a 5-rep ensemble.

conplex_score.py scores on the native cosine-similarity scale and uses the
sidecar ONLY for the MCC-optimal threshold (no Platt is applied to ConPLex
similarities). So calibrating reps 1..4 means: run each rep on the training
corpus validation split, then pick the MCC-optimal threshold on the native
similarity. Platt a/b are written as identity (1.0, 0.0) for schema parity.

The canonical rep0 sidecar is produced by the training pipeline; this fills the
known gap for reps 1..4 (see CLAUDE.md committee-artifacts note).

Usage:
    python scripts/inference/build_conplex_calibration.py \
        --corpus non_human --reps 1,2,3,4 \
        --val-tsv scaffolds_splits/output/non_human_val.tsv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference" / "models"))
sys.path.insert(0, str(REPO_ROOT / "ConPLex"))

from device_utils import pick_device, empty_cache  # noqa: E402
from conplex_scale import canonical_similarity  # noqa: E402
from src.featurizers.molecule import MorganFeaturizer  # type: ignore  # noqa: E402
from src.featurizers.protein import ProtBertFeaturizer  # type: ignore  # noqa: E402
from src.architectures import SimpleCoembedding  # type: ignore  # noqa: E402
# reuse the exact scoring + featurization helpers from the score script
from conplex_score import (  # type: ignore  # noqa: E402
    predict, _featurize_unique, _conplex_ckpt, calibration_sidecar,
)

CONPLEX_ROOT = REPO_ROOT / "ConPLex"


def mcc_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Two-pass sweep (100 coarse + 100 fine +/-0.05) for max-MCC threshold."""
    lo, hi = float(scores.min()), float(scores.max())
    best_t, best_mcc = 0.5, -2.0
    for t in np.linspace(lo, hi, 100):
        mcc = matthews_corrcoef(y, (scores >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    for t in np.linspace(best_t - 0.05, best_t + 0.05, 100):
        mcc = matthews_corrcoef(y, (scores >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t, best_mcc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], required=True)
    ap.add_argument("--seeds", default="123,456,789,1024",
                    help="comma-separated canonical seeds to calibrate "
                         "(42->rep0 already shipped; 123/456/789/1024 -> rep1..4)")
    ap.add_argument("--val-tsv", type=Path, required=True,
                    help="scaffold val TSV with canonical_smiles/seq/label columns")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = pick_device()
    print(f"  device: {device}", file=sys.stderr)

    df = pd.read_csv(args.val_tsv, sep="\t")
    smiles = df["canonical_smiles"].astype(str).tolist()
    seqs = df["seq"].astype(str).tolist()
    y = df["label"].to_numpy(dtype=int)
    print(f"  val pairs: {len(df)}  (positives={int(y.sum())}/{len(y)})", file=sys.stderr)

    print("  loading ProtBert + Morgan featurizers...", file=sys.stderr)
    prot_feat = ProtBertFeaturizer()
    drug_feat = MorganFeaturizer()
    d_all = _featurize_unique(drug_feat, smiles, "ligands")
    p_all = _featurize_unique(prot_feat, seqs, "proteins")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        ckpt = _conplex_ckpt(args.corpus, seed=seed)
        if not ckpt.exists():
            print(f"  [seed {seed}] SKIP — ckpt missing: {ckpt}", file=sys.stderr)
            continue
        model = SimpleCoembedding().to(device).eval()
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        sims = canonical_similarity(predict(model, d_all, p_all, device,
                                            batch_size=args.batch_size))
        del model
        empty_cache(device)

        thr, mcc = mcc_optimal_threshold(sims, y)
        out_path = calibration_sidecar(args.corpus, seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "platt_a": 1.0, "platt_b": 0.0,  # unused by conplex scoring
            "threshold": float(thr),
            "corpus": args.corpus,
            "n_val": int(len(y)),
            "source_run": str(ckpt),
            "val_mcc": float(mcc),
            "note": "threshold-only calibration (ConPLex scores on native similarity)",
        }
        with open(out_path, "w") as fh:
            json.dump(sidecar, fh, indent=2)
        print(f"  [seed {seed}] thr={thr:.4f} val_mcc={mcc:.4f} -> {out_path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
