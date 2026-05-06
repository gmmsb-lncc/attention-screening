"""Properly refit Platt calibration sidecars by re-running the model on val.

Replaces refit_calibration_offline.py for cases where val predictions are
not saved to disk. The training pipeline saves test predictions in
raw_predictions.npz (variable named "y_prob" but actually post-Platt-applied
test probabilities). The Platt parameters used at training time were never
serialised, so reconstructing inference-time calibration requires re-running
the model on the val set to obtain raw logits.

For each (corpus, seed) cell:
  1. Load the model via dtkinase_score's build_model + load_checkpoint.
  2. Re-encode val proteins/ligands (cache reused when present).
  3. Score every val pair to obtain raw logits.
  4. Fit Platt(val_logits → val_y_true) via sklearn LogisticRegression.
  5. Sweep MCC-optimal threshold on post-Platt val probabilities.
  6. Write sidecar JSON next to the checkpoint.

This is heavier than refit_calibration_offline.py (requires GPU + model
load + val inference) but produces honest sidecars that the inference
pipeline can apply correctly.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "inference" / "models"))
from dtkinase_score import (  # type: ignore  # noqa: E402
    build_model, load_checkpoint, load_esm2_8m, load_molformer,
    encode_proteins, encode_ligands, score_pair, resolve_ckpt,
    CANONICAL_CONFIG,
)
sys.path.insert(0, str(REPO / "scripts" / "inference"))
from device_utils import pick_device, empty_cache  # type: ignore  # noqa: E402


CORPUS_VAL_TSV = {
    "human":     REPO / "scaffolds_splits" / "output" / "scenarios" / "Sc"
                 / "human_val.tsv.gz",
    "non_human": REPO / "scaffolds_splits" / "output" / "scenarios" / "Sc"
                 / "non_human_val.tsv.gz",
}


def load_val_pairs(corpus: str) -> pd.DataFrame:
    path = CORPUS_VAL_TSV.get(corpus)
    if path is None or not path.exists():
        raise FileNotFoundError(f"val TSV missing for {corpus}: {path}")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, sep="\t")
    keep = ["seq_id", "seq", "chembl_id", "canonical_smiles", "label"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"missing cols in {path}: {missing}")
    out = df[keep].rename(columns={
        "seq_id": "uniprot",
        "seq": "sequence",
        "canonical_smiles": "smiles",
        "label": "y",
    })
    out["uniprot"] = out["uniprot"].astype(str)
    return out


def mcc_optimal_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pmin, pmax = float(probs.min()), float(probs.max())
    coarse = np.linspace(pmin, pmax, 100)
    best_t, best_mcc = 0.5, -1.0
    for t in coarse:
        m = matthews_corrcoef(y, (probs >= t).astype(int))
        if m > best_mcc:
            best_mcc, best_t = m, t
    fine = np.linspace(max(pmin, best_t - 0.05), min(pmax, best_t + 0.05), 100)
    for t in fine:
        m = matthews_corrcoef(y, (probs >= t).astype(int))
        if m > best_mcc:
            best_mcc, best_t = m, t
    return float(best_t), float(best_mcc)


def score_val_set(model, val_df: pd.DataFrame, prot_mats: dict,
                  lig_mats: dict, device: torch.device) -> np.ndarray:
    n = len(val_df)
    logits = np.empty(n, dtype=np.float64)
    missing = 0
    for i, row in enumerate(val_df.itertuples(index=False)):
        try:
            pmat = prot_mats[row.uniprot]
            lmat = lig_mats[row.chembl_id]
        except KeyError:
            missing += 1
            logits[i] = 0.0
            continue
        logits[i] = score_pair(model, pmat, lmat, device)
        if i % 5000 == 0:
            print(f"    scored {i}/{n}", file=sys.stderr)
    if missing > 0:
        print(f"  WARN: {missing} pairs had missing embeddings (set logit=0)",
              file=sys.stderr)
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["human", "non_human"], required=True)
    ap.add_argument("--root", type=Path, required=True,
                    help="dir containing seed_*/level4_cnn_model.pt")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    ap.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}; corpus: {args.corpus}; root: {args.root}")

    val_df = load_val_pairs(args.corpus)
    print(f"val pairs: {len(val_df)}")

    # Encode unique proteins / ligands ONCE (shared across seeds).
    print("encoding val proteins (ESM-2)...")
    esm_model, esm_alpha = load_esm2_8m(device)
    prot_mats = encode_proteins(
        esm_model, esm_alpha,
        val_df[["uniprot", "sequence"]].drop_duplicates(subset=["uniprot"])
              .itertuples(index=False, name=None),
        device,
    )
    del esm_model, esm_alpha; empty_cache(device)

    print("encoding val ligands (MoLFormer)...")
    mol_model, mol_tok = load_molformer(device)
    lig_mats = encode_ligands(
        mol_model, mol_tok,
        val_df[["chembl_id", "smiles"]].drop_duplicates(subset=["chembl_id"])
              .itertuples(index=False, name=None),
        device, batch_size=64,
    )
    del mol_model, mol_tok; empty_cache(device)
    print(f"  cached {len(prot_mats)} proteins / {len(lig_mats)} ligands")

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    model = build_model(args.config, device)
    y = val_df["y"].to_numpy(dtype=int)

    for seed in seed_list:
        ckpt = resolve_ckpt(args.corpus, seed) if not args.root \
               else args.root / f"seed_{seed}" / "level4_cnn_model.pt"
        if not ckpt.exists():
            print(f"  SKIP seed_{seed}: ckpt missing at {ckpt}")
            continue
        print(f"\n--- seed_{seed} ---")
        load_checkpoint(model, ckpt, device)
        t0 = time.time()
        logits = score_val_set(model, val_df, prot_mats, lig_mats, device)
        print(f"  val inference: {time.time()-t0:.1f}s for {len(logits)} pairs")

        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(logits.reshape(-1, 1), y)
        a = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
        cal_probs = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        thr, mcc = mcc_optimal_threshold(cal_probs, y)

        sidecar = {
            "platt_a":   a,
            "platt_b":   b,
            "threshold": thr,
            "calibration_metric": "mcc",
            "val_score": mcc,
            "n_val":     int(len(y)),
            "model":     "dtkinase",
            "corpus":    args.corpus,
            "seed":      int(seed),
            "source":    "val_inference (proper Platt fit on raw logits)",
            "note":      "platt_a, platt_b apply to raw model logit at inference",
        }
        out_path = ckpt.parent / "level4_cnn_calibration.json"
        out_path.write_text(json.dumps(sidecar, indent=2))
        print(f"  wrote {out_path}: a={a:+.4f} b={b:+.4f} thr={thr:.4f} val_mcc={mcc:.4f}")


if __name__ == "__main__":
    main()
