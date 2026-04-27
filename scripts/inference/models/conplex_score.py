"""ConPLex scoring adapter for the committee orchestrator.

Reads pairs.tsv and emits scores_conplex.csv. Wraps existing
infer_conplex_universal.py logic (ProtBERT + Morgan FP cosine
similarity). Must run inside the `conplex` conda env.

ConPLex outputs cosine similarity in [-1, 1]; the adapter rescales to
probability in [0, 1] via (sim + 1) / 2 for committee soft mean
compatibility. The native MCC-optimal threshold is loaded from the
sidecar JSON adjacent to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

CONPLEX_ROOT = REPO_ROOT / "ConPLex"
sys.path.insert(0, str(CONPLEX_ROOT))

from src.featurizers.molecule import MorganFeaturizer  # type: ignore
from src.featurizers.protein  import ProtBertFeaturizer  # type: ignore
from src.architectures        import SimpleCoembedding  # type: ignore


def _conplex_ckpt(corpus: str, seed: int = 42) -> Path:
    """Resolve ConPLex trained checkpoint by (corpus, seed).

    The original ConPLex training pipeline indexes replicates as rep{0..4}
    rather than canonical seeds. Canonical mapping mirrors the order used
    by run_conplex_universal_eval.sh: seeds={42,123,456,789,1024}
    correspond to reps={0,1,2,3,4} respectively.

    Path layout:
        ConPLex/best_models/trained_{corpus}_rep{rep}/trained_{corpus}_rep{rep}_best_model.pt

    Fallback: if the consolidated `_best_model.pt` is absent (training did
    not consolidate yet for this rep), return the highest-epoch checkpoint
    in the same directory.
    """
    seed_to_rep = {42: 0, 123: 1, 456: 2, 789: 3, 1024: 4}
    rep = seed_to_rep.get(seed, 0)
    name = f"trained_{corpus}_rep{rep}"
    seed_dir = CONPLEX_ROOT / "best_models" / name
    final = seed_dir / f"{name}_best_model.pt"
    if final.exists():
        return final
    epoch_ckpts = sorted(seed_dir.glob(f"{name}_best_model_epoch*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return final  # let downstream raise FileNotFoundError with this path


CANONICAL_CKPT_BY_CORPUS = {
    "human":     _conplex_ckpt("human",     seed=42),
    "non_human": _conplex_ckpt("non_human", seed=42),
    "all":       _conplex_ckpt("all",       seed=42),
}
CALIBRATION_SIDECAR_BY_CORPUS = {
    "human":     CONPLEX_ROOT / "results_universal" / "human"     / "seed_42" / "conplex_calibration.json",
    "non_human": CONPLEX_ROOT / "results_universal" / "non_human" / "seed_42" / "conplex_calibration.json",
    "all":       CONPLEX_ROOT / "results_universal" / "all"       / "seed_42" / "conplex_calibration.json",
}


def load_calibration(ckpt_path: Path, corpus: str) -> dict:
    """Try sidecar next to ckpt first, then fall back to results_universal layout."""
    for sidecar in (ckpt_path.parent / "conplex_calibration.json",
                    CALIBRATION_SIDECAR_BY_CORPUS.get(corpus)):
        if sidecar is not None and sidecar.exists():
            return json.loads(sidecar.read_text())
    print(f"  warn: no calibration sidecar found; using thr=0.5", file=sys.stderr)
    return {"threshold": 0.5}


def _featurize_unique(featurizer, items, label):
    """Featurize each unique item once, return per-item embedding stack."""
    cache = {}
    embs = []
    for it in items:
        if it not in cache:
            cache[it] = torch.as_tensor(featurizer(it), dtype=torch.float32)
        embs.append(cache[it])
    print(f"  {label}: {len(items)} pairs, {len(cache)} unique featurized",
          file=sys.stderr)
    return torch.stack(embs)


@torch.no_grad()
def predict(model, d_all, p_all, device, batch_size=1024):
    use_amp = device.type == "cuda"
    sims = []
    n = d_all.shape[0]
    for i in range(0, n, batch_size):
        d_b = d_all[i:i+batch_size].to(device, non_blocking=True)
        p_b = p_all[i:i+batch_size].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            sim = model(d_b, p_b)
        sims.append(sim.float().cpu().numpy())
    return np.concatenate(sims)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score pairs.tsv with ConPLex.")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], default="all")
    ap.add_argument("--ckpt",  type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or CANONICAL_CKPT_BY_CORPUS[args.corpus]
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    calib = load_calibration(ckpt, args.corpus)
    print(f"  ckpt: {ckpt}", file=sys.stderr)
    print(f"  thr: {calib['threshold']:.3f}", file=sys.stderr)

    df = pd.read_csv(args.pairs, sep="\t")
    smiles = df["smiles"].astype(str).tolist()
    seqs   = df["sequence"].astype(str).tolist()

    print("  loading ProtBert + Morgan featurizers...", file=sys.stderr)
    prot_feat = ProtBertFeaturizer()
    drug_feat = MorganFeaturizer()
    d_all = _featurize_unique(drug_feat, smiles, "ligands")
    p_all = _featurize_unique(prot_feat, seqs,    "proteins")

    print("  loading SimpleCoembedding model...", file=sys.stderr)
    model = SimpleCoembedding().to(device).eval()
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    sims = predict(model, d_all, p_all, device, batch_size=args.batch_size)
    # rescale cosine [-1, 1] → probability [0, 1]
    probs = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)

    out = pd.DataFrame({
        "uniprot":   df["uniprot"].to_numpy(),
        "chembl_id": df["chembl_id"].to_numpy(),
        "prob":      probs,
        "pred":      (probs >= calib["threshold"]).astype(int),
        "threshold": calib["threshold"],
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"  wrote {len(out)} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
