"""ConPLex scoring adapter for the committee orchestrator.

Reads pairs.tsv and emits scores_conplex.csv. Wraps existing
infer_conplex_universal.py logic (ProtBERT + Morgan FP cosine
similarity). Must run inside the `conplex` conda env.

The default SimpleCoembedding projects both modalities through ReLU, so its
cosine similarity is non-negative and already lies in [0, 1]. The adapter
preserves that native scale because the MCC-optimal thresholds and the
canonical committee artifacts use it directly. The CSV column remains named
``prob`` for interface compatibility, but it is not a calibrated probability.
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
from device_utils import pick_device, empty_cache  # noqa: E402
from conplex_scale import canonical_similarity  # noqa: E402

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


CANONICAL_SEEDS = (42, 123, 456, 789, 1024)
CANONICAL_CKPT_BY_CORPUS = {
    c: _conplex_ckpt(c, seed=42)
    for c in ("human", "non_human", "all")
}


def calibration_sidecar(corpus: str, seed: int) -> Path:
    return (CONPLEX_ROOT / "results_universal" / corpus
            / f"seed_{seed}" / "conplex_calibration.json")


CALIBRATION_SIDECAR_BY_CORPUS = {
    c: calibration_sidecar(c, 42) for c in ("human", "non_human", "all")
}


def load_calibration(ckpt_path: Path, corpus: str, seed: int = 42) -> dict:
    """Try sidecar for the given seed first, then ckpt-adjacent, then default."""
    for sidecar in (calibration_sidecar(corpus, seed),
                    ckpt_path.parent / "conplex_calibration.json",
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
    ap.add_argument("--ckpt",  type=Path, default=None,
                    help="explicit ckpt path (overrides --corpus + --seeds)")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                    help="comma-separated seeds for the 5-seed ensemble; "
                         "internally maps to ConPLex rep0..rep4 via the "
                         "canonical seeds={42,123,456,789,1024} ↔ reps={0,1,2,3,4} order")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = pick_device()
    print(f"  device: {device}", file=sys.stderr)

    if args.ckpt is not None:
        seed_specs = [(args.ckpt, load_calibration(args.ckpt, args.corpus, 42))]
    else:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        seed_specs = []
        for s in seed_list:
            ckpt = _conplex_ckpt(args.corpus, seed=s)
            if not ckpt.exists():
                sys.exit(f"checkpoint not found for seed {s}: {ckpt}")
            seed_specs.append((ckpt, load_calibration(ckpt, args.corpus, s)))
    print(f"  ensemble: {len(seed_specs)} seed(s) (corpus={args.corpus})",
          file=sys.stderr)

    df = pd.read_csv(args.pairs, sep="\t")
    smiles = df["smiles"].astype(str).tolist()
    seqs   = df["sequence"].astype(str).tolist()

    print("  loading ProtBert + Morgan featurizers...", file=sys.stderr)
    prot_feat = ProtBertFeaturizer()
    drug_feat = MorganFeaturizer()
    d_all = _featurize_unique(drug_feat, smiles, "ligands")
    p_all = _featurize_unique(prot_feat, seqs,    "proteins")

    n = len(df)
    scores_per_seed = np.empty((len(seed_specs), n), dtype=np.float64)
    thresholds_per_seed = np.empty(len(seed_specs), dtype=np.float64)

    for k, (ckpt, calib) in enumerate(seed_specs):
        print(f"  [seed {k+1}/{len(seed_specs)}] ckpt={ckpt.parent.name} "
              f"thr={calib['threshold']:.3f}", file=sys.stderr)
        model = SimpleCoembedding().to(device).eval()
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        sims = predict(model, d_all, p_all, device, batch_size=args.batch_size)
        scores_per_seed[k] = canonical_similarity(sims)
        thresholds_per_seed[k] = float(calib["threshold"])
        del model
        empty_cache(device)

    scores = scores_per_seed.mean(axis=0)
    threshold = float(thresholds_per_seed.mean())
    out = pd.DataFrame({
        "uniprot":   df["uniprot"].to_numpy(),
        "chembl_id": df["chembl_id"].to_numpy(),
        "prob":      scores,
        "pred":      (scores >= threshold).astype(int),
        "threshold": threshold,
        "n_seeds":   len(seed_specs),
        "prob_std":  scores_per_seed.std(axis=0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"  wrote {len(out)} rows → {args.out} "
          f"(ensemble of {len(seed_specs)} seeds, mean threshold={threshold:.3f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
