"""GraphBAN scoring adapter for the committee orchestrator.

Reads pairs.tsv and emits scores_graphban.csv with the uniform schema.
Wraps existing infer_graphban_universal.py logic. Must run inside the
`graphban` conda env (DGL + ESM-1b dependencies).

GraphBAN requires pre-computed feature cache (ESM-1b protein embeddings +
ChemBERTa molecular fingerprints) under
GraphBAN/results/{corpus}/feature_cache/features_extracted.pkl. New
sequences/SMILES not present in cache are encoded on-the-fly (slow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
from device_utils import pick_device, empty_cache  # noqa: E402

GRAPHBAN_SRC = REPO_ROOT / "GraphBAN" / "src" / "case_study"
if not GRAPHBAN_SRC.exists():
    GRAPHBAN_SRC = REPO_ROOT / "GraphBAN" / "src"
if not GRAPHBAN_SRC.exists():
    sys.exit(f"GraphBAN upstream not at {GRAPHBAN_SRC}")
sys.path.insert(0, str(GRAPHBAN_SRC))

from dataloader import DTIDataset  # type: ignore
from models import GraphBAN  # type: ignore
import dgl

CANONICAL_SEEDS = (42, 123, 456, 789, 1024)


def ckpt_seed_dir(corpus: str, seed: int) -> Path:
    return REPO_ROOT / "GraphBAN" / "results" / corpus / f"seed_{seed}"


def calibration_sidecar(corpus: str, seed: int) -> Path:
    return (REPO_ROOT / "GraphBAN" / "results_universal" / corpus
            / f"seed_{seed}" / "graphban_calibration.json")


CKPT_DIR_BY_CORPUS = {c: ckpt_seed_dir(c, 42) for c in ("human", "non_human", "all")}
CALIBRATION_SIDECAR_BY_CORPUS = {
    c: calibration_sidecar(c, 42) for c in ("human", "non_human", "all")
}
# kinase_inductive.yaml is the training config for the kinase committee weights
# (DECODER.BINARY=2 matches the checkpoint's mlp_classifier.fc4 = (2,128); the
# upstream GraphBAN.yaml default has BINARY=1 and is absent from this repo).
CANONICAL_CONFIG = REPO_ROOT / "GraphBAN" / "configs" / "kinase_inductive.yaml"


def _to_f32(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _cast_graph_f32(g):
    for k in list(g.ndata.keys()):
        t = g.ndata[k]
        if t.is_floating_point() and t.dtype != torch.float32:
            g.ndata[k] = t.to(torch.float32)
    for k in list(g.edata.keys()):
        t = g.edata[k]
        if t.is_floating_point() and t.dtype != torch.float32:
            g.edata[k] = t.to(torch.float32)
    return g


def graph_collate_func(batch):
    drugs, fcfps, proteins, esms = zip(*batch)
    drugs = [_cast_graph_f32(d) for d in drugs]
    return (
        dgl.batch(drugs),
        _to_f32(np.stack([np.asarray(f, dtype=np.float32) for f in fcfps])),
        torch.as_tensor(np.stack(proteins)),
        _to_f32(np.stack([np.asarray(e, dtype=np.float32) for e in esms])),
    )


def best_checkpoint(seed_dir: Path) -> Path:
    candidates = sorted(seed_dir.glob("best_model_epoch_*.pth"))
    if not candidates:
        sys.exit(f"no best_model_epoch_*.pth under {seed_dir}")
    return candidates[0]


def load_calibration(corpus: str, seed: int = 42) -> dict:
    sidecar = calibration_sidecar(corpus, seed)
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    print(f"  warn: no calibration sidecar at {sidecar}; using thr=0.5",
          file=sys.stderr)
    return {"threshold": 0.5}


@torch.no_grad()
def predict(model, loader, device):
    ps = []
    for v_d, fcfp, v_p, esm in loader:
        v_d = v_d.to(device, non_blocking=True)
        fcfp = fcfp.to(device, non_blocking=True)
        v_p = v_p.to(device, non_blocking=True)
        esm = esm.to(device, non_blocking=True)
        # case_study GraphBAN.forward expects (v_d, fcfp, v_p, esm, device)
        out = model(v_d, fcfp, v_p, esm, device)
        f = out[-1] if isinstance(out, tuple) else out
        prob = torch.softmax(f.float(), dim=1)[:, 1]
        prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        ps.append(prob.cpu().numpy())
    return np.concatenate(ps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score pairs.tsv with GraphBAN.")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], default="all")
    ap.add_argument("--ckpt",  type=Path, default=None,
                    help="explicit ckpt path (overrides --corpus + --seeds)")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                    help="comma-separated seeds for the 5-seed ensemble")
    ap.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    print(f"  device: {device}", file=sys.stderr)

    if args.ckpt is not None:
        seed_specs = [(args.ckpt, load_calibration(args.corpus, 42))]
    else:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        seed_specs = [
            (best_checkpoint(ckpt_seed_dir(args.corpus, s)),
             load_calibration(args.corpus, s)) for s in seed_list
        ]
    print(f"  ensemble: {len(seed_specs)} seed(s) (corpus={args.corpus})",
          file=sys.stderr)

    cfg = yaml.safe_load(open(args.config))

    df = pd.read_csv(args.pairs, sep="\t")
    drugban_df = pd.DataFrame({
        "SMILES":  df["smiles"].astype(str),
        "Protein": df["sequence"].astype(str),
        "Y":       np.zeros(len(df), dtype=int),
    })
    # case_study/dataloader.py expects pre-computed `fcfp` (ChemBERTa-77M-MTR
    # CLS, 384-d) and `esm` (ESM-1b mean-pool, 1280-d) columns. Compute them
    # by importing GraphBAN/run_baseline.py's feature extractors.
    sys.path.insert(0, str(REPO_ROOT / "GraphBAN"))
    from run_baseline import extract_esm_features, extract_chemberta_features  # type: ignore
    drugban_df = extract_esm_features(drugban_df, device)
    drugban_df = extract_chemberta_features(drugban_df, device)
    dataset = DTIDataset(drugban_df.index.values, drugban_df)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=graph_collate_func)

    n = len(df)
    probs_per_seed = np.empty((len(seed_specs), n), dtype=np.float64)
    thresholds_per_seed = np.empty(len(seed_specs), dtype=np.float64)

    for k, (ckpt, calib) in enumerate(seed_specs):
        print(f"  [seed {k+1}/{len(seed_specs)}] ckpt={ckpt.parent.name} "
              f"thr={calib['threshold']:.3f}", file=sys.stderr)
        model = GraphBAN(**cfg).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        probs_per_seed[k] = predict(model, loader, device)
        thresholds_per_seed[k] = calib["threshold"]
        del model
        empty_cache(device)

    probs = probs_per_seed.mean(axis=0)
    threshold = float(thresholds_per_seed.mean())
    out = pd.DataFrame({
        "uniprot":   df["uniprot"].to_numpy(),
        "chembl_id": df["chembl_id"].to_numpy(),
        "prob":      probs,
        "pred":      (probs >= threshold).astype(int),
        "threshold": threshold,
        "n_seeds":   len(seed_specs),
        "prob_std":  probs_per_seed.std(axis=0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"  wrote {len(out)} rows → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
