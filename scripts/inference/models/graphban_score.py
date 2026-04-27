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

GRAPHBAN_SRC = REPO_ROOT / "GraphBAN" / "src" / "case_study"
if not GRAPHBAN_SRC.exists():
    GRAPHBAN_SRC = REPO_ROOT / "GraphBAN" / "src"
if not GRAPHBAN_SRC.exists():
    sys.exit(f"GraphBAN upstream not at {GRAPHBAN_SRC}")
sys.path.insert(0, str(GRAPHBAN_SRC))

from dataloader import DTIDataset  # type: ignore
from models import GraphBAN  # type: ignore
import dgl

CKPT_DIR_BY_CORPUS = {
    # Trained ckpts under GraphBAN/results/{corpus}/seed_X/best_model_epoch_<N>.pth.
    "human":     REPO_ROOT / "GraphBAN" / "results" / "human"     / "seed_42",
    "non_human": REPO_ROOT / "GraphBAN" / "results" / "non_human" / "seed_42",
    "all":       REPO_ROOT / "GraphBAN" / "results" / "all"       / "seed_42",
}
CALIBRATION_SIDECAR_BY_CORPUS = {
    "human":     REPO_ROOT / "GraphBAN" / "results_universal" / "human"     / "seed_42" / "graphban_calibration.json",
    "non_human": REPO_ROOT / "GraphBAN" / "results_universal" / "non_human" / "seed_42" / "graphban_calibration.json",
    "all":       REPO_ROOT / "GraphBAN" / "results_universal" / "all"       / "seed_42" / "graphban_calibration.json",
}
CANONICAL_CONFIG = REPO_ROOT / "GraphBAN" / "configs" / "GraphBAN.yaml"


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


def load_calibration(corpus: str) -> dict:
    sidecar = CALIBRATION_SIDECAR_BY_CORPUS[corpus]
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
        out = model(v_d, fcfp, v_p, esm)
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
    ap.add_argument("--ckpt",  type=Path, default=None)
    ap.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dir = CKPT_DIR_BY_CORPUS[args.corpus]
    ckpt = args.ckpt or best_checkpoint(seed_dir)
    cfg = yaml.safe_load(open(args.config))
    calib = load_calibration(args.corpus)
    print(f"  ckpt: {ckpt}", file=sys.stderr)
    print(f"  thr: {calib['threshold']:.3f}", file=sys.stderr)

    df = pd.read_csv(args.pairs, sep="\t")
    drugban_df = pd.DataFrame({
        "SMILES":  df["smiles"].astype(str),
        "Protein": df["sequence"].astype(str),
        "Y":       np.zeros(len(df), dtype=int),
    })
    dataset = DTIDataset(drugban_df.index.values, drugban_df)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=graph_collate_func)

    model = GraphBAN(**cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    probs = predict(model, loader, device)
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
