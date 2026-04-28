"""DrugBAN scoring adapter for the committee orchestrator.

Reads pairs.tsv (uniprot, sequence, chembl_id, smiles, source) and emits
scores_drugban.csv (uniprot, chembl_id, prob, pred, threshold).

Wraps the existing infer_drugban_universal.py inference pipeline, but
replaces the universal_split TSV input with the committee's pairs.tsv
format. Must run inside the `drugban` conda env (see DrugBAN/setup_env.sh).

The threshold is loaded from a sidecar JSON adjacent to the checkpoint
(``drugban_calibration.json``); falls back to 0.5 with a warning if missing.
The DrugBAN native protocol uses F1-optimal threshold on validation, which
the sidecar persists from the original training run.
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

DRUGBAN_SRC = REPO_ROOT / "DrugBAN" / "src"
if not (DRUGBAN_SRC / "dataloader.py").exists():
    sys.exit(
        f"DrugBAN upstream not installed at {DRUGBAN_SRC}.\n"
        f"Install via: bash DrugBAN/setup_env.sh"
    )
sys.path.insert(0, str(DRUGBAN_SRC))

from dataloader import DTIDataset  # type: ignore
from models import DrugBAN  # type: ignore

try:
    from dataloader import graph_collate_func  # type: ignore
except ImportError:
    import dgl

    def graph_collate_func(batch):  # type: ignore
        drugs, proteins, labels = zip(*batch)
        return (
            dgl.batch(drugs),
            torch.as_tensor(np.array(proteins)),
            torch.as_tensor(np.array(labels)),
        )


CANONICAL_SEEDS = (42, 123, 456, 789, 1024)


def ckpt_seed_dir(corpus: str, seed: int) -> Path:
    """DrugBAN trained ckpts are under DrugBAN/results/{corpus}/seed_X/."""
    return REPO_ROOT / "DrugBAN" / "results" / corpus / f"seed_{seed}"


def calibration_sidecar(corpus: str, seed: int) -> Path:
    """Calibration sidecar is under DrugBAN/results_universal/results_universal/."""
    return (REPO_ROOT / "DrugBAN" / "results_universal" / "results_universal"
            / corpus / f"seed_{seed}" / "drugban_calibration.json")


# Backwards-compat single-seed map (seed_42 default).
CKPT_DIR_BY_CORPUS = {c: ckpt_seed_dir(c, 42) for c in ("human", "non_human", "all")}
CALIBRATION_SIDECAR_BY_CORPUS = {
    c: calibration_sidecar(c, 42) for c in ("human", "non_human", "all")
}
CANONICAL_CONFIG = REPO_ROOT / "DrugBAN" / "configs" / "DrugBAN.yaml"


def pairs_to_drugban_df(pairs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pairs_path, sep="\t")
    return pd.DataFrame({
        "SMILES":  df["smiles"].astype(str),
        "Protein": df["sequence"].astype(str),
        "Y":       np.zeros(len(df), dtype=int),  # placeholder; not used in inference
    })


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
    for v_d, v_p, _ in loader:
        v_d = v_d.to(device, non_blocking=True)
        v_p = v_p.to(device, non_blocking=True)
        _, _, _, f = model(v_d, v_p)
        prob = torch.softmax(f.float(), dim=1)[:, 1]
        prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        ps.append(prob.cpu().numpy())
    return np.concatenate(ps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score pairs.tsv with DrugBAN.")
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--corpus", choices=["human", "non_human", "all"], default="all")
    ap.add_argument("--ckpt",  type=Path, default=None,
                    help="explicit ckpt path (overrides --corpus + --seeds; "
                         "single-seed mode)")
    ap.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                    help="comma-separated seeds for the 5-seed ensemble "
                         "(default: all five canonical seeds)")
    ap.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    print(f"  device: {device}", file=sys.stderr)

    # Resolve ensemble: list of (ckpt_path, calibration) per seed.
    if args.ckpt is not None:
        seed_specs = [(args.ckpt, load_calibration(args.corpus, 42))]
    else:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        seed_specs = []
        for s in seed_list:
            sd = ckpt_seed_dir(args.corpus, s)
            seed_specs.append((best_checkpoint(sd), load_calibration(args.corpus, s)))
    print(f"  ensemble: {len(seed_specs)} seed(s) (corpus={args.corpus})",
          file=sys.stderr)

    cfg = yaml.safe_load(open(args.config))

    df = pairs_to_drugban_df(args.pairs)
    dataset = DTIDataset(df.index.values, df)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=graph_collate_func)

    pairs = pd.read_csv(args.pairs, sep="\t")
    n = len(pairs)
    probs_per_seed = np.empty((len(seed_specs), n), dtype=np.float64)
    thresholds_per_seed = np.empty(len(seed_specs), dtype=np.float64)

    for k, (ckpt, calib) in enumerate(seed_specs):
        print(f"  [seed {k+1}/{len(seed_specs)}] ckpt={ckpt.parent.name} "
              f"thr={calib['threshold']:.3f}", file=sys.stderr)
        model = DrugBAN(**cfg).to(device)
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
        "uniprot":   pairs["uniprot"].to_numpy(),
        "chembl_id": pairs["chembl_id"].to_numpy(),
        "prob":      probs,
        "pred":      (probs >= threshold).astype(int),
        "threshold": threshold,
        "n_seeds":   len(seed_specs),
        "prob_std":  probs_per_seed.std(axis=0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"  wrote {len(out)} rows → {args.out} "
          f"(ensemble of {len(seed_specs)} seeds, mean threshold={threshold:.3f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
