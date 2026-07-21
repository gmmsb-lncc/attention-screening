"""CMA-DTI five-seed scoring adapter for committee inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
CMADTI_SCRIPTS = REPO_ROOT / "scripts" / "cmadti"
sys.path.insert(0, str(CMADTI_SCRIPTS))

from features import ensure_feature_cache  # noqa: E402
from modeling import CachedCMA  # noqa: E402
from train_canonical import IndexedDataset, collate_factory, predict  # noqa: E402


def checkpoint_path(corpus: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "cmadti" / f"cmadti_{corpus}_seed{seed}" / "best_model.pt"


def calibration_path(corpus: str, seed: int) -> Path:
    return REPO_ROOT / "results" / "cmadti" / f"cmadti_{corpus}_seed{seed}" / "cmadti_calibration.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", choices=("human", "non_human", "all"), default="all")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/cmadti_universal.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("canonical CMA-DTI inference requires CUDA")
    device = torch.device("cuda")
    pairs = pd.read_csv(args.pairs, sep="\t")
    required = {"uniprot", "sequence", "chembl_id", "smiles"}
    missing = required.difference(pairs.columns)
    if missing:
        sys.exit(f"pairs.tsv missing columns: {sorted(missing)}")
    frame = pd.DataFrame({
        "SMILES": pairs["smiles"].astype(str),
        "Protein": pairs["sequence"].astype(str),
        "target_id": pairs["uniprot"].astype(str),
        "chembl_id": pairs["chembl_id"].astype(str),
        "Y": np.zeros(len(pairs), dtype=int),
    })
    config = yaml.safe_load(args.config.read_text())
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame[["SMILES", "Protein", "target_id"]], index=False)
        .values.tobytes()
    ).hexdigest()[:16]
    cache_root = REPO_ROOT / "cache" / "cmadti_committee" / digest
    protein_store, smiles_store = ensure_feature_cache(frame, cache_root, config, device)
    dataset = IndexedDataset(frame, int(config["model"]["max_drug_nodes"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_factory(protein_store, smiles_store))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.ckpt is not None and len(seeds) != 1:
        sys.exit("--ckpt requires exactly one --seeds value")
    probabilities = np.empty((len(seeds), len(frame)), dtype=np.float64)
    thresholds = np.empty(len(seeds), dtype=np.float64)
    for index, seed in enumerate(seeds):
        checkpoint = args.ckpt or checkpoint_path(args.corpus, seed)
        sidecar = calibration_path(args.corpus, seed)
        if not checkpoint.exists() or not sidecar.exists():
            sys.exit(f"missing CMA-DTI checkpoint/calibration for corpus={args.corpus} seed={seed}")
        calibration = json.loads(sidecar.read_text())
        if calibration.get("corpus") != args.corpus or int(calibration.get("seed", -1)) != seed:
            sys.exit(f"calibration provenance mismatch: {sidecar}")
        model = CachedCMA(config, device).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("corpus") != args.corpus or int(payload.get("seed", -1)) != seed:
            sys.exit(f"checkpoint provenance mismatch: {checkpoint}")
        model.load_state_dict(payload["state_dict"])
        _labels, probabilities[index], _loss = predict(model, loader, device)
        thresholds[index] = float(calibration["threshold"])
        del model
        torch.cuda.empty_cache()
    mean_prob = probabilities.mean(0)
    threshold = float(thresholds.mean())
    output = pd.DataFrame({
        "uniprot": pairs["uniprot"], "chembl_id": pairs["chembl_id"],
        "prob": mean_prob, "pred": (mean_prob >= threshold).astype(int),
        "threshold": threshold, "n_seeds": len(seeds),
        "prob_std": probabilities.std(0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"wrote {len(output)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
