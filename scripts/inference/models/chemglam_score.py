"""ChemGLaM scoring adapter for committee inference.

Reads the canonical ``pairs.tsv`` contract and emits ``scores_chemglam.csv``
with the same operational schema as DT-Kinase, DrugBAN, GraphBAN and ConPLex.
The five canonical checkpoints are ensembled by arithmetic mean; each seed's
MCC-optimal validation threshold comes from ``chemglam_calibration.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CHEMGLAM_ROOT = REPO_ROOT / "ChemGLaM"
sys.path.insert(0, str(CHEMGLAM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

from device_utils import empty_cache, pick_device  # noqa: E402
from chemglam.data.datamodule import DTIDataModule  # noqa: E402
from chemglam.model.chemglam import ChemGLaM  # noqa: E402
from chemglam.utils.config import Config  # noqa: E402


CANONICAL_SEEDS = (42, 123, 456, 789, 1024)
BASE_CONFIG = REPO_ROOT / "configs" / "chemglam_universal.json"


def checkpoint_path(corpus: str, seed: int) -> Path:
    return REPO_ROOT / "logs" / f"chemglam_{corpus}_seed{seed}" / "best_checkpoint.ckpt"


def calibration_path(corpus: str, seed: int) -> Path:
    return (REPO_ROOT / "results" / "chemglam" / f"chemglam_{corpus}_seed{seed}"
            / "chemglam_calibration.json")


def load_calibration(corpus: str, seed: int) -> dict:
    path = calibration_path(corpus, seed)
    if not path.exists():
        sys.exit(
            f"calibration sidecar not found for ChemGLaM seed {seed}: {path}\n"
            "finish the canonical benchmark before enabling ChemGLaM in the committee"
        )
    result = json.loads(path.read_text())
    if result.get("corpus") != corpus or int(result.get("seed", -1)) != seed:
        sys.exit(f"calibration provenance mismatch in {path}")
    return result


def _validate_pairs(pairs: pd.DataFrame) -> None:
    required = {"uniprot", "sequence", "chembl_id", "smiles"}
    missing = required.difference(pairs.columns)
    if missing:
        sys.exit(f"pairs.tsv missing columns: {sorted(missing)}")
    conflicts = pairs.groupby("uniprot")["sequence"].nunique(dropna=False)
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
        sys.exit(f"same uniprot maps to different sequences: {bad}")


def _materialize_inference_config(
    pairs: pd.DataFrame, corpus: str, seed: int, directory: Path
) -> Config:
    data_path = directory / "pairs.csv"
    config_path = directory / "config.json"
    pd.DataFrame({
        "smiles": pairs["smiles"].astype(str),
        "target_sequence": pairs["sequence"].astype(str),
        "target_id": pairs["uniprot"].astype(str),
    }).to_csv(data_path, index=False)

    base = json.loads(BASE_CONFIG.read_text())
    digest_data = pd.util.hash_pandas_object(
        pairs[["uniprot", "sequence"]].drop_duplicates(), index=False
    ).values.tobytes()
    digest = hashlib.sha256(digest_data).hexdigest()[:16]
    base.update(
        experiment_name=f"chemglam_committee_{corpus}_seed{seed}",
        cache_dir=f"chemglam_committee/{digest}",
        dataset_csv_path=str(data_path),
        split_json_path=None,
        checkpoint_path=str(checkpoint_path(corpus, seed)),
        target_columns=None,
        seed=seed,
        deterministic_eval=True,
        num_workers=0,
    )
    config_path.write_text(json.dumps(base, indent=2) + "\n")
    return Config(config_path)


@torch.inference_mode()
def predict(model: ChemGLaM, loader, device: torch.device) -> np.ndarray:
    probabilities = []
    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        logits, _ = model(batch)
        prob = torch.sigmoid(logits.float()).reshape(-1)
        probabilities.append(
            torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0).cpu().numpy()
        )
    return np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", choices=("human", "non_human", "all"), default="all")
    parser.add_argument("--ckpt", type=Path,
                        help="explicit checkpoint (single seed; requires --seeds with one value)")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = pick_device()
    if device.type != "cuda":
        sys.exit("ChemGLaM canonical inference requires CUDA (ESM-2 3B preprocessing is CUDA-bound)")
    pairs = pd.read_csv(args.pairs, sep="\t")
    _validate_pairs(pairs)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.ckpt is not None and len(seeds) != 1:
        sys.exit("--ckpt requires exactly one value in --seeds")

    specs = []
    for seed in seeds:
        ckpt = args.ckpt if args.ckpt is not None else checkpoint_path(args.corpus, seed)
        if not ckpt.exists():
            sys.exit(f"ChemGLaM checkpoint not found for seed {seed}: {ckpt}")
        specs.append((seed, ckpt, load_calibration(args.corpus, seed)))

    probabilities = np.empty((len(specs), len(pairs)), dtype=np.float64)
    thresholds = np.empty(len(specs), dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="chemglam_committee_") as temp:
        temp_dir = Path(temp)
        # Protein embeddings depend only on the input pairs and are reused by
        # all seed checkpoints through the deterministic cache key.
        config = _materialize_inference_config(pairs, args.corpus, specs[0][0], temp_dir)
        config.batch_size = args.batch_size
        datamodule = DTIDataModule(config)
        datamodule.prepare_data()
        datamodule.setup("predict")
        if len(datamodule.dataset) != len(pairs):
            sys.exit(
                "ChemGLaM canonicalization dropped input rows; fix invalid SMILES before committee scoring"
            )
        loader = datamodule.predict_dataloader()

        for index, (seed, ckpt, calibration) in enumerate(specs):
            print(
                f"  [seed {index + 1}/{len(specs)}] ckpt={ckpt.parent.name} "
                f"thr={float(calibration['threshold']):.3f}", file=sys.stderr,
            )
            config.seed = seed
            config.checkpoint_path = str(ckpt)
            model = ChemGLaM.load_from_checkpoint(ckpt, config=config, map_location=device)
            model.to(device).eval()
            probabilities[index] = predict(model, loader, device)
            thresholds[index] = float(calibration["threshold"])
            del model
            empty_cache(device)

    mean_probability = probabilities.mean(axis=0)
    mean_threshold = float(thresholds.mean())
    output = pd.DataFrame({
        "uniprot": pairs["uniprot"].to_numpy(),
        "chembl_id": pairs["chembl_id"].to_numpy(),
        "prob": mean_probability,
        "pred": (mean_probability >= mean_threshold).astype(int),
        "threshold": mean_threshold,
        "n_seeds": len(specs),
        "prob_std": probabilities.std(axis=0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"  wrote {len(output)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
