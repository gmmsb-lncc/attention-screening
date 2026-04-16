#!/usr/bin/env python3
"""Evaluate a trained v7 checkpoint on a different dataset's test split.

Used by run_cross_species.sh for Objective 5 validation.

Usage:
    python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \\
        --checkpoint results/.../best_model.pt \\
        --eval-dataset non_human \\
        --split test \\
        --output metrics.json \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

from benchmark.levels.level4_cnn import (  # noqa: E402
    InteractionMapCNN, _best_mcc_threshold, _platt_calibrate,
)


def load_eval_loader(dataset: str, split: str, seed: int):
    """Load the benchmark-standard test/val loader for the target dataset."""
    # NOTE: delegate to the canonical data loading path used by the
    # benchmark. The exact import lives under benchmark/levels/level4_cnn.py
    # (look for _build_loaders or its equivalent) — adapt here if the
    # signature differs.
    from benchmark.levels.level4_cnn import _build_loaders  # type: ignore
    _, val_loader, test_loader = _build_loaders(
        dataset=dataset, embedding="8M", seed=seed,
    )
    return test_loader if split == "test" else val_loader


def evaluate(model, loader, calibrator, threshold, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            prot = batch["protein"].to(device)
            lig = batch["ligand"].to(device)
            prot_mask = batch.get("protein_mask")
            lig_mask = batch.get("ligand_mask")
            if prot_mask is not None:
                prot_mask = prot_mask.to(device)
            if lig_mask is not None:
                lig_mask = lig_mask.to(device)
            y = batch["label"].to(device)
            logits = model(prot, lig, prot_mask, lig_mask)
            logits_all.append(logits.cpu().numpy())
            y_all.append(y.cpu().numpy())

    logits_np = np.concatenate(logits_all).reshape(-1)
    y_np = np.concatenate(y_all).reshape(-1).astype(int)

    if calibrator is not None:
        probs = calibrator.predict_proba(logits_np.reshape(-1, 1))[:, 1]
    else:
        probs = 1.0 / (1.0 + np.exp(-logits_np))

    preds = (probs >= threshold).astype(int)

    return {
        "mcc": float(matthews_corrcoef(y_np, preds)),
        "auroc": float(roc_auc_score(y_np, probs)),
        "f1": float(f1_score(y_np, preds)),
        "accuracy": float(accuracy_score(y_np, preds)),
        "precision": float(precision_score(y_np, preds, zero_division=0)),
        "recall": float(recall_score(y_np, preds, zero_division=0)),
        "threshold": float(threshold),
        "n_samples": int(len(y_np)),
        "raw_logits": logits_np.tolist(),
        "raw_labels": y_np.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-dataset", required=True,
                    choices=["human", "non_human", "all"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Rebuild the model from the checkpoint config
    model = InteractionMapCNN(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    calibrator = ckpt.get("calibrator")  # fitted on original training corpus
    threshold = ckpt.get("threshold", 0.5)

    loader = load_eval_loader(args.eval_dataset, args.split, args.seed)
    metrics = evaluate(model, loader, calibrator, threshold, device)
    metrics["source_checkpoint"] = args.checkpoint
    metrics["eval_dataset"] = args.eval_dataset
    metrics["seed"] = args.seed

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(
        f"[{args.eval_dataset}/{args.split}/seed={args.seed}] "
        f"MCC={metrics['mcc']:.4f} AUROC={metrics['auroc']:.4f} "
        f"F1={metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
