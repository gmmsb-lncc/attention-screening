#!/usr/bin/env python3
"""Train/evaluate one canonical CMA-DTI seed without test leakage."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CMA_ROOT = REPO_ROOT / "CMA-DTI"
sys.path.insert(0, str(CMA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataloader import DTIDataset  # type: ignore  # noqa: E402
from features import RaggedStore, ensure_feature_cache  # noqa: E402
from modeling import CachedCMA  # noqa: E402
from protocol import best_mcc_threshold, metrics  # noqa: E402


UPSTREAM_THRESHOLD_AUDIT = {
    "status": "confirmed_in_pinned_official_code",
    "finding": "official code selects a decision threshold using test labels",
    "publication": "threshold selected on validation and applied to test",
    "paper_result_leakage": "undetermined_without_original_run_artifacts",
    "affected_metrics": ["f1", "accuracy", "precision", "sensitivity", "specificity"],
    "unaffected_components": ["model_weights", "best_checkpoint", "auroc", "auprc"],
    "canonical_remediation": "validation MCC threshold frozen before test evaluation",
    "report": "docs/06-validation-reports/CMADTI_REPRODUCIBILITY_AUDIT.md",
}


class IndexedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, max_nodes: int):
        self.frame = frame.reset_index(drop=True)
        self.base = DTIDataset(np.arange(len(frame)), self.frame, max_drug_nodes=max_nodes)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        graph, smiles, _protein, label = self.base[index]
        row = self.frame.iloc[index]
        return graph, str(smiles), str(row["target_id"]), float(label), int(index)


def collate_factory(proteins: RaggedStore, smiles: RaggedStore):
    def collate(batch):
        graphs, smiles_keys, target_ids, labels, indices = zip(*batch)
        chem = [torch.from_numpy(smiles.get(key)) for key in smiles_keys]
        prot = [torch.from_numpy(proteins.get(key)) for key in target_ids]
        chem_lengths = torch.tensor([len(value) for value in chem])
        prot_lengths = torch.tensor([len(value) for value in prot])
        chem_pad = pad_sequence(chem, batch_first=True)
        prot_pad = pad_sequence(prot, batch_first=True)
        chem_mask = torch.arange(chem_pad.shape[1])[None] < chem_lengths[:, None]
        prot_mask = torch.arange(prot_pad.shape[1])[None] < prot_lengths[:, None]
        return (
            dgl.batch(graphs), chem_pad, chem_mask, prot_pad, prot_mask,
            torch.tensor(labels, dtype=torch.float32), np.asarray(indices),
        )
    return collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_inputs(batch, device):
    graph, chem, chem_mask, prot, prot_mask, labels, indices = batch
    return (
        graph.to(device), chem.to(device), chem_mask.to(device),
        prot.to(device), prot_mask.to(device), labels.to(device), indices,
    )


@torch.inference_mode()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels_out, probabilities, indices_out = [], [], []
    total_loss = 0.0
    for batch in loader:
        graph, chem, chem_mask, prot, prot_mask, labels, indices = move_inputs(batch, device)
        logits, _ = model(graph, chem, chem_mask, prot, prot_mask)
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        ).item()
        labels_out.append(labels.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        indices_out.append(indices)
    order = np.argsort(np.concatenate(indices_out))
    return (
        np.concatenate(labels_out)[order], np.concatenate(probabilities)[order],
        total_loss / len(order),
    )


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        graph, chem, chem_mask, prot, prot_mask, labels, _ = move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(graph, chem, chem_mask, prot, prot_mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def raw_fields(split: str, frame: pd.DataFrame, y: np.ndarray, prob: np.ndarray) -> dict:
    fields = {
        f"{split}_y_true": y.astype(np.int8),
        f"{split}_y_prob": prob.astype(np.float32),
    }
    for column in ("source_row", "target_id", "chembl_id", "SMILES", "Protein", "dataset_source"):
        if column in frame:
            fields[f"{split}_{column.lower()}"] = frame[column].fillna("").to_numpy(dtype=str)
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("human", "non_human", "all"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cmadti_universal.yaml"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise SystemExit("canonical CMA-DTI training requires CUDA")
    device = torch.device("cuda")
    set_seed(args.seed)
    config = yaml.safe_load(args.config.read_text())
    data_root = args.data_root or Path("data/cmadti") / args.corpus
    output = args.output or Path("results/cmadti") / f"cmadti_{args.corpus}_seed{args.seed}"
    cache_root = args.cache_root or Path("results/cmadti/feature_cache") / args.corpus
    output.mkdir(parents=True, exist_ok=True)
    frames = {split: pd.read_csv(data_root / f"{split}.csv") for split in ("train", "val", "test")}
    combined = pd.concat(frames.values(), ignore_index=True)
    protein_store, smiles_store = ensure_feature_cache(
        combined, cache_root, config, device, batch_size=args.feature_batch_size
    )
    collate = collate_factory(protein_store, smiles_store)
    max_nodes = int(config["model"]["max_drug_nodes"])
    batch_size = int(config["training"]["batch_size"])
    datasets = {split: IndexedDataset(frame, max_nodes) for split, frame in frames.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0, collate_fn=collate, generator=generator),
        "train_eval": DataLoader(datasets["train"], batch_size=batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=0, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate),
    }
    checkpoint = output / "best_model.pt"
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(CMA_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    model = CachedCMA(config, device).to(device)
    history = []
    best_epoch, best_auroc, best_state = 0, -np.inf, None
    if args.reuse_checkpoint and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("corpus") != args.corpus or int(payload.get("seed", -1)) != args.seed:
            raise ValueError(f"checkpoint provenance mismatch: {checkpoint}")
        if payload.get("config") == config:
            model.load_state_dict(payload["state_dict"])
            best_epoch = int(payload["best_epoch"])
            best_auroc = float(payload["best_val_auroc"])
        else:
            print(
                f"checkpoint config differs from the canonical configuration; "
                f"retraining instead of reusing {checkpoint}"
            )
    if best_epoch == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            loss = train_epoch(model, loaders["train"], optimizer, device)
            val_y, val_prob, val_loss = predict(model, loaders["val"], device)
            val_auroc = metrics(val_y, val_prob, 0.5)["auroc"]
            history.append({"epoch": epoch, "train_loss": loss,
                            "val_loss": val_loss, "val_auroc": val_auroc})
            print(f"epoch={epoch} train_loss={loss:.6f} val_loss={val_loss:.6f} val_auroc={val_auroc:.6f}")
            if val_auroc >= best_auroc:
                best_epoch, best_auroc = epoch, val_auroc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save({
            "state_dict": best_state, "best_epoch": best_epoch,
            "best_val_auroc": best_auroc, "seed": args.seed,
            "corpus": args.corpus, "config": config, "upstream_commit": upstream_commit,
        }, checkpoint)

    predictions = {}
    for split, loader_name in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        predictions[split] = predict(model, loaders[loader_name], device)
    threshold = best_mcc_threshold(predictions["val"][0], predictions["val"][1])
    split_metrics = {
        split: {**metrics(y, prob, threshold), "loss": loss}
        for split, (y, prob, loss) in predictions.items()
    }
    result = {
        "model": "CMA-DTI", "corpus": args.corpus, "seed": args.seed,
        "split": "universal_scaffold", "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "model_selection": "validation AUROC (upstream CMA-DTI criterion)",
        "threshold_optimization": (
            "validation MCC-optimal; publication specifies validation-selected threshold "
            "applied to test but does not specify its optimization metric"
        ),
        "checkpoint": str(checkpoint), "config": config,
        "upstream_commit": upstream_commit,
        "methodology_audit": {
            **UPSTREAM_THRESHOLD_AUDIT,
            "audited_upstream_commit": upstream_commit,
        },
        "elapsed_seconds": time.time() - started,
        "train": split_metrics["train"], "validation": split_metrics["val"],
        "test": split_metrics["test"], "history": history,
    }
    (output / "cmadti_results.json").write_text(json.dumps(result, indent=2) + "\n")
    calibration = {
        "threshold": threshold, "calibration_metric": "mcc",
        "val_score": split_metrics["val"]["mcc"], "n_val": len(frames["val"]),
        "model": "cmadti", "corpus": args.corpus, "seed": args.seed,
        "source": str(output / "raw_predictions.npz"), "checkpoint": str(checkpoint),
        "methodology_audit": {
            **UPSTREAM_THRESHOLD_AUDIT,
            "audited_upstream_commit": upstream_commit,
        },
    }
    (output / "cmadti_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    fields = {"threshold": np.asarray(threshold, dtype=np.float64)}
    for split, (y, prob, _loss) in predictions.items():
        fields.update(raw_fields(split, frames[split], y, prob))
    fields.update(y_true=predictions["test"][0].astype(np.int8),
                  y_prob=predictions["test"][1].astype(np.float32))
    np.savez_compressed(output / "raw_predictions.npz", **fields)
    print(json.dumps({"validation": split_metrics["val"], "test": split_metrics["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
