#!/usr/bin/env python3
"""Evaluate a trained v7 checkpoint on any {human, non_human, all} test split.

Used by run_cross_species.sh (H<->NH) and run_cross_matrix.sh (full 3x3).

v7 training saves only a bare state_dict (level4_cnn_model.pt).
Calibrator + threshold are recomputed here on the training corpus val set,
matching the protocol used during training (see benchmark/levels/level4_cnn.py).

Usage:
    python3 scripts/thesis_followups/eval_checkpoint_on_dataset.py \\
        --checkpoint results/.../level4_cnn_model.pt \\
        --train-corpus human \\
        --eval-dataset non_human \\
        --split test \\
        --output metrics.json \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from benchmark.config import (  # noqa: E402
    DEFAULT_SCAFFOLD_SPLIT_DIR,
    PROTEIN_DIMS,
    SUPPORTED_EMBEDDINGS,
    MOLFORMER_DIM,
)
from benchmark.levels.level4_cnn import (  # noqa: E402
    InteractionMapCNN,
    _evaluate,
    _platt_calibrate,
)
from benchmark.levels.matrix_utils import build_matrix_dataloaders  # noqa: E402


_CORPUS_FILTER = {"human": "human", "non_human": "non_human", "all": None}


def _load_v7_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _build_model(v7_config: dict, embedding_name: str, device: torch.device) -> InteractionMapCNN:
    l4 = v7_config["level4_cnn"]
    adapter = l4.get("adapter", {})
    full_emb = SUPPORTED_EMBEDDINGS.get(embedding_name, embedding_name)
    protein_dim = PROTEIN_DIMS.get(full_emb, 320)
    # Match training-time gate (benchmark/levels/level4_cnn.py:_train_interaction_cnn):
    # contrastive_dim is zeroed when contrastive_weight == 0 so the projection
    # heads are NOT instantiated. Otherwise ckpt mismatch: model has
    # prot_contrast_proj / lig_contrast_proj but state_dict lacks them.
    contrastive_weight = float(l4.get("contrastive_weight", 0.0))
    raw_contrastive_dim = int(l4.get("contrastive_dim", 128))
    contrastive_dim = raw_contrastive_dim if contrastive_weight > 0 else 0
    model = InteractionMapCNN(
        protein_dim=protein_dim,
        ligand_dim=MOLFORMER_DIM,
        num_heads=int(l4.get("num_heads", 8)),
        head_dim=int(l4.get("head_dim", 32)),
        cnn_channels=int(l4.get("channels", 64)),
        dropout=float(l4.get("dropout", 0.35)),
        variant=l4.get("variant", "v7"),
        num_cross_layers=int(l4.get("num_cross_layers", 2)),
        mlp_head=bool(l4.get("mlp_head", False)),
        cosine_sim=bool(l4.get("cosine_sim", False)),
        use_adapter=bool(adapter.get("enabled", False)),
        adapter_bottleneck_prot=int(adapter.get("prot_dim", 256)),
        adapter_bottleneck_lig=int(adapter.get("lig_dim", 512)),
        adapter_layers=int(adapter.get("layers", 1)),
        adapter_self_attn=bool(adapter.get("self_attn", False)),
        contrastive_dim=contrastive_dim,
        cosine_feat=bool(l4.get("cosine_feat", False)),
    )
    if bool(l4.get("double", False)):
        model = model.double()
    return model.to(device)


def _load_checkpoint_state(ckpt_path: Path, model: torch.nn.Module, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip torch.compile prefix if present
    cleaned = {}
    for k, v in state.items():
        k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        # Backward-compat: old ckpts used `query` (singular) before multi-head
        # pool refactor that renamed it to `queries` (plural). The shape is
        # identical for pool_num_heads=1 (canonical), so a key rename is
        # sufficient. See _AxisAttentionPool in benchmark/levels/level4_cnn.py.
        if k.endswith(".query"):
            k = k[:-len(".query")] + ".queries"
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)


def _build_loader(corpus: str, embedding: str, batch_size: int, mode: str):
    """Return (train, val, test|None) loaders for a given corpus.

    ``embedding`` is the shorthand (e.g. "8M") from v7.yaml.
    ``build_matrix_dataloaders`` expects the FULL model name
    (e.g. "esm2_t6_8M_UR50D") because that's what names the on-disk
    `results/.../{model_name}/build/{protein_matrices,molformer_matrix}/`
    subdirs. Translate via SUPPORTED_EMBEDDINGS.
    """
    scaffold_dir = str(REPO / DEFAULT_SCAFFOLD_SPLIT_DIR)
    dataset_type = "all"  # always resolve embedding dirs from both corpora
    full_emb = SUPPORTED_EMBEDDINGS.get(embedding, embedding)
    return build_matrix_dataloaders(
        dataset_type=dataset_type,
        embedding_name=full_emb,
        scaffold_split_dir=scaffold_dir,
        batch_size=batch_size,
        dataset_source_filter=_CORPUS_FILTER[corpus],
        mode=mode,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to level4_cnn_model.pt (bare state_dict)")
    ap.add_argument("--train-corpus", required=True,
                    choices=["human", "non_human", "all"],
                    help="Corpus used to train the checkpoint; its val set is used for calibration + threshold")
    ap.add_argument("--eval-dataset", required=True,
                    choices=["human", "non_human", "all"],
                    help="Corpus whose test/val split is evaluated")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/v7.yaml")
    ap.add_argument("--embedding", default=None,
                    help="Embedding name (defaults to v7.yaml setting)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Batch size (defaults to v7.yaml setting)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v7_cfg = _load_v7_config(REPO / args.config)
    l4 = v7_cfg["level4_cnn"]
    embedding = args.embedding or v7_cfg.get("embedding", "8M")
    batch_size = args.batch_size or int(l4.get("batch_size", 128))

    # 1) Build model + load weights ------------------------------------
    model = _build_model(v7_cfg, embedding, device)
    _load_checkpoint_state(Path(args.checkpoint), model, device)
    model.eval()

    # 2) Fit Platt calibrator on the training corpus val set -----------
    _, train_val_loader, _ = _build_loader(args.train_corpus, embedding, batch_size, mode="train")
    print(f"Fitting Platt on {args.train_corpus} val set…")
    calibrator = _platt_calibrate(model, train_val_loader, device)

    # 3) Derive MCC-optimal threshold on the training corpus val set ---
    val_result = _evaluate(
        model, train_val_loader, device,
        calibrator=calibrator, desc=f"Calib-val ({args.train_corpus})",
    )
    val_threshold = val_result["threshold"]

    # 4) Evaluate on requested split of the eval corpus ----------------
    _, eval_val_loader, eval_test_loader = _build_loader(
        args.eval_dataset, embedding, batch_size, mode="test",
    )
    eval_loader = eval_test_loader if args.split == "test" else eval_val_loader
    metrics = _evaluate(
        model, eval_loader, device,
        threshold=val_threshold, desc=f"Eval ({args.eval_dataset}/{args.split})",
        calibrator=calibrator,
    )

    # 5) Serialize -----------------------------------------------------
    out = {
        "mcc": float(metrics["mcc"]),
        "mcc_at_05": float(metrics["mcc_at_05"]),
        "auroc": float(metrics["auroc"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "threshold": float(val_threshold),
        "platt_a": float(calibrator.coef_[0, 0]),
        "platt_b": float(calibrator.intercept_[0]),
        "n_samples": int(len(metrics["y_true"])),
        "raw_logits": None,  # kept None to mirror legacy schema; y_prob is post-Platt
        "raw_labels": metrics["y_true"].tolist(),
        "y_prob": metrics["y_prob"].tolist(),
        "source_checkpoint": args.checkpoint,
        "train_corpus": args.train_corpus,
        "eval_dataset": args.eval_dataset,
        "split": args.split,
        "seed": args.seed,
        "embedding": embedding,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)

    print(
        f"[{args.train_corpus}->{args.eval_dataset}/{args.split} seed={args.seed}] "
        f"MCC={out['mcc']:.4f} AUROC={out['auroc']:.4f} F1={out['f1']:.4f} "
        f"thr={val_threshold:.3f} n={out['n_samples']}"
    )


if __name__ == "__main__":
    main()
