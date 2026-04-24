#!/usr/bin/env python3
"""DT-Kinase v8 — training entry point.

Minimal training loop mirroring v7's protocol (Focal Loss + AdamW +
early stop + Platt + MCC-optimal threshold) but using
InteractionMapCNNv8 + v8 dataloaders.

Usage:
    python3 scripts/v8/train_v8.py \\
        --config configs/v8.yaml \\
        --dataset non_human \\
        --seed 42 \\
        --ablation v8-full \\
        --output-dir results/v8/non_human/full/seed_42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.config import (  # noqa: E402
    SUPPORTED_EMBEDDINGS, PROTEIN_DIMS, MOLFORMER_DIM, DEFAULT_SCAFFOLD_SPLIT_DIR,
)
from benchmark.levels.level4_cnn import FocalLoss  # noqa: E402
from benchmark.levels.level4_cnn_v8 import (  # noqa: E402
    InteractionMapCNNv8, build_v8_dataloaders,
)


# ============================================================================
# Ablation presets: which v8 features are enabled per config label
# ============================================================================

_ABLATION_ENABLES = {
    "v8-lig":  dict(chemberta=True,  admet=True,  classyfire=True,
                    biobert=False, pfam=False, taxonomy=False),
    "v8-prot": dict(chemberta=False, admet=False, classyfire=False,
                    biobert=True,  pfam=True,  taxonomy=True),
    "v8-full": dict(chemberta=True,  admet=True,  classyfire=True,
                    biobert=True,  pfam=True,  taxonomy=True),
}

_CORPUS_FILTER = {"human": "human", "non_human": "non_human", "all": None}


def _infer_dim_from_cache(cache_root: Path, corpus: str, feature: str,
                           default: int) -> int:
    """Peek at one .npy to learn the actual feature dimensionality."""
    d = cache_root / f"{feature}_{corpus}"
    if not d.exists():
        return default
    for p in d.iterdir():
        if p.suffix == ".npy":
            arr = np.load(p, mmap_mode="r")
            return int(arr.shape[-1])
    return default


def _evaluate(model, loader, device, *, threshold=None, calibrator=None,
              desc=""):
    model.eval()
    logits_all, labels_all = [], []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            p  = batch["protein_matrix"].to(device, non_blocking=True)
            l  = batch["ligand_matrix"].to(device, non_blocking=True)
            pm = batch["protein_mask"].to(device, non_blocking=True)
            lm = batch["ligand_mask"].to(device, non_blocking=True)
            y  = batch["label"].numpy()
            extra = {k: batch[k].to(device, non_blocking=True)
                     for k in ("chemberta_tokens", "chemberta_mask", "biobert_tokens",
                               "biobert_mask", "admet", "classyfire", "pfam", "taxonomy")
                     if k in batch}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(p, l, pm, lm, **extra)
            logits_all.append(logits.float().cpu().numpy().ravel())
            labels_all.append(y)
    logits_np = np.concatenate(logits_all)
    labels_np = np.concatenate(labels_all).astype(int)
    if calibrator is not None:
        probs = calibrator.predict_proba(logits_np.reshape(-1, 1))[:, 1]
    else:
        probs = 1.0 / (1.0 + np.exp(-logits_np))
    if threshold is None:
        thr, mcc = _best_mcc_threshold(labels_np, probs)
    else:
        thr = threshold
        mcc = float(matthews_corrcoef(labels_np, (probs >= thr).astype(int)))
    preds = (probs >= thr).astype(int)
    metrics = dict(
        mcc=mcc,
        accuracy=float(accuracy_score(labels_np, preds)),
        f1=float(f1_score(labels_np, preds, zero_division=0)),
        precision=float(precision_score(labels_np, preds, zero_division=0)),
        recall=float(recall_score(labels_np, preds, zero_division=0)),
        auroc=float(roc_auc_score(labels_np, probs)) if len(set(labels_np)) == 2 else 0.0,
        threshold=float(thr),
        y_true=labels_np,
        y_prob=probs,
        raw_logits=logits_np,
    )
    if desc:
        tqdm.write(f"    {desc}: MCC={mcc:.4f} thr={thr:.3f} AUROC={metrics['auroc']:.4f}")
    return metrics


def _best_mcc_threshold(y, p):
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5, 0.0
    grid = np.linspace(0.01, 0.99, 100)
    best, best_mcc = 0.5, -1.0
    for t in grid:
        m = float(matthews_corrcoef(y, (p >= t).astype(int)))
        if m > best_mcc:
            best, best_mcc = float(t), m
    lo, hi = max(0.01, best - 0.05), min(0.99, best + 0.05)
    for t in np.linspace(lo, hi, 100):
        m = float(matthews_corrcoef(y, (p >= t).astype(int)))
        if m > best_mcc:
            best, best_mcc = float(t), m
    return best, best_mcc


def _fit_platt(model, val_loader, device):
    model.eval()
    logits_all, labels_all = [], []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in val_loader:
            p  = batch["protein_matrix"].to(device, non_blocking=True)
            l  = batch["ligand_matrix"].to(device, non_blocking=True)
            pm = batch["protein_mask"].to(device, non_blocking=True)
            lm = batch["ligand_mask"].to(device, non_blocking=True)
            extra = {k: batch[k].to(device, non_blocking=True)
                     for k in ("chemberta_tokens", "chemberta_mask", "biobert_tokens",
                               "biobert_mask", "admet", "classyfire", "pfam", "taxonomy")
                     if k in batch}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(p, l, pm, lm, **extra)
            logits_all.append(logits.float().cpu().numpy().ravel())
            labels_all.append(batch["label"].numpy())
    x = np.concatenate(logits_all).reshape(-1, 1)
    y = np.concatenate(labels_all).astype(int)
    cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000).fit(x, y)
    tqdm.write(f"    Platt: a={cal.coef_[0, 0]:.4f}, b={cal.intercept_[0]:.4f}")
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v8.yaml")
    ap.add_argument("--dataset", required=True, choices=["human", "non_human", "all"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ablation", required=True, choices=list(_ABLATION_ENABLES))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load((REPO / args.config).read_text())
    l4 = cfg["level4_cnn_v8"]
    adapter = l4.get("adapter", {})
    v8 = l4["v8"]
    cache_root = REPO / v8["cache_root"]

    enabled_raw = _ABLATION_ENABLES[args.ablation]
    # Keep only features enabled in config (feature-level gating on top of ablation)
    per_feature = {
        "chemberta":   enabled_raw["chemberta"]  and v8["ligand"]["chemberta"]["enable"],
        "admet":       enabled_raw["admet"]      and v8["ligand"]["admet"]["enable"],
        "classyfire":  enabled_raw["classyfire"] and v8["ligand"]["classyfire"]["enable"],
        "biobert":     enabled_raw["biobert"]    and v8["protein"]["biobert"]["enable"],
        "pfam":        enabled_raw["pfam"]       and v8["protein"]["pfam"]["enable"],
        "taxonomy":    enabled_raw["taxonomy"]   and v8["protein"]["taxonomy"]["enable"],
    }
    print(f"[v8 {args.ablation}/{args.dataset}/seed={args.seed}] enabled: {per_feature}")

    # Auto-detect real feature dims from the cache (fallback to config values
    # when a cache is absent). Essential for variable-width features like
    # ADMET (dim depends on admet_ai version / physchem flag), ClassyFire
    # (vocab size depends on corpus), Pfam (top-K is corpus-specific), and
    # Taxonomy (lineage vocab is corpus-specific).
    admet_dim = _infer_dim_from_cache(cache_root, args.dataset, "admet",
                                       v8["ligand"]["admet"]["dim"])
    classyfire_dim = _infer_dim_from_cache(cache_root, args.dataset, "classyfire",
                                           v8["ligand"]["classyfire"]["dim"])
    chemberta_dim = _infer_dim_from_cache(cache_root, args.dataset, "chemberta",
                                           v8["ligand"]["chemberta"]["dim"])
    biobert_dim = _infer_dim_from_cache(cache_root, args.dataset, "biobert",
                                         v8["protein"]["biobert"]["dim"])
    pfam_dim = _infer_dim_from_cache(cache_root, args.dataset, "pfam",
                                     v8["protein"]["pfam"]["dim"])
    taxonomy_dim = _infer_dim_from_cache(cache_root, args.dataset, "taxonomy",
                                         v8["protein"]["taxonomy"]["dim"])
    print(f"  feature dims (detected): chemberta={chemberta_dim}  admet={admet_dim}  "
          f"classyfire={classyfire_dim}  biobert={biobert_dim}  pfam={pfam_dim}  "
          f"taxonomy={taxonomy_dim}")

    embedding_short = cfg.get("embedding", "8M")
    embedding_name = SUPPORTED_EMBEDDINGS.get(embedding_short, embedding_short)
    protein_dim = PROTEIN_DIMS.get(embedding_name, 320)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_double = bool(l4.get("double", False))
    no_amp = bool(l4.get("no_amp", False))
    dtype = torch.float64 if use_double else torch.float32
    use_amp = device.type == "cuda" and not no_amp and not use_double

    batch_size = int(l4.get("batch_size", 128))
    scaffold_dir = str(REPO / DEFAULT_SCAFFOLD_SPLIT_DIR)

    train_loader, val_loader, test_loader = build_v8_dataloaders(
        dataset_type="all",            # always use universal matrices
        embedding_name=embedding_name,
        scaffold_split_dir=scaffold_dir,
        batch_size=batch_size,
        enabled=per_feature,
        cache_root=cache_root,
        dataset_source_filter=_CORPUS_FILTER[args.dataset],
        cache_corpus=args.dataset,     # v8 caches built per corpus (non_human/human/all)
        mode="test",
    )

    # ---- Build v8 model ----
    model = InteractionMapCNNv8(
        protein_dim=protein_dim,
        ligand_dim=MOLFORMER_DIM,
        num_heads=int(l4["num_heads"]),
        head_dim=int(l4["head_dim"]),
        cnn_channels=int(l4["channels"]),
        dropout=float(l4["dropout"]),
        variant=l4.get("variant", "v7"),
        num_cross_layers=2,
        mlp_head=bool(l4.get("mlp_head", False)),
        cosine_sim=bool(l4.get("cosine_sim", False)),
        use_adapter=bool(adapter.get("enabled", False)),
        adapter_bottleneck_prot=int(adapter.get("prot_dim", 256)),
        adapter_bottleneck_lig=int(adapter.get("lig_dim", 512)),
        adapter_layers=int(adapter.get("layers", 1)),
        adapter_self_attn=bool(adapter.get("self_attn", False)),
        contrastive_dim=0,
        cosine_feat=False,
        # v8 feature gates
        enable_chemberta=per_feature["chemberta"],
        enable_admet=per_feature["admet"],
        enable_classyfire=per_feature["classyfire"],
        enable_biobert=per_feature["biobert"],
        enable_pfam=per_feature["pfam"],
        enable_taxonomy=per_feature["taxonomy"],
        chemberta_dim=chemberta_dim,
        biobert_dim=biobert_dim,
        admet_dim=admet_dim,
        classyfire_dim=classyfire_dim,
        pfam_dim=pfam_dim,
        taxonomy_dim=taxonomy_dim,
    ).to(device=device, dtype=dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model built: {trainable:,} trainable params")

    # ---- Optimizer (differential LR for adapter) ----
    lr = 1e-3
    weight_decay = float(l4.get("weight_decay", 0.02))
    lr_mult = float(adapter.get("lr_mult", 1.0))
    if lr_mult != 1.0 and adapter.get("enabled", False):
        a_ids = set()
        ap_list = list(model.prot_adapter.parameters()) + list(model.lig_adapter.parameters())
        for p in ap_list:
            a_ids.add(id(p))
        other = [p for p in model.parameters() if id(p) not in a_ids and p.requires_grad]
        optim = torch.optim.AdamW(
            [{"params": ap_list, "lr": lr * lr_mult}, {"params": other, "lr": lr}],
            weight_decay=weight_decay,
            fused=(device.type == "cuda" and not use_double),
        )
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  fused=(device.type == "cuda" and not use_double))

    # ---- Loss ----
    # Focal loss with adaptive positive weight
    labels_train = []
    for batch in train_loader:
        labels_train.append(batch["label"].numpy())
        break  # sample enough for alpha estimate on first batch — repeat over dataset below
    # Proper estimate: iterate once
    pos = 0; tot = 0
    for batch in train_loader:
        yy = batch["label"].numpy()
        pos += int(yy.sum()); tot += len(yy)
    alpha_pos = float(np.clip((tot - pos) / max(pos, 1), 1.0, 20.0))
    print(f"  alpha_pos={alpha_pos:.3f}  (pos={pos}/{tot})")
    loss_fn = FocalLoss(gamma=float(l4.get("focal_gamma", 2.0)), alpha=alpha_pos)

    scaler = torch.amp.GradScaler(enabled=use_amp)
    epochs = int(cfg.get("epochs", 200))
    patience = int(l4.get("patience", 20))
    ckpt_every = int(l4.get("checkpoint_every", 5))

    best_val_mcc = -1.0
    best_state = None
    epochs_without_improve = 0
    start_epoch = 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    # ---- Resume from checkpoint if present ----
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            optim.load_state_dict(ck["optim_state"])
            if "scaler_state" in ck and ck["scaler_state"] is not None:
                scaler.load_state_dict(ck["scaler_state"])
            best_val_mcc = float(ck["best_val_mcc"])
            best_state = {k: v.to(device=device, dtype=dtype) for k, v in ck["best_state"].items()}
            epochs_without_improve = int(ck["epochs_without_improve"])
            start_epoch = int(ck["epoch"]) + 1
            if "torch_rng" in ck:
                torch.random.set_rng_state(ck["torch_rng"])
            if "np_rng" in ck:
                np.random.set_state(ck["np_rng"])
            print(f"  resume: checkpoint @ epoch {ck['epoch']}  "
                  f"best_val_MCC={best_val_mcc:.4f}  epochs_without_improve={epochs_without_improve}")
        except Exception as e:
            print(f"  [warn] checkpoint corrupted or incompatible ({e}) — starting from scratch")
            start_epoch = 1

    def _save_checkpoint(epoch: int) -> None:
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optim_state": optim.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "best_val_mcc": best_val_mcc,
            "best_state": best_state if best_state is not None else
                          {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epochs_without_improve": epochs_without_improve,
            "torch_rng": torch.random.get_rng_state(),
            "np_rng": np.random.get_state(),
            "ablation": args.ablation,
            "dataset": args.dataset,
            "seed": args.seed,
        }, tmp)
        os.replace(tmp, ckpt_path)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            p  = batch["protein_matrix"].to(device=device, dtype=dtype, non_blocking=True)
            l  = batch["ligand_matrix"].to(device=device, dtype=dtype, non_blocking=True)
            pm = batch["protein_mask"].to(device, non_blocking=True)
            lm = batch["ligand_mask"].to(device, non_blocking=True)
            y  = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            extra = {k: batch[k].to(device, non_blocking=True)
                     for k in ("chemberta_tokens", "chemberta_mask", "biobert_tokens",
                               "biobert_mask", "admet", "classyfire", "pfam", "taxonomy")
                     if k in batch}
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(p, l, pm, lm, **extra).squeeze(-1)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            running += loss.item() * y.size(0)
        train_loss = running / max(len(train_loader.dataset), 1)

        val_metrics = _evaluate(model, val_loader, device, desc=f"epoch {epoch} val")
        val_mcc = val_metrics["mcc"]
        print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  val_MCC={val_mcc:.4f}")

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Periodic resumable checkpoint
        if epoch % ckpt_every == 0 or val_mcc > best_val_mcc - 1e-12:
            _save_checkpoint(epoch)

        if epochs_without_improve >= patience:
            print(f"  early stop at epoch {epoch} (best val_MCC={best_val_mcc:.4f})")
            _save_checkpoint(epoch)
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # ---- Platt + threshold + test ----
    calibrator = _fit_platt(model, val_loader, device)
    val_metrics = _evaluate(model, val_loader, device, calibrator=calibrator, desc="val (post-Platt)")
    val_thr = val_metrics["threshold"]
    test_metrics = _evaluate(model, test_loader, device,
                             threshold=val_thr, calibrator=calibrator, desc="test")

    # ---- Persist artifacts ----
    np.savez(
        out_dir / "raw_predictions.npz",
        val_y_true=val_metrics["y_true"], val_y_prob=val_metrics["y_prob"],
        val_raw_logits=val_metrics["raw_logits"],
        test_y_true=test_metrics["y_true"], test_y_prob=test_metrics["y_prob"],
        test_raw_logits=test_metrics["raw_logits"],
    )
    torch.save(model.state_dict(), out_dir / "v8_model.pt")
    report = {
        "ablation": args.ablation,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_val_mcc": float(best_val_mcc),
        "val_threshold": float(val_thr),
        "platt_a": float(calibrator.coef_[0, 0]),
        "platt_b": float(calibrator.intercept_[0]),
        "test_mcc": float(test_metrics["mcc"]),
        "test_auroc": float(test_metrics["auroc"]),
        "test_f1": float(test_metrics["f1"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "n_test": int(len(test_metrics["y_true"])),
        "enabled": per_feature,
        "feature_dims": dict(chemberta=chemberta_dim, admet=admet_dim,
                             classyfire=classyfire_dim, biobert=biobert_dim,
                             pfam=pfam_dim, taxonomy=taxonomy_dim),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    # Training complete — remove resumable checkpoint to save disk space.
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"[done] MCC={test_metrics['mcc']:.4f} AUROC={test_metrics['auroc']:.4f} "
          f"→ {out_dir}")


if __name__ == "__main__":
    main()
