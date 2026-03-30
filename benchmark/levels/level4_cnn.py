"""Level 4 CNN — 2D Interaction Map + Hierarchical Attention Pooling.

Architecture:
  1. Multi-head projection  (K pairs of Linear projections)
  2. Interaction maps       prot_head_k @ lig_head_k^T → [B, K, seq_p, seq_l]
  3. 2D CNN                 Conv2d layers extract local interaction patterns
  4. Hierarchical pooling   Attention along ligand axis, then protein axis
  5. Classifier             Dropout → Linear(C, 1) → BCEWithLogitsLoss

This level captures **which residues interact with which atoms**, using
CNN to learn spatial patterns in the interaction maps and attention
pooling to selectively aggregate relevant positions.

End-to-end training — no separate MLP stage.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.config import (
    MOLFORMER_DIM,
    PROTEIN_DIMS,
    SUPPORTED_EMBEDDINGS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    split_loader_for_feature_extraction,
)


# ======================================================================
# Model components
# ======================================================================


class _AxisAttentionPool(nn.Module):
    """Learnable-query attention pool along seq dimension.

    Input:  [B, L, D]
    Output: [B, D]

    Uses dot-product attention with a learnable query vector.
    Mask convention: ``True = padding`` (will be ignored).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool [B, L, D] → [B, D]."""
        q = self.query.expand(x.size(0), -1, -1)          # [B, 1, D]
        scores = torch.bmm(q, x.transpose(1, 2)) * self.scale  # [B, 1, L]

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)                # [B, 1, L]
        pooled = torch.bmm(attn, x)                         # [B, 1, D]
        return self.norm(pooled.squeeze(1))                  # [B, D]


class _HierarchicalPool(nn.Module):
    """Pool 2D CNN output [B, C, H, W] → [B, C] via two-stage attention.

    Stage 1 — ligand axis:  for each protein position h, pool across
              W (ligand) to get a single vector → [B, H, C].
    Stage 2 — protein axis: pool across H (protein) → [B, C].

    This lets the model learn:
      (1) which ligand atoms matter for each protein residue
      (2) which protein residues matter overall
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.lig_pool = _AxisAttentionPool(channels)
        self.prot_pool = _AxisAttentionPool(channels)

    def forward(
        self,
        x: torch.Tensor,
        prot_mask: torch.Tensor,
        lig_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : [B, C, H, W]  (CNN feature maps)
        prot_mask : [B, H]  1=valid, 0=padding
        lig_mask  : [B, W]  1=valid, 0=padding
        """
        B, C, H, W = x.shape

        # --- Stage 1: pool along W (ligand) per protein position ------
        # Reshape [B, C, H, W] → [B*H, W, C]
        x_lig = x.permute(0, 2, 3, 1).reshape(B * H, W, C)

        # Expand lig mask: [B, W] → [B*H, W], True=padding
        lig_pad = (lig_mask == 0).unsqueeze(1).expand(-1, H, -1).reshape(B * H, W)

        after_lig = self.lig_pool(x_lig, lig_pad)   # [B*H, C]
        after_lig = after_lig.reshape(B, H, C)        # [B, H, C]

        # --- Stage 2: pool along H (protein) -------------------------
        prot_pad = (prot_mask == 0)                    # [B, H], True=padding
        result = self.prot_pool(after_lig, prot_pad)   # [B, C]

        return result


class InteractionMapCNN(nn.Module):
    """2D CNN on protein–ligand interaction maps.

    Architecture variants:
      v7 (original): K pairs of linear projections (prot_dim→head_dim,
          lig_dim→head_dim) then dot-product → K interaction maps.
      v8 (BAN):      Full Bilinear Attention Network — each head has a
          weight matrix W_k [prot_dim, lig_dim] that computes interaction
          directly in the original embedding spaces without any projection
          bottleneck.  Preserves 100% of ligand information.

    Common pipeline (both variants):
      1. Interaction maps           [B, K, seq_p, seq_l]
      2. 4-layer 2D CNN with dilated convolution
      3. Hierarchical attention pooling
      4. Linear classifier → BCEWithLogitsLoss
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        cnn_channels: int = 64,
        dropout: float = 0.3,
        variant: str = "v7",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.num_heads = num_heads

        if variant == "v7":
            # Original: project both sides to head_dim, then dot-product
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            self.prot_heads = nn.ModuleList([
                nn.Linear(protein_dim, head_dim) for _ in range(num_heads)
            ])
            self.lig_heads = nn.ModuleList([
                nn.Linear(ligand_dim, head_dim) for _ in range(num_heads)
            ])
        elif variant == "v8":
            # Full Bilinear Attention: W_k[prot_dim, lig_dim] per head.
            # score(i,j) = protein[i] @ W_k @ ligand[j]
            # No projection bottleneck — uses full embeddings.
            self.head_dim = 0  # not used in v8
            self.scale = ligand_dim ** -0.5
            self.W_ban = nn.Parameter(
                torch.empty(num_heads, protein_dim, ligand_dim)
            )
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose 'v7' or 'v8'.")

        # 2D CNN on interaction maps [B, num_heads, seq_p, seq_l]
        # Layer 1-2: 3×3 local patterns (receptive field 5×5)
        # Layer 3: dilated 3×3 (effective 5×5, receptive field 9×9)
        # Layer 4: 3×3 consolidation (receptive field 13×13)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_heads, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
        )

        # Hierarchical attention pooling
        self.pool = _HierarchicalPool(cnn_channels)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cnn_channels, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.variant == "v7":
            for heads in (self.prot_heads, self.lig_heads):
                for h in heads:
                    nn.init.xavier_uniform_(h.weight)
                    nn.init.zeros_(h.bias)
        elif self.variant == "v8":
            # Xavier uniform on each head's W_k[prot_dim, lig_dim]
            for k in range(self.num_heads):
                nn.init.xavier_uniform_(self.W_ban[k])
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        protein_matrix : [B, seq_p, protein_dim]
        ligand_matrix  : [B, seq_l, ligand_dim]
        protein_mask   : [B, seq_p]  1=valid, 0=padding
        ligand_mask    : [B, seq_l]  1=valid, 0=padding

        Returns
        -------
        logits : [B, 1]
        """
        # --- Build multi-head interaction maps -----------------------
        if self.variant == "v7":
            maps: list[torch.Tensor] = []
            for ph, lh in zip(self.prot_heads, self.lig_heads):
                p = ph(protein_matrix)                             # [B, seq_p, head_dim]
                l = lh(ligand_matrix)                              # [B, seq_l, head_dim]
                m = torch.bmm(p, l.transpose(1, 2)) * self.scale  # [B, seq_p, seq_l]
                maps.append(m)
            interaction = torch.stack(maps, dim=1)  # [B, K, seq_p, seq_l]

        elif self.variant == "v8":
            # Full Bilinear: protein @ W_k @ ligand^T for each head k.
            # Step 1: project protein into ligand space per head
            #   protein_matrix: [B, sp, prot_dim]
            #   W_ban:          [K, prot_dim, lig_dim]
            #   result:         [B, K, sp, lig_dim]
            p_proj = torch.einsum(
                'bip, kpd -> bkid', protein_matrix, self.W_ban
            )
            # Step 2: dot-product with ligand in full lig_dim space
            #   p_proj:         [B, K, sp, lig_dim]
            #   ligand_matrix:  [B, sl, lig_dim]
            #   result:         [B, K, sp, sl]
            interaction = torch.matmul(
                p_proj,
                ligand_matrix.unsqueeze(1).transpose(-1, -2),
            ) * self.scale

        # Mask padding positions
        mask_2d = protein_mask.unsqueeze(2) * ligand_mask.unsqueeze(1)  # [B, sp, sl]
        interaction = interaction * mask_2d.unsqueeze(1)

        # --- 2D CNN ---------------------------------------------------
        features = self.cnn(interaction)  # [B, cnn_channels, seq_p, seq_l]
        features = features * mask_2d.unsqueeze(1)  # re-apply mask after CNN

        # --- Hierarchical attention pool → classification ------------
        pooled = self.pool(features, protein_mask, ligand_mask)  # [B, C]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, 1]

        return logits


# ======================================================================
# Threshold sweep
# ======================================================================

def _best_mcc_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Return (threshold, MCC) that maximises MCC."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    grid = np.linspace(0.05, 0.95, 37)
    anchors = np.unique(np.clip(y_proba, 0.0, 1.0))
    thresholds = np.unique(np.concatenate([grid, anchors]))

    best_thr, best_mcc = 0.5, -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, best_mcc


# ======================================================================
# Training loop
# ======================================================================

def _compute_pos_weight(labels: np.ndarray) -> float:
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(np.clip(n_neg / max(n_pos, 1), 1.0, 20.0))


def _extract_labels(loader: DataLoader) -> np.ndarray:
    """Extract all labels from a DataLoader without loading matrices."""
    ds = loader.dataset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        if hasattr(base, "_df") and "label" in base._df.columns:
            return base._df.iloc[ds.indices]["label"].to_numpy(dtype=np.int64)
    if hasattr(ds, "_df") and "label" in ds._df.columns:
        return ds._df["label"].to_numpy(dtype=np.int64)
    labels = []
    for i in range(len(ds)):
        labels.append(int(ds[i][2]))
    return np.asarray(labels, dtype=np.int64)


def _save_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    epoch: int,
    best_score: float,
    best_state: dict | None,
    no_improve: int,
) -> None:
    """Save a training checkpoint for resuming interrupted runs."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  # type: ignore[union-attr]
        "best_score": best_score,
        "best_state": best_state,
        "no_improve": no_improve,
    }
    if use_amp:
        payload["scaler_state_dict"] = scaler.state_dict()
    # Write to tmp then rename for atomicity
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    tqdm.write(f"    Checkpoint saved at epoch {epoch}: {path}")

def _train_interaction_cnn(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    protein_dim: int,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 50,
    seed: int = 42,
    num_heads: int = 8,
    head_dim: int = 32,
    cnn_channels: int = 64,
    dropout: float = 0.3,
    variant: str = "v7",
    train_to_zero: bool = False,
    train_to_zero_threshold: float = 0.01,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 50,
) -> tuple[InteractionMapCNN, dict]:
    """Train InteractionMapCNN end-to-end.

    Returns the best model and evaluation metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CPU / GPU optimization ---------------------------------------
    if device.type == "cpu":
        n_cpus = os.cpu_count() or 1
        torch.set_num_threads(n_cpus)
        torch.set_num_interop_threads(max(1, n_cpus // 2))
        tqdm.write(f"    CPU mode: using {n_cpus} threads")
    else:
        # GPU: enable TF32 for faster matmuls on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Precision flags ----------------------------------------------
    use_double = os.getenv("BENCHMARK_LEVEL4CNN_DOUBLE", "1") == "1"
    no_amp = os.getenv("BENCHMARK_LEVEL4CNN_NO_AMP", "0") == "1"
    deterministic = os.getenv("BENCHMARK_LEVEL4CNN_DETERMINISTIC", "0") == "1"

    dtype = torch.float64 if use_double else torch.float32
    use_amp = device.type == "cuda" and not no_amp and not use_double
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        tqdm.write("    Deterministic mode: ON (cudnn.benchmark=False)")
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    precision_info = [f"variant={variant}"]
    if use_double:
        precision_info.append("float64")
    if no_amp or use_double:
        precision_info.append("AMP=OFF")
    else:
        precision_info.append("AMP=ON")
    if deterministic:
        precision_info.append("deterministic")
    tqdm.write(f"    Precision: {', '.join(precision_info)}")

    # --- Build model --------------------------------------------------
    model = InteractionMapCNN(
        protein_dim=protein_dim,
        ligand_dim=MOLFORMER_DIM,
        num_heads=num_heads,
        head_dim=head_dim,
        cnn_channels=cnn_channels,
        dropout=dropout,
        variant=variant,
    ).to(device=device, dtype=dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    variant_info = (
        f"head_dim={head_dim}" if variant == "v7"
        else f"BAN W[{num_heads}x{protein_dim}x{MOLFORMER_DIM}]"
    )
    tqdm.write(
        f"    CNN InteractionMap: variant={variant}, heads={num_heads}, "
        f"{variant_info}, cnn_channels={cnn_channels}, dropout={dropout:.2f}\n"
        f"    Trainable params: {trainable:,} / {total:,}"
    )

    # --- Optimiser, loss, scheduler -----------------------------------
    weight_decay = float(os.getenv("BENCHMARK_LEVEL4CNN_WEIGHT_DECAY", "0.02"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_labels = _extract_labels(train_loader)
    pos_weight = _compute_pos_weight(train_labels)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=dtype, device=device),
    )
    tqdm.write(f"    pos_weight={pos_weight:.2f}, lr={lr:.1e}, wd={weight_decay}")
    if train_to_zero:
        tqdm.write(
            f"    *** TRAIN-TO-ZERO mode: early stopping DISABLED, "
            f"training until train_loss & val_loss < {train_to_zero_threshold:.4f} "
            f"(max {epochs} epochs) ***"
        )

    # --- Checkpoint paths ---------------------------------------------
    ckpt_path = os.path.join(checkpoint_dir, "training_checkpoint.pt") if checkpoint_dir else None

    # --- Training loop ------------------------------------------------
    best_score = -float("inf")
    best_state = None
    no_improve = 0
    start_epoch = 1

    # --- Resume from checkpoint if available --------------------------
    if ckpt_path and os.path.exists(ckpt_path):
        tqdm.write(f"    Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_score = ckpt.get("best_score", -float("inf"))
        no_improve = ckpt.get("no_improve", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if ckpt.get("best_state") is not None:
            best_state = ckpt["best_state"]
        tqdm.write(
            f"    Resumed at epoch {start_epoch}, "
            f"best_val_mcc={best_score:.4f}, no_improve={no_improve}"
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            p = batch["protein_matrix"].to(device=device, dtype=dtype)
            l = batch["ligand_matrix"].to(device=device, dtype=dtype)
            pm = batch["protein_mask"].to(device)
            lm = batch["ligand_mask"].to(device)
            y = batch["label"].to(device=device, dtype=dtype).unsqueeze(1)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(p, l, pm, lm)
                loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = running_loss / max(n_batches, 1)

        # --- Validation -----------------------------------------------
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_probs: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in val_loader:
                p = batch["protein_matrix"].to(device=device, dtype=dtype)
                l = batch["ligand_matrix"].to(device=device, dtype=dtype)
                pm = batch["protein_mask"].to(device)
                lm = batch["ligand_mask"].to(device)
                y = batch["label"].to(device=device, dtype=dtype).unsqueeze(1)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(p, l, pm, lm)
                    val_loss += criterion(logits, y).item()

                val_n += 1
                val_probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
                val_targets.append(y.cpu().numpy().ravel())

        avg_val = val_loss / max(val_n, 1)
        probs = np.concatenate(val_probs) if val_probs else np.array([])
        targets = np.concatenate(val_targets) if val_targets else np.array([])

        thr, val_mcc = _best_mcc_threshold(targets.astype(int), probs)

        # Early stopping on val_mcc (disabled in train-to-zero mode)
        improved = val_mcc > best_score
        marker = " ★" if improved else ""

        if train_to_zero:
            tqdm.write(
                f"    Epoch {epoch:3d}: loss={avg_train:.6f}, "
                f"val_loss={avg_val:.6f}, val_mcc={val_mcc:.4f}, "
                f"thr={thr:.3f}{marker}"
            )
        else:
            tqdm.write(
                f"    Epoch {epoch:3d}: loss={avg_train:.4f}, "
                f"val_loss={avg_val:.4f}, val_mcc={val_mcc:.4f}, "
                f"thr={thr:.3f} ({no_improve}/{patience}){marker}"
            )
        sys.stdout.flush()
        sys.stderr.flush()

        if improved:
            best_score = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if train_to_zero:
            # In train-to-zero mode: stop only when both losses are below threshold
            if avg_train < train_to_zero_threshold and avg_val < train_to_zero_threshold:
                tqdm.write(
                    f"    ✓ Train-to-zero converged at epoch {epoch}: "
                    f"train_loss={avg_train:.6f}, val_loss={avg_val:.6f} "
                    f"(both < {train_to_zero_threshold})"
                )
                break
        else:
            if no_improve >= patience:
                tqdm.write(
                    f"    Early stopping at epoch {epoch} "
                    f"(best_val_mcc={best_score:.4f})"
                )
                break

        # --- Periodic checkpoint save ---------------------------------
        if ckpt_path and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            assert isinstance(ckpt_path, str)
            _save_training_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler,
                use_amp, epoch, best_score, best_state, no_improve,
            )

    # --- Restore best model -------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    # --- Clean up checkpoint (training complete) -----------------------
    if checkpoint_dir is not None:
        assert isinstance(checkpoint_dir, str)
        final_ckpt = os.path.join(checkpoint_dir, "training_checkpoint.pt")
        if os.path.exists(final_ckpt):
            os.remove(final_ckpt)
            tqdm.write(f"    Removed training checkpoint (training complete)")

    return model, {"best_val_mcc": best_score}


# ======================================================================
# Evaluation
# ======================================================================

@torch.inference_mode()
def _evaluate(
    model: InteractionMapCNN,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = None,
    desc: str = "",
) -> dict:
    """Evaluate model and return full metrics dict.

    If `threshold` is provided (e.g., from val-optimized), use it directly.
    Otherwise, sweep to find the MCC-optimal threshold on this data.

    Returns dict with: mcc, threshold, accuracy, f1, precision, recall, auroc,
    plus raw y_true and y_prob arrays.
    """
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    # Auto-detect model dtype for double precision support
    model_dtype = next(model.parameters()).dtype
    eval_amp = device.type == "cuda" and model_dtype != torch.float64

    for batch in loader:
        p = batch["protein_matrix"].to(device=device, dtype=model_dtype)
        l = batch["ligand_matrix"].to(device=device, dtype=model_dtype)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)
        y = batch["label"].numpy()

        with torch.amp.autocast(device_type=device.type, enabled=eval_amp):
            logits = model(p, l, pm, lm)

        probs = torch.sigmoid(logits.float()).cpu().numpy().ravel()
        all_probs.append(probs)
        all_targets.append(y)

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)

    if threshold is None:
        thr, mcc = _best_mcc_threshold(targets, probs)
    else:
        thr = threshold
        preds_for_mcc = (probs >= thr).astype(int)
        mcc = float(matthews_corrcoef(targets, preds_for_mcc))

    preds = (probs >= thr).astype(int)
    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    prec = float(precision_score(targets, preds, zero_division=0))
    rec = float(recall_score(targets, preds, zero_division=0))
    try:
        auroc = float(roc_auc_score(targets, probs))
    except ValueError:
        auroc = 0.0

    if desc:
        tqdm.write(
            f"    {desc}: MCC={mcc:.4f}, AUROC={auroc:.4f}, "
            f"F1={f1:.4f}, thr={thr:.3f}, acc={acc:.4f}"
        )

    return {
        "mcc": mcc,
        "threshold": thr,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auroc": auroc,
        "y_true": targets,
        "y_prob": probs,
    }


# ======================================================================
# Runner
# ======================================================================

def _load_frozen_selection(output_dir: str, cache_filename: str) -> dict | None:
    """Load frozen selection from train artifact for test mode."""
    test_token = f"{os.sep}test{os.sep}"
    train_token = f"{os.sep}train{os.sep}"
    if test_token not in output_dir:
        return None
    train_dir = output_dir.replace(test_token, train_token, 1)
    path = os.path.join(train_dir, cache_filename)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


class Level4CNNRunner(BaseLevelRunner):
    """2D Interaction Map CNN — end-to-end protein–ligand classification."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def knn_is_deterministic(self) -> bool:
        return False

    @property
    def level_tag(self) -> str:
        return "level4_cnn"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level4_cnn_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 4 CNN results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Building Level 4 CNN (seed {seed})...")

        # --- Resolve dimensions from embedding model ------------------
        full_emb = SUPPORTED_EMBEDDINGS.get(self.embedding_name, self.embedding_name)
        protein_dim = PROTEIN_DIMS.get(full_emb, 320)

        # --- Hyperparameters ------------------------------------------
        variant = os.getenv("BENCHMARK_LEVEL4CNN_VARIANT", "v7")
        num_heads = int(os.getenv("BENCHMARK_LEVEL4CNN_NUM_HEADS", "8"))
        head_dim = int(os.getenv("BENCHMARK_LEVEL4CNN_HEAD_DIM", "32"))
        cnn_channels = int(os.getenv("BENCHMARK_LEVEL4CNN_CHANNELS", "64"))
        dropout = float(os.getenv("BENCHMARK_LEVEL4CNN_DROPOUT", "0.3"))
        lr = float(os.getenv("BENCHMARK_LEVEL4CNN_LR", str(self._config.learning_rate)))
        batch_size = int(os.getenv("BENCHMARK_LEVEL4CNN_BATCH_SIZE", str(self._config.batch_size)))

        # --- Build dataloaders ----------------------------------------
        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        # In train mode: use all train data (CNN is end-to-end, no
        # separate feature extraction stage that could leak).
        model_train_loader = train_loader

        # --- Train-to-zero mode ----------------------------------------
        train_to_zero = os.getenv("BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO", "0") == "1"
        train_to_zero_thr = float(os.getenv("BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR", "0.01"))

        # --- Checkpoint frequency --------------------------------------
        checkpoint_every = int(os.getenv("BENCHMARK_LEVEL4CNN_CHECKPOINT_EVERY", "50"))

        # --- Train ----------------------------------------------------
        tqdm.write(f"  Training InteractionMapCNN (variant={variant})...")
        model, train_info = _train_interaction_cnn(
            train_loader=model_train_loader,
            val_loader=val_loader,
            protein_dim=protein_dim,
            lr=lr,
            epochs=self._config.epochs,
            patience=int(os.getenv("BENCHMARK_LEVEL4CNN_PATIENCE", "20")),
            seed=seed,
            num_heads=num_heads,
            head_dim=head_dim,
            cnn_channels=cnn_channels,
            dropout=dropout,
            variant=variant,
            train_to_zero=train_to_zero,
            train_to_zero_threshold=train_to_zero_thr,
            checkpoint_dir=output_dir,
            checkpoint_every=checkpoint_every,
        )

        device = next(model.parameters()).device

        # --- Evaluate -------------------------------------------------
        # Step 1: Evaluate on val to get val-optimized threshold
        val_result = _evaluate(model, val_loader, device, desc="Eval (val)")
        val_threshold = val_result["threshold"]

        if self.mode == "train":
            eval_result = val_result
        else:
            # Step 2: Apply val threshold to test set (fair protocol)
            eval_result = _evaluate(
                model, test_loader, device,
                threshold=val_threshold, desc="Eval (test)",
            )

        # --- Save results in standard format --------------------------
        sc_key = "Split by Scaffold"
        cnn_metrics = {
            "mcc": round(eval_result["mcc"], 6),
            "threshold": round(eval_result["threshold"], 4),
            "accuracy": round(eval_result["accuracy"], 6),
            "f1": round(eval_result["f1"], 6),
            "precision": round(eval_result["precision"], 6),
            "recall": round(eval_result["recall"], 6),
            "auc": round(eval_result["auroc"], 6),
            "best_val_mcc": round(train_info["best_val_mcc"], 6),
            "val_threshold": round(val_threshold, 4),
        }
        result = {sc_key: {"MLP": cnn_metrics}}

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, "level4_cnn_model.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Save raw predictions for reproducibility
        np.savez(
            os.path.join(output_dir, "raw_predictions.npz"),
            y_true=eval_result["y_true"],
            y_prob=eval_result["y_prob"],
        )

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 4 CNN (seed {seed}): MCC={eval_result['mcc']:.4f}, "
            f"AUROC={eval_result['auroc']:.4f}, F1={eval_result['f1']:.4f}"
        )
        return result
