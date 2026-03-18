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
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
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

    Creates K interaction maps from multi-head projections, processes
    them with a 2D CNN, and pools with hierarchical attention.
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        num_heads: int = 8,
        head_dim: int = 32,
        cnn_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Multi-head projections for interaction maps
        self.prot_heads = nn.ModuleList([
            nn.Linear(protein_dim, head_dim) for _ in range(num_heads)
        ])
        self.lig_heads = nn.ModuleList([
            nn.Linear(ligand_dim, head_dim) for _ in range(num_heads)
        ])

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
        for heads in (self.prot_heads, self.lig_heads):
            for h in heads:
                nn.init.xavier_uniform_(h.weight)
                nn.init.zeros_(h.bias)
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
        maps: list[torch.Tensor] = []
        for ph, lh in zip(self.prot_heads, self.lig_heads):
            p = ph(protein_matrix)                             # [B, seq_p, head_dim]
            l = lh(ligand_matrix)                              # [B, seq_l, head_dim]
            m = torch.bmm(p, l.transpose(1, 2)) * self.scale  # [B, seq_p, seq_l]
            maps.append(m)

        interaction = torch.stack(maps, dim=1)  # [B, K, seq_p, seq_l]

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
) -> tuple[InteractionMapCNN, dict]:
    """Train InteractionMapCNN end-to-end.

    Returns the best model and evaluation metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Build model --------------------------------------------------
    model = InteractionMapCNN(
        protein_dim=protein_dim,
        ligand_dim=MOLFORMER_DIM,
        num_heads=num_heads,
        head_dim=head_dim,
        cnn_channels=cnn_channels,
        dropout=dropout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tqdm.write(
        f"    CNN InteractionMap: heads={num_heads}, head_dim={head_dim}, "
        f"cnn_channels={cnn_channels}, dropout={dropout:.2f}\n"
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
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )
    tqdm.write(f"    pos_weight={pos_weight:.2f}, lr={lr:.1e}, wd={weight_decay}")

    # --- Training loop ------------------------------------------------
    best_score = -float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            p = batch["protein_matrix"].to(device)
            l = batch["ligand_matrix"].to(device)
            pm = batch["protein_mask"].to(device)
            lm = batch["ligand_mask"].to(device)
            y = batch["label"].to(device).unsqueeze(1)

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
                p = batch["protein_matrix"].to(device)
                l = batch["ligand_matrix"].to(device)
                pm = batch["protein_mask"].to(device)
                lm = batch["ligand_mask"].to(device)
                y = batch["label"].to(device).unsqueeze(1)

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

        # Early stopping on val_mcc
        improved = val_mcc > best_score
        marker = " ★" if improved else ""

        tqdm.write(
            f"    Epoch {epoch:3d}: loss={avg_train:.4f}, "
            f"val_loss={avg_val:.4f}, val_mcc={val_mcc:.4f}, "
            f"thr={thr:.3f} ({no_improve}/{patience}){marker}"
        )

        if improved:
            best_score = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(
                    f"    Early stopping at epoch {epoch} "
                    f"(best_val_mcc={best_score:.4f})"
                )
                break

    # --- Restore best model -------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    return model, {"best_val_mcc": best_score}


# ======================================================================
# Evaluation
# ======================================================================

@torch.inference_mode()
def _evaluate(
    model: InteractionMapCNN,
    loader: DataLoader,
    device: torch.device,
    desc: str = "",
) -> tuple[float, float, float]:
    """Evaluate model and return (mcc, threshold, accuracy)."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        p = batch["protein_matrix"].to(device)
        l = batch["ligand_matrix"].to(device)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)
        y = batch["label"].numpy()

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(p, l, pm, lm)

        probs = torch.sigmoid(logits.float()).cpu().numpy().ravel()
        all_probs.append(probs)
        all_targets.append(y)

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    thr, mcc = _best_mcc_threshold(targets, probs)
    preds = (probs >= thr).astype(int)
    acc = float((preds == targets).mean())

    if desc:
        tqdm.write(f"    {desc}: MCC={mcc:.4f}, thr={thr:.3f}, acc={acc:.4f}")

    return mcc, thr, acc


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
        num_heads = int(os.getenv("BENCHMARK_LEVEL4CNN_NUM_HEADS", "8"))
        head_dim = int(os.getenv("BENCHMARK_LEVEL4CNN_HEAD_DIM", "32"))
        cnn_channels = int(os.getenv("BENCHMARK_LEVEL4CNN_CHANNELS", "64"))
        dropout = float(os.getenv("BENCHMARK_LEVEL4CNN_DROPOUT", "0.3"))
        lr = float(os.getenv("BENCHMARK_LEVEL4CNN_LR", str(self._config.learning_rate)))

        # --- Build dataloaders ----------------------------------------
        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=self._config.batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        # In train mode: use all train data (CNN is end-to-end, no
        # separate feature extraction stage that could leak).
        model_train_loader = train_loader

        # --- Train ----------------------------------------------------
        tqdm.write("  Training InteractionMapCNN...")
        model, train_info = _train_interaction_cnn(
            train_loader=model_train_loader,
            val_loader=val_loader,
            protein_dim=protein_dim,
            lr=lr,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 50,
            seed=seed,
            num_heads=num_heads,
            head_dim=head_dim,
            cnn_channels=cnn_channels,
            dropout=dropout,
        )

        device = next(model.parameters()).device

        # --- Evaluate -------------------------------------------------
        if self.mode == "train":
            mcc, thr, acc = _evaluate(model, val_loader, device, "Eval (val)")
        else:
            mcc, thr, acc = _evaluate(model, test_loader, device, "Eval (test)")

        # --- Save results in standard format --------------------------
        sc_key = "Split by Scaffold"
        cnn_metrics = {
            "mcc": round(mcc, 6),
            "threshold": round(thr, 4),
            "accuracy": round(acc, 6),
            "best_val_mcc": round(train_info["best_val_mcc"], 6),
        }
        result = {sc_key: {"MLP": cnn_metrics}}

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, "level4_cnn_model.pt")
        torch.save(model.state_dict(), ckpt_path)

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(f"  Level 4 CNN (seed {seed}): MCC={mcc:.4f}")
        return result
