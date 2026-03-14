"""Level 1c — Ligand embedding + attention pooling + KNN/MLP.

Uses the **same** MoLFormer per-token ligand matrices as Levels 2–3,
but applies a **learned attention pooling** mechanism to produce a
fixed-size vector — **ligand only, no protein**.

Architecture:
  1. Linear projection  (768 → hidden_dim)
  2. Attention pooling   (learnable query, multi-head attention)
  3. Output              [B, hidden_dim]

This level isolates the contribution of *learned aggregation* on
compound embeddings.  Comparing 1b vs 1c shows whether attention
pooling improves over simple averaging on compound representations
alone, before any protein information is introduced.

Comparison axes:
  - **1b vs 1c**: Mean pool vs attention pool (both ligand-only)
  - **1c vs 3**: Ligand-only attn pool vs protein+ligand attn pool

Training protocol (consistent with all levels):
    - Attention pooling model trained on the **training** split
        (validation split for early stopping).
  - Training hyperparameters (epochs, patience, learning rate) are
    taken from the CLI / ``BenchmarkConfig`` — the same values that
        control Level 3.
    - In ``train`` mode: fit on held-out train features, evaluate on val features.
    - In ``test`` mode: fit on val features, evaluate on test features.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all levels.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import BenchmarkConfig
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    split_loader_for_feature_extraction,
)
from benchmark.levels.protocol import sanitize_features
from benchmark.levels.se3_features import SE3FeatureLoader, build_se3_loader
from benchmark.runtime import (
    batch_size_fallbacks,
    is_cuda_oom_error,
    resolve_effective_batch_size,
)


# ---------------------------------------------------------------------------
# Ligand-only attention pooling model
# ---------------------------------------------------------------------------

class _LigandAttentionPoolingModel(nn.Module):
    """Projection + attention pooling for ligand matrices only.

    Mirrors Level 3's ``_AttentionPoolingModel`` but processes
    only the ligand modality — no protein branch.
    """

    def __init__(
        self,
        ligand_input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ligand_pool = _AttentionPool(hidden_dim, num_heads, dropout)

        # Init
        nn.init.xavier_uniform_(self.ligand_proj[0].weight)
        nn.init.zeros_(self.ligand_proj[0].bias)

    def forward(
        self,
        ligand_matrix: torch.Tensor,
        ligand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return attention-pooled ligand vector [B, hidden_dim]."""
        l_attn = (ligand_mask == 0) if ligand_mask is not None else None
        ligand = self.ligand_proj(ligand_matrix)
        return self.ligand_pool(ligand, l_attn)


class _AttentionPool(nn.Module):
    """Learnable-query attention pooling (Set Transformer style).

    Identical to the one used by Level 3 — duplicated here so that
    Level 1c is self-contained.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool ``[B, seq_len, dim]`` → ``[B, dim]``."""
        query = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attention(query=query, key=x, value=x, key_padding_mask=mask)
        return self.norm(pooled).squeeze(1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_ligand_attention_pooling(
    train_loader: DataLoader,
    val_loader: DataLoader,
    ligand_dim: int = 768,
    hidden_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.2,
    lr: float = 1e-4,
    epochs: int = 500,
    patience: int | None = 5,
    seed: int = 42,
    log_prefix: str = "",
) -> _LigandAttentionPoolingModel:
    """Train ligand-only projection + attention pooling.

    Uses a lightweight BCE objective to learn which tokens matter.
    The actual classification is delegated to KNN/MLP afterwards.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _LigandAttentionPoolingModel(
        ligand_input_dim=ligand_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    # Auxiliary classification head (discarded after training)
    aux_head = nn.Linear(hidden_dim, 1).to(device)
    nn.init.xavier_uniform_(aux_head.weight)
    nn.init.zeros_(aux_head.bias)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_mcc = float("-inf")
    best_state = None
    wait = 0

    start_time = time.time()
    log_every = max(1, epochs // 20)
    tqdm.write(
        f"{log_prefix} Training start: epochs={epochs}, patience={patience}, "
        f"device={device}, ligand_dim={ligand_dim}, log_every={log_every}"
    )

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"{log_prefix} epochs",
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        # --- train ---
        model.train()
        aux_head.train()
        train_loss = 0.0
        n_batches = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for batch in train_loader:
            l = batch["ligand_matrix"].to(device)
            lm = batch["ligand_mask"].to(device)
            y = batch["label"].to(device).unsqueeze(1)

            features = model(l, lm)
            logits = aux_head(features)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            y_true = y.detach().cpu().numpy().astype(int).ravel()
            y_hat = (torch.sigmoid(logits).detach().cpu().numpy().ravel() >= 0.5).astype(int)
            train_true.append(y_true)
            train_pred.append(y_hat)

        scheduler.step()

        # --- validate ---
        model.eval()
        aux_head.eval()
        val_loss = 0.0
        val_n = 0
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                l = batch["ligand_matrix"].to(device)
                lm = batch["ligand_mask"].to(device)
                y = batch["label"].to(device).unsqueeze(1)

                features = model(l, lm)
                logits = aux_head(features)
                val_loss += criterion(logits, y).item()
                val_n += 1

                y_true = y.detach().cpu().numpy().astype(int).ravel()
                y_hat = (torch.sigmoid(logits).detach().cpu().numpy().ravel() >= 0.5).astype(int)
                val_true.append(y_true)
                val_pred.append(y_hat)

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)
        train_mcc = _safe_mcc(train_true, train_pred)
        val_mcc = _safe_mcc(val_true, val_pred)

        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / max(epoch, 1)) * max(epochs - epoch, 0)
            progress = 100.0 * epoch / max(epochs, 1)
            best_mcc_so_far = max(best_val_mcc, val_mcc)
            epoch_bar.set_postfix_str(
                f"train_mcc={train_mcc:.4f} val_mcc={val_mcc:.4f} best_mcc={best_mcc_so_far:.4f} wait={wait} eta={eta/60:.1f}m"
            )
            tqdm.write(
                f"{log_prefix} Epoch {epoch}/{epochs} ({progress:5.1f}%) "
                f"train_mcc={train_mcc:.4f} val_mcc={val_mcc:.4f} "
                f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} best_val_mcc={best_mcc_so_far:.4f} "
                f"wait={wait} eta={eta/60:.1f}m"
            )

        improved_mcc = val_mcc > best_val_mcc
        tie_better_loss = np.isclose(val_mcc, best_val_mcc) and avg_val < best_val_loss
        if improved_mcc or tie_better_loss:
            best_val_mcc = val_mcc
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                tqdm.write(
                    f"{log_prefix} Early stopping at epoch {epoch}/{epochs} "
                    f"(val_mcc={val_mcc:.4f}, best_mcc={best_val_mcc:.4f})"
                )
                break

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)
    model.eval()
    return model


def _safe_mcc(y_true_parts: list[np.ndarray], y_pred_parts: list[np.ndarray]) -> float:
    """Compute MCC safely for potentially single-class batches."""
    if not y_true_parts or not y_pred_parts:
        return 0.0
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(
    model: _LigandAttentionPoolingModel,
    loader: DataLoader,
    device: torch.device,
    se3_loader: SE3FeatureLoader | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention-pooled ligand-only features."""
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    se3_logged = False

    for batch in loader:
        l = batch["ligand_matrix"].to(device)
        lm = batch["ligand_mask"].to(device)

        features = model(l, lm).cpu().numpy()
        if se3_loader is not None:
            se3_batch = se3_loader.get_batch(batch["chembl_id"])
            if se3_batch.shape[1] > 0:
                if not se3_logged:
                    sem_dim = int(features.shape[1])
                    se3_dim = int(se3_batch.shape[1])
                    fused_dim = sem_dim + se3_dim
                    tqdm.write(
                        f"  [SE3] Active fusion (Level 1c): semantic_dim={sem_dim}, se3_dim={se3_dim}, fused_ligand_dim={fused_dim}"
                    )
                    se3_logged = True
                features = np.concatenate([features, se3_batch], axis=-1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(features)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Level 1c runner
# ---------------------------------------------------------------------------

class Level1cRunner(BaseLevelRunner):
    """Ligand embedding -> attention pooling -> KNN/MLP (no protein)."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def knn_is_deterministic(self) -> bool:
        return False  # Learned feature extractor → KNN input varies per seed

    @property
    def uses_gpu(self) -> bool:
        return True

    @property
    def level_tag(self) -> str:
        return "level1c_ligattn"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Train ligand attention pooling, extract features, run KNN/MLP."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level1c_knn_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 1c results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Building Level 1c ligand attention pooling (seed {seed})...")

        se3_loader = build_se3_loader(
            feature_dir=self._config.se3_features_dir,
            use_se3_ligand=self._config.use_se3_ligand,
            default_feature_dirs=self._config.resolved_se3_feature_dirs,
        )
        if self._config.use_se3_ligand and se3_loader is None:
            tqdm.write("  WARNING: SE3 ligand fusion requested, but no valid .npy structural vectors were found. Proceeding with semantic ligand features only.")
        elif se3_loader is not None:
            tqdm.write(
                f"  [SE3] Structural vectors loaded: dim={se3_loader.dim}. Ligand concatenation enabled."
            )

        log_prefix = f"[L1c|seed {seed}]"
        start_batch_size = resolve_effective_batch_size(self._config.batch_size)
        candidates = batch_size_fallbacks(start_batch_size)
        last_oom: RuntimeError | None = None

        for attempt_idx, batch_size in enumerate(candidates, start=1):
            try:
                tqdm.write(
                    f"{log_prefix} Attempt {attempt_idx}/{len(candidates)} with batch_size={batch_size}"
                )

                train_loader, val_loader, test_loader = build_matrix_dataloaders(
                    dataset_type=self.dataset,
                    embedding_name=self.embedding_name,
                    scaffold_split_dir=self.scaffold_split_dir,
                    batch_size=batch_size,
                    dataset_source_filter=self._config.dataset_source_filter,
                    mode=self.mode,
                    ligand_model=self._config.ligand_model,
                )

                tqdm.write(
                    f"  Dataloaders: train={len(train_loader)} batches ({len(train_loader.dataset)} samples), "
                    f"val={len(val_loader)} batches ({len(val_loader.dataset)} samples)"
                )

                # In train mode, split training data to avoid train-set optimism:
                # model trains on 80% of train, KNN features from held-out 20%.
                feat_extract_loader = None
                if self.mode == "train":
                    model_train_loader, feat_extract_loader = split_loader_for_feature_extraction(
                        train_loader, seed=seed,
                    )
                else:
                    model_train_loader = train_loader

                tqdm.write(f"{log_prefix} Training ligand projection + attention pooling...")
                model = _train_ligand_attention_pooling(
                    train_loader=model_train_loader,
                    val_loader=val_loader,
                    ligand_dim=self._config.ligand_dim,
                    lr=self._config.learning_rate,
                    epochs=self._config.epochs,
                    patience=self._config.resolved_patience,
                    seed=seed,
                    log_prefix=log_prefix,
                )

                device = next(model.parameters()).device

                if self.mode == "train":
                    tqdm.write("  Extracting ligand attention-pooled features (held-out train + val)...")
                    x_fit, y_fit = _extract_features(model, feat_extract_loader, device, se3_loader=se3_loader)
                    x_eval, y_eval = _extract_features(model, val_loader, device, se3_loader=se3_loader)
                else:
                    tqdm.write("  Extracting ligand attention-pooled features (val + test)...")
                    x_fit, y_fit = _extract_features(model, val_loader, device, se3_loader=se3_loader)
                    x_eval, y_eval = _extract_features(model, test_loader, device, se3_loader=se3_loader)

                # Sanitize
                for name, arr in [("fit", x_fit), ("eval", x_eval)]:
                    arr_sanitized, bad = sanitize_features(arr)
                    if bad:
                        tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                        arr[:] = arr_sanitized

                # Train canonical KNN/MLP on fit features
                tqdm.write("  Training KNN + MLP (canonical classifiers)...")
                models = train_knn_mlp(x_fit, y_fit, x_eval, y_eval, seed)
                break
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc):
                    raise
                last_oom = exc
                if batch_size == 1:
                    raise RuntimeError(
                        f"{log_prefix} CUDA OOM even with batch_size=1"
                    ) from exc
                next_batch = max(1, batch_size // 2)
                tqdm.write(
                    f"{log_prefix} CUDA OOM with batch_size={batch_size}; retrying with {next_batch}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            raise RuntimeError(
                f"{log_prefix} Failed all batch-size attempts due to CUDA OOM"
            ) from last_oom

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, "level1c_ligattn_model.pt")
        torch.save(model.state_dict(), checkpoint_path)

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 1c (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result
