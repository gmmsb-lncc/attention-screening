"""Level 3 — Embedding matrices + attention pooling + KNN/MLP.

Loads the **same** per-residue ESM-2 protein matrices and per-token
MoLFormer ligand matrices used by Levels 2 and 4, but applies a
**learned attention pooling** mechanism to produce fixed-size vectors.

Architecture (identical to Level 4 minus cross-attention):
  1. Linear projection  (protein_dim → hidden_dim / ligand_dim → hidden_dim)
  2. Attention pooling   (learnable query, multi-head attention, [B, hidden_dim])
  3. Concatenation       [protein_vec ‖ ligand_vec]  →  [B, 2 × hidden_dim]

Difference from other levels:
  - Level 2: mean pooling (no parameters)
  - **Level 3: projection + attention pooling (light trainable parameters)**
  - Level 4: projection + cross-attention + attention pooling (full model)

This level isolates the contribution of *learned aggregation* without
the inter-modality interaction that cross-attention provides.

Training protocol (consistent with all levels):
  - Attention pooling model trained on the **training** split
    (validation split for early stopping).
  - Training hyperparameters (epochs, patience, learning rate) are
    taken from the CLI / ``BenchmarkConfig`` — the same values that
    control Level 4 — so both learned levels train under identical
    budgets.
  - Features extracted from the **validation** split — the model was
    *not* directly trained on val, only used it for model selection,
    so val features are free of train-set optimism.
  - KNN/MLP classifiers trained on val features.
  - Evaluation on the hold-out **test** split.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all four levels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp, train_mlp_only
from benchmark.config import (
    MOLFORMER_DIM,
    PROTEIN_DIMS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    split_loader_for_feature_extraction,
)


def _load_frozen_mlp_selection_from_train(
    output_dir: str,
    cache_filename: str,
) -> dict[str, object] | None:
    """Load frozen MLP selection from corresponding train artifact for same seed."""
    test_token = f"{os.sep}test{os.sep}"
    train_token = f"{os.sep}train{os.sep}"
    if test_token not in output_dir:
        return None

    train_seed_dir = output_dir.replace(test_token, train_token, 1)
    train_cache_path = os.path.join(train_seed_dir, cache_filename)
    if not os.path.exists(train_cache_path):
        return None

    with open(train_cache_path) as fh:
        payload = json.load(fh)
    scaffold_key = next(iter(payload.keys()), None)
    if not scaffold_key:
        return None
    mlp_block = payload.get(scaffold_key, {}).get("MLP", {})
    selection = mlp_block.get("mlp_selection")
    return selection if isinstance(selection, dict) else None


def _extract_binary_labels_from_loader(loader: DataLoader) -> np.ndarray:
    """Extract labels from loader dataset without materializing matrices when possible."""
    ds = loader.dataset

    # Subset(MatrixDataset)
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        if hasattr(base, "_df") and "label" in base._df.columns:
            return base._df.iloc[ds.indices]["label"].to_numpy(dtype=np.int64)

    # MatrixDataset fast path
    if hasattr(ds, "_df") and "label" in ds._df.columns:
        return ds._df["label"].to_numpy(dtype=np.int64)

    # Fallback (may touch dataset items)
    labels = []
    for i in range(len(ds)):
        item = ds[i]
        labels.append(int(item[2]))
    return np.asarray(labels, dtype=np.int64)


def _compute_pos_weight(labels: np.ndarray) -> float:
    """Compute BCE pos_weight = N_negative / N_positive with clipping."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(np.clip(n_neg / max(n_pos, 1), 1.0, 20.0))


def _best_mcc_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Return threshold that maximizes MCC on a validation vector."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    grid = np.linspace(0.05, 0.95, 37)
    anchors = np.unique(np.clip(y_proba, 0.0, 1.0))
    thresholds = np.unique(np.concatenate([grid, anchors]))

    best_thr = 0.5
    best_mcc = -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        mcc = float(matthews_corrcoef(y_true, pred))
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, best_mcc


# ---------------------------------------------------------------------------
# Attention-pooling model (no cross-attention — isolation experiment)
# ---------------------------------------------------------------------------

class _AttentionPoolingModel(nn.Module):
    """Projection + attention pooling for protein–ligand pairs.

    This is a strict subset of Level5LiteModel: same encoders and
    attention pooling, but **no cross-attention layers**.  In forward
    mode it only returns the concatenated feature vector, not logits.
    """

    def __init__(
        self,
        protein_input_dim: int,
        ligand_input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        interaction_features: bool = True,
        multilayer_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.interaction_features = interaction_features
        self.multilayer_layers = multilayer_layers or []

        # Dropout applied to the final concatenated representation to
        # reduce overfitting when downstream classifiers are trained on
        # the extracted features.
        self.feature_dropout = nn.Dropout(dropout)

        # Projection layers (identical to Level5LiteModel encoders)
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention pooling (identical to Level5LiteModel pools)
        self.protein_pool = _AttentionPool(hidden_dim, num_heads, dropout)
        self.ligand_pool = _AttentionPool(hidden_dim, num_heads, dropout)

        # Per-layer projections and pools for multi-layer MoLFormer.
        # Each intermediate layer gets its own dedicated projection
        # (Linear → LN → GELU → Drop) and attention pool, trained
        # jointly so gradients optimise each projector for its layer.
        if self.multilayer_layers:
            self.ml_projectors = nn.ModuleDict({
                f"layer_{l}": nn.Sequential(
                    nn.Linear(ligand_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for l in self.multilayer_layers
            })
            self.ml_pools = nn.ModuleDict({
                f"layer_{l}": _AttentionPool(hidden_dim, num_heads, dropout)
                for l in self.multilayer_layers
            })

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.protein_proj, self.ligand_proj):
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)
        if self.multilayer_layers:
            for proj in self.ml_projectors.values():
                nn.init.xavier_uniform_(proj[0].weight)
                nn.init.zeros_(proj[0].bias)

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
        ligand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return concatenated [protein_vec ‖ ligand_vec].

        During training the aux_head expects ``2 × hidden_dim`` input,
        so this method always returns the plain concatenation.  For
        downstream feature extraction use :meth:`forward_with_interactions`
        which appends element-wise product and absolute difference.

        Parameters
        ----------
        protein_matrix : Tensor [B, prot_len, protein_input_dim]
        ligand_matrix  : Tensor [B, lig_len, ligand_input_dim]
        protein_mask   : Tensor [B, prot_len] — 1 = real, 0 = padding
        ligand_mask    : Tensor [B, lig_len]  — 1 = real, 0 = padding

        Returns
        -------
        Tensor [B, 2 × hidden_dim]
        """
        # Invert masks for PyTorch attention (True = padding)
        p_attn = (protein_mask == 0) if protein_mask is not None else None
        l_attn = (ligand_mask == 0) if ligand_mask is not None else None

        # Project
        protein = self.protein_proj(protein_matrix)   # [B, prot_len, hidden]
        ligand = self.ligand_proj(ligand_matrix)       # [B, lig_len, hidden]

        # Attention pool
        protein_vec = self.protein_pool(protein, p_attn)  # [B, hidden]
        ligand_vec = self.ligand_pool(ligand, l_attn)      # [B, hidden]

        combined = torch.cat([protein_vec, ligand_vec], dim=-1)  # [B, 2*hidden]
        return self.feature_dropout(combined)

    def forward_with_interactions(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
        ligand_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return enriched feature vector with interaction terms.

        Output: ``[prot_vec ‖ lig_vec ‖ prot*lig ‖ |prot−lig| ‖ cos(prot,lig)]``
        giving ``4 × hidden_dim + 1`` dimensions when interaction features
        are enabled, otherwise falls back to ``2 × hidden_dim``.
        """
        p_attn = (protein_mask == 0) if protein_mask is not None else None
        l_attn = (ligand_mask == 0) if ligand_mask is not None else None

        protein = self.protein_proj(protein_matrix)
        ligand = self.ligand_proj(ligand_matrix)

        protein_vec = self.protein_pool(protein, p_attn)
        ligand_vec = self.ligand_pool(ligand, l_attn)

        parts = [protein_vec, ligand_vec]
        if self.interaction_features:
            parts.append(protein_vec * ligand_vec)              # co-activation
            parts.append(torch.abs(protein_vec - ligand_vec))   # complementarity
            # Cosine similarity — global directional compatibility [B, 1]
            cos_sim = torch.nn.functional.cosine_similarity(
                protein_vec, ligand_vec, dim=-1,
            ).unsqueeze(-1)
            parts.append(cos_sim)

        combined = torch.cat(parts, dim=-1)
        return self.feature_dropout(combined)

    def forward_multilayer(
        self,
        ml_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Project and attention-pool multi-layer tensors.

        Each layer is processed by its **dedicated** projector and pool,
        producing a vector of ``hidden_dim`` per layer.  Vectors are
        concatenated, giving ``len(multilayer_layers) * hidden_dim``.

        Parameters
        ----------
        ml_tensors : dict
            Mapping ``"layer_{i}"`` → Tensor ``[B, seq_len, 768]``.
            Tensors should already be on the correct device.

        Returns
        -------
        Tensor ``[B, K * hidden_dim]`` or ``None`` if no layers matched.
        """
        vecs: list[torch.Tensor] = []
        for l in self.multilayer_layers:
            key = f"layer_{l}"
            if key in ml_tensors and key in self.ml_projectors:
                projected = self.ml_projectors[key](ml_tensors[key])  # [B, seq, hidden]
                pooled = self.ml_pools[key](projected, mask=None)     # [B, hidden]
                vecs.append(pooled)
        if vecs:
            return torch.cat(vecs, dim=-1)
        return None


class _AttentionPool(nn.Module):
    """Learnable-query attention pooling (Set Transformer style).

    Identical to ``AttentionPooling`` in
    ``crossattention_split_analysis.models.level5_lite.attention``.
    Duplicated here so that Level 3 is self-contained and does not
    depend on Level 4's model internals.
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
# Training loop (very lightweight — only pooling weights to learn)
# ---------------------------------------------------------------------------

# Optimal hidden_dim per ESM-2 model size.  Determined empirically:
#   8M  (d_P=320)  → 256  — small encoder, compact representation suffices
#   150M (d_P=640) → 384  — matches Level 4 capacity for fair comparison
#   650M (d_P=1280)→ 432  — 1280//3 rounded to multiple of 12 heads
_HIDDEN_DIM_BY_PROTEIN_DIM: dict[int, int] = {
    320:  256,   # ESM-2 8M
    640:  384,   # ESM-2 150M
    1280: 432,   # ESM-2 650M
}


def _train_attention_pooling(
    train_loader: DataLoader,
    val_loader: DataLoader,
    downstream_fit_loader: DataLoader | None,
    protein_dim: int,
    hidden_dim: int | None = None,
    num_heads: int = 8,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 10,
    seed: int = 42,
    model_selection_metric: str = "val_loss",
    multilayer_layers: list[int] | None = None,
    multilayer_dirs: list[Path] | None = None,
) -> tuple[_AttentionPoolingModel, nn.Module]:
    """Train the projection + attention pooling model.

    Uses a lightweight binary cross-entropy objective just to learn
    *which tokens matter* — the actual classification is delegated to
    the canonical KNN/MLP afterwards.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Resolve hyperparameters: env var > argument > auto-scale --------

    # hidden_dim
    if hidden_dim is None:
        env_hd = os.getenv("BENCHMARK_LEVEL3_HIDDEN_DIM", "").strip()
        if env_hd:
            hidden_dim = int(env_hd)
        elif protein_dim in _HIDDEN_DIM_BY_PROTEIN_DIM:
            hidden_dim = _HIDDEN_DIM_BY_PROTEIN_DIM[protein_dim]
        else:
            raw = max(256, protein_dim // 3)
            hidden_dim = ((raw + num_heads - 1) // num_heads) * num_heads

    # dropout (env var overrides function argument)
    env_do = os.getenv("BENCHMARK_LEVEL3_DROPOUT", "").strip()
    if env_do:
        dropout = float(env_do)

    # learning rate (env var overrides function argument)
    env_lr = os.getenv("BENCHMARK_LEVEL3_LR", "").strip()
    if env_lr:
        lr = float(env_lr)

    # weight decay (env var overrides hardcoded default)
    weight_decay = 0.02
    env_wd = os.getenv("BENCHMARK_LEVEL3_WEIGHT_DECAY", "").strip()
    if env_wd:
        weight_decay = float(env_wd)

    # Whether the aux_head should receive interaction features (product
    # and abs-diff) during training.  When enabled the training signal
    # is aligned with the features actually extracted for the downstream
    # MLP, potentially improving representation quality at the cost of a
    # slightly larger aux_head.  Default: "1" (enabled).
    aux_interactions = os.getenv(
        "BENCHMARK_LEVEL3_AUX_INTERACTIONS", "1"
    ).strip().lower() not in {"0", "false", "no"}

    tqdm.write(
        f"    L3 config: hidden_dim={hidden_dim}, dropout={dropout:.2f}, "
        f"lr={lr:.1e}, weight_decay={weight_decay}, protein_dim={protein_dim}, "
        f"aux_interactions={aux_interactions}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    torch.backends.cudnn.benchmark = True

    use_interaction = os.getenv("BENCHMARK_LEVEL3_INTERACTION_FEATURES", "1") == "1"
    ml_layers = multilayer_layers or []
    ml_dirs = multilayer_dirs or []
    model = _AttentionPoolingModel(
        protein_input_dim=protein_dim,
        ligand_input_dim=MOLFORMER_DIM,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        interaction_features=use_interaction,
        multilayer_layers=ml_layers,
    ).to(device)

    # Auxiliary supervision head used ONLY to shape representations
    # (discarded afterwards — KNN/MLP remain the real classifiers).
    # When aux_interactions is True the head receives the same
    # 4×hidden + 1 (cosine sim) features, aligning training gradients.
    # When multilayer is active, K*hidden extra dims are appended.
    if aux_interactions and use_interaction:
        aux_input_dim = hidden_dim * 4 + 1   # prot|lig|prod|diff|cos
    else:
        aux_input_dim = hidden_dim * 2        # prot|lig
    if ml_layers:
        aux_input_dim += len(ml_layers) * hidden_dim
    aux_head = nn.Sequential(
        nn.Linear(aux_input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    nn.init.xavier_uniform_(aux_head[0].weight)
    nn.init.zeros_(aux_head[0].bias)
    nn.init.xavier_uniform_(aux_head[3].weight)
    nn.init.zeros_(aux_head[3].bias)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_labels = _extract_binary_labels_from_loader(train_loader)
    pos_weight = _compute_pos_weight(train_labels)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )

    best_val_loss = float("inf")
    best_val_mcc = -1.0
    best_selection_score = float("inf") if model_selection_metric == "val_loss" else -1.0
    best_downstream_mcc = -1.0
    best_state = None
    best_aux_state = None
    wait = 0
    eval_every = max(1, int(os.getenv("BENCHMARK_LEVEL3_DOWNSTREAM_EVAL_EVERY", "10")))

    epoch_iter = tqdm(
        range(1, epochs + 1),
        desc=f"    AttnPool seed {seed}",
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
    )

    for epoch in epoch_iter:
        # --- train ---
        model.train()
        aux_head.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            p = batch["protein_matrix"].to(device)
            l = batch["ligand_matrix"].to(device)
            pm = batch["protein_mask"].to(device)
            lm = batch["ligand_mask"].to(device)
            y = batch["label"].to(device).unsqueeze(1)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if aux_interactions and use_interaction:
                    features = model.forward_with_interactions(p, l, pm, lm)
                else:
                    features = model(p, l, pm, lm)
                # Append multi-layer features if configured.
                if ml_layers and ml_dirs:
                    ml_tensors = _load_multilayer_tensors(
                        batch["chembl_id"], ml_dirs, ml_layers, device,
                    )
                    ml_feats = model.forward_multilayer(ml_tensors)
                    if ml_feats is not None:
                        features = torch.cat([features, ml_feats], dim=-1)
                logits = aux_head(features)
                loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- validate ---
        model.eval()
        aux_head.eval()
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
                    if aux_interactions and use_interaction:
                        features = model.forward_with_interactions(p, l, pm, lm)
                    else:
                        features = model(p, l, pm, lm)
                    # Append multi-layer features if configured.
                    if ml_layers and ml_dirs:
                        ml_tensors = _load_multilayer_tensors(
                            batch["chembl_id"], ml_dirs, ml_layers, device,
                        )
                        ml_feats = model.forward_multilayer(ml_tensors)
                        if ml_feats is not None:
                            features = torch.cat([features, ml_feats], dim=-1)
                    logits = aux_head(features)
                    val_loss += criterion(logits, y).item()
                val_n += 1
                val_probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
                val_targets.append(y.cpu().numpy().ravel())

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)
        probs = np.concatenate(val_probs) if val_probs else np.array([], dtype=np.float32)
        targets = np.concatenate(val_targets) if val_targets else np.array([], dtype=np.float32)
        if targets.size > 0 and len(np.unique(targets.astype(int))) > 1:
            preds = (probs >= 0.5).astype(int)
            val_mcc = float(matthews_corrcoef(targets.astype(int), preds))
        else:
            val_mcc = 0.0
        epoch_iter.set_postfix(
            train_loss=f"{avg_train:.4f}",
            val_loss=f"{avg_val:.4f}",
            val_mcc=f"{val_mcc:.3f}",
            best_sel=f"{best_selection_score:.4f}",
            pos_w=f"{pos_weight:.2f}",
        )

        thr_mcc, tuned_val_mcc = _best_mcc_threshold(targets.astype(int), probs)

        downstream_mcc = -1.0
        if model_selection_metric == "downstream_mcc":
            should_eval_downstream = (epoch == 1) or (epoch % eval_every == 0)
            if should_eval_downstream:
                fit_loader = downstream_fit_loader if downstream_fit_loader is not None else train_loader
                x_fit_ds, y_fit_ds = _extract_features(model, fit_loader, device)
                x_val_ds, y_val_ds = _extract_features(model, val_loader, device)
                downstream_mcc = _compute_downstream_mcc_proxy(
                    x_fit=x_fit_ds,
                    y_fit=y_fit_ds,
                    x_val=x_val_ds,
                    y_val=y_val_ds,
                    seed=seed + epoch,
                )
            else:
                downstream_mcc = best_downstream_mcc

        if model_selection_metric == "val_loss":
            current_score = avg_val
            improved = current_score < (best_selection_score - 1e-12)
            if (not improved) and abs(current_score - best_selection_score) <= 1e-12:
                improved = tuned_val_mcc > best_val_mcc
        elif model_selection_metric == "downstream_mcc":
            current_score = downstream_mcc
            improved = current_score > best_selection_score
            if (not improved) and np.isclose(current_score, best_selection_score):
                improved = tuned_val_mcc > best_val_mcc
        else:
            current_score = tuned_val_mcc
            improved = current_score > best_selection_score

        if improved:
            best_selection_score = current_score
            best_val_loss = avg_val
            best_val_mcc = tuned_val_mcc
            if model_selection_metric == "downstream_mcc":
                best_downstream_mcc = downstream_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_aux_state = {k: v.cpu().clone() for k, v in aux_head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                tqdm.write(
                    f"    Early stopping at epoch {epoch} "
                    f"(val_loss={avg_val:.6f}, val_mcc={val_mcc:.4f}, tuned_mcc={tuned_val_mcc:.4f}, "
                    f"downstream_mcc={downstream_mcc:.4f}, "
                    f"best_sel={best_selection_score:.6f}, thr={thr_mcc:.3f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_aux_state is not None:
        aux_head.load_state_dict(best_aux_state)

    model.to(device)
    model.eval()
    aux_head.to(device)
    aux_head.eval()
    return model, aux_head


# ---------------------------------------------------------------------------
# Multi-layer MoLFormer feature extraction helpers
# ---------------------------------------------------------------------------

def _resolve_multilayer_dirs(
    dataset_type: str,
    embedding_name: str,
) -> list[Path]:
    """Return directories that may contain multilayer .npz files."""
    from benchmark.config import EMBEDDING_BASE_PATH
    if dataset_type in ("all",):
        base_paths = [
            "./results/protein_model_benchmark_human_v2",
            "./results/protein_model_benchmark_non_human_v2",
        ]
    else:
        base_paths = [EMBEDDING_BASE_PATH.format(dataset_type=dataset_type)]
    return [
        Path(bp) / embedding_name / "build" / "molformer_multilayer"
        for bp in base_paths
    ]


def _load_multilayer_tensors(
    chembl_ids: Sequence[str],
    multilayer_dirs: list[Path],
    layers: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load multi-layer .npz data as padded tensors for training.

    Returns a dict ``{"layer_4": Tensor[B, max_seq, 768], ...}``
    suitable for passing to ``model.forward_multilayer()``.
    Tensors are padded to the maximum sequence length in the batch.
    """
    # Collect raw matrices per layer per molecule.
    per_layer: dict[str, list[np.ndarray]] = {f"layer_{l}": [] for l in layers}
    for cid in chembl_ids:
        fname = f"{cid}_multilayer.npz"
        loaded = False
        for d in multilayer_dirs:
            path = d / fname
            if path.exists():
                data = np.load(path)
                for l in layers:
                    key = f"layer_{l}"
                    if key in data:
                        per_layer[key].append(data[key].astype(np.float32))
                    else:
                        per_layer[key].append(np.zeros((1, 768), dtype=np.float32))
                loaded = True
                break
        if not loaded:
            for l in layers:
                per_layer[f"layer_{l}"].append(np.zeros((1, 768), dtype=np.float32))

    # Pad to common seq length per layer and convert to tensors.
    result: dict[str, torch.Tensor] = {}
    for key, mats in per_layer.items():
        max_len = max(m.shape[0] for m in mats)
        dim = mats[0].shape[1]
        padded = np.zeros((len(mats), max_len, dim), dtype=np.float32)
        for i, mat in enumerate(mats):
            padded[i, :mat.shape[0], :] = mat
        result[key] = torch.from_numpy(padded).to(device)
    return result


def _load_multilayer_features(
    chembl_ids: Sequence[str],
    multilayer_dirs: list[Path],
    layers: list[int],
    model: _AttentionPoolingModel | None = None,
    device: torch.device | None = None,
) -> np.ndarray | None:
    """Load selected layers and attention-pool them via trained per-layer projectors.

    Uses ``model.ml_projectors[key]`` and ``model.ml_pools[key]`` which
    are dedicated projections trained end-to-end for each intermediate
    layer, instead of sharing the last-layer weights.

    Returns
    -------
    np.ndarray of shape ``(len(chembl_ids), len(layers) * hidden_dim)``
    or ``None`` if model has no multilayer support.
    """
    if model is None or device is None:
        return None
    if not hasattr(model, "ml_projectors") or not model.multilayer_layers:
        return None

    hidden_dim = model.hidden_dim
    # Load as tensors and forward through dedicated projectors.
    ml_tensors = _load_multilayer_tensors(chembl_ids, multilayer_dirs, layers, device)
    with torch.no_grad():
        ml_out = model.forward_multilayer(ml_tensors)
    if ml_out is not None:
        return ml_out.cpu().numpy()

    # Fallback: zeros.
    return np.zeros((len(chembl_ids), len(layers) * hidden_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature extraction with trained attention pooling
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _extract_features(
    model: _AttentionPoolingModel,
    loader: DataLoader,
    device: torch.device,
    desc: str | None = None,
    aux_head: nn.Module | None = None,
    include_aux_channel: bool = False,
    multilayer_dirs: list[Path] | None = None,
    multilayer_layers: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention-pooled features from a dataloader.

    When *multilayer_dirs* and *multilayer_layers* are provided, also
    loads intermediate MoLFormer hidden states from pre-computed .npz
    files, attention-pools each selected layer via the trained model,
    and concatenates to the main features.
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    batch_iter = loader
    if desc:
        batch_iter = tqdm(
            loader,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    if aux_head is not None:
        aux_head.eval()

    # Check whether to use interaction-enriched features.
    use_interactions = (
        hasattr(model, "interaction_features")
        and model.interaction_features
        and hasattr(model, "forward_with_interactions")
    )

    for batch in batch_iter:
        p = batch["protein_matrix"].to(device)
        l = batch["ligand_matrix"].to(device)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)

        if use_interactions:
            features_t = model.forward_with_interactions(p, l, pm, lm)
        else:
            features_t = model(p, l, pm, lm)

        if include_aux_channel and aux_head is not None:
            # Determine whether aux_head was trained with interaction features
            # by inspecting its input dimension.  When trained with
            # aux_interactions=True it expects 4×hidden+1; otherwise 2×hidden.
            # When multi-layer is active, K*hidden extra dims are also expected.
            aux_expects_interactions = aux_head[0].in_features > model.hidden_dim * 2
            if aux_expects_interactions:
                aux_input = features_t if use_interactions else model.forward_with_interactions(p, l, pm, lm)
            else:
                aux_input = model(p, l, pm, lm) if use_interactions else features_t

            # Append multi-layer features to aux_input if the model has
            # per-layer projectors (matching the training-time aux_head dim).
            if hasattr(model, "ml_projectors") and model.multilayer_layers and multilayer_dirs and multilayer_layers:
                ml_tensors = _load_multilayer_tensors(
                    batch["chembl_id"], multilayer_dirs, multilayer_layers, device,
                )
                ml_feats_t = model.forward_multilayer(ml_tensors)
                if ml_feats_t is not None:
                    aux_input = torch.cat([aux_input, ml_feats_t], dim=-1)

            # Extract rich aux representation: penultimate hidden layer
            # (hidden_dim activations) + final probability (1 scalar).
            # This gives the downstream MLP access to the learned
            # intermediate representation instead of just 1 number.
            aux_hidden = torch.nn.functional.gelu(aux_head[0](aux_input))  # [B, hidden_dim]
            aux_proba = torch.sigmoid(aux_head[3](aux_head[2](aux_hidden)))  # [B, 1]
            features_t = torch.cat([features_t, aux_hidden, aux_proba], dim=1)

        features = features_t.cpu().numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Append attention-pooled multi-layer features if requested.
        if multilayer_dirs and multilayer_layers:
            ml_feats = _load_multilayer_features(
                batch["chembl_id"], multilayer_dirs, multilayer_layers,
                model=model, device=device,
            )
            if ml_feats is not None:
                features = np.concatenate([features, ml_feats], axis=1)

        all_features.append(features)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def _compute_downstream_mcc_proxy(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> float:
    """Estimate downstream MCC using a lightweight MLP on extracted features."""
    if x_fit.size == 0 or x_val.size == 0 or len(np.unique(y_fit.astype(int))) < 2:
        return -1.0

    scaler = StandardScaler()
    x_fit_sc = scaler.fit_transform(x_fit).astype(np.float32)
    x_val_sc = scaler.transform(x_val).astype(np.float32)

    # Align proxy regularisation with the real MLP grid search (alpha ≥ 5e-3)
    # so that checkpoint selection reflects actual downstream performance.
    mlp = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        alpha=5e-3,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=1e-5,
        random_state=seed,
    )
    mlp.fit(x_fit_sc, y_fit)
    val_proba = mlp.predict_proba(x_val_sc)[:, 1]
    _, val_mcc = _best_mcc_threshold(y_val.astype(int), val_proba)
    return float(val_mcc)


# ---------------------------------------------------------------------------
# Level 3 runner
# ---------------------------------------------------------------------------

class Level3Runner(BaseLevelRunner):
    """Embedding matrices -> attention pooling -> KNN/MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def knn_is_deterministic(self) -> bool:
        return False  # Learned feature extractor → KNN input varies per seed

    @property
    def level_tag(self) -> str:
        return "level3_attnpool"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Train attention pooling, extract features, run KNN/MLP."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level3_knn_mlp_results.json")
        strict_freeze = (
            self.mode == "test"
            and os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        )
        if os.path.exists(cache_path) and not self.force and not strict_freeze:
            tqdm.write(f"  Loading cached Level 3 results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)
        if os.path.exists(cache_path) and not self.force and strict_freeze:
            tqdm.write("  Strict test mode: ignoring cached Level 3 results and recomputing.")

        tqdm.write(f"  Building Level 3 attention pooling (seed {seed})...")

        # Resolve protein dimension from embedding model
        protein_dim = PROTEIN_DIMS.get(self.embedding_name, 640)

        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=self._config.batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        # In train mode, split training data to avoid train-set optimism:
        # model trains on 80% of train, KNN features from the held-out 20%.
        feat_extract_loader = None
        if self.mode == "train":
            model_train_loader, feat_extract_loader = split_loader_for_feature_extraction(
                train_loader, seed=seed,
            )
        else:
            model_train_loader = train_loader

        # Train projection + attention pooling
        tqdm.write("  Training projection + attention pooling...")
        local_selection_metric = os.getenv(
            "BENCHMARK_LEVEL3_SELECTION_METRIC",
            self._config.model_selection_metric,
        ).strip().lower()
        if local_selection_metric not in {"val_loss", "mcc", "downstream_mcc"}:
            raise ValueError(
                "BENCHMARK_LEVEL3_SELECTION_METRIC must be one of: "
                "val_loss, mcc, downstream_mcc"
            )

        # Multi-layer MoLFormer features (optional).
        ml_layers_str = os.getenv("BENCHMARK_LEVEL3_MULTILAYER_LAYERS", "").strip()
        if ml_layers_str:
            ml_layers = [int(x) for x in ml_layers_str.split(",")]
            ml_dirs = _resolve_multilayer_dirs(self.dataset, self.embedding_name)
            tqdm.write(f"  Multi-layer features enabled: layers {ml_layers}")
        else:
            ml_layers = None
            ml_dirs = None

        model, aux_head = _train_attention_pooling(
            train_loader=model_train_loader,
            val_loader=val_loader,
            downstream_fit_loader=feat_extract_loader,
            protein_dim=protein_dim,
            lr=self._config.learning_rate,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            seed=seed,
            model_selection_metric=local_selection_metric,
            multilayer_layers=ml_layers,
            multilayer_dirs=ml_dirs,
        )

        use_aux_channel = os.getenv("BENCHMARK_LEVEL3_USE_AUX_CHANNEL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        device = next(model.parameters()).device

        if self.mode == "train":
            tqdm.write("  Extracting attention-pooled features (held-out train + val)...")
            x_fit, y_fit = _extract_features(
                model,
                feat_extract_loader,
                device,
                desc="    Feature extraction (fit)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
                multilayer_dirs=ml_dirs,
                multilayer_layers=ml_layers,
            )
            x_eval, y_eval = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (eval)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
                multilayer_dirs=ml_dirs,
                multilayer_layers=ml_layers,
            )
        else:
            tqdm.write("  Extracting attention-pooled features (val + test)...")
            x_fit, y_fit = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (fit)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
                multilayer_dirs=ml_dirs,
                multilayer_layers=ml_layers,
            )
            x_eval, y_eval = _extract_features(
                model,
                test_loader,
                device,
                desc="    Feature extraction (eval)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
                multilayer_dirs=ml_dirs,
                multilayer_layers=ml_layers,
            )

        # Sanitize
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Train canonical KNN/MLP (same as all levels)
        tqdm.write("  Training KNN + MLP (canonical classifiers)...")
        frozen_selection = None
        if self.mode == "test":
            frozen_selection = _load_frozen_mlp_selection_from_train(
                output_dir=output_dir,
                cache_filename="level3_knn_mlp_results.json",
            )
            if strict_freeze and frozen_selection is None:
                raise RuntimeError(
                    "Missing frozen train selection for Level 3 test run. "
                    "Run train phase first or set BENCHMARK_REQUIRE_TRAIN_SELECTION=0."
                )

        models = train_knn_mlp(
            x_fit,
            y_fit,
            x_eval,
            y_eval,
            seed,
            frozen_mlp_selection=frozen_selection,
        )

        sc_key = "Split by Scaffold"
        result = {sc_key: models}

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, "level3_attnpool_model.pt")
        torch.save(
            {
                "encoder": model.state_dict(),
                "aux_head": aux_head.state_dict(),
                "use_aux_channel": use_aux_channel,
            },
            checkpoint_path,
        )

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 3 (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result


class Level3aRunner(Level3Runner):
    """Level 3a — attention pooling + MLP only (skip KNN for speed)."""

    @property
    def level_tag(self) -> str:
        return "level3a_attnpool_mlp"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        """Train attention pooling, extract features, run MLP only."""
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level3a_mlp_results.json")
        strict_freeze = (
            self.mode == "test"
            and os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        )
        if os.path.exists(cache_path) and not self.force and not strict_freeze:
            tqdm.write(f"  Loading cached Level 3a results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)
        if os.path.exists(cache_path) and not self.force and strict_freeze:
            tqdm.write("  Strict test mode: ignoring cached Level 3a results and recomputing.")

        tqdm.write(f"  Building Level 3a attention pooling (seed {seed})...")

        protein_dim = PROTEIN_DIMS.get(self.embedding_name, 640)
        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=self._config.batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        feat_extract_loader = None
        use_full_train_features = os.getenv(
            "BENCHMARK_LEVEL3_FULL_TRAIN_FEATURES", "0"
        ).strip().lower() not in {"0", "false", "no"}

        if self.mode == "train" and not use_full_train_features:
            # Conservative: split train 80/20 to avoid train-set optimism.
            model_train_loader, feat_extract_loader = split_loader_for_feature_extraction(
                train_loader, seed=seed,
            )
        else:
            # Transfer learning mode: attention pooling learns a latent-space
            # projection (not classification), so extracting features from the
            # same data is standard practice.  Val/test remain unseen.
            model_train_loader = train_loader

        tqdm.write("  Training projection + attention pooling...")
        local_selection_metric = os.getenv(
            "BENCHMARK_LEVEL3_SELECTION_METRIC",
            self._config.model_selection_metric,
        ).strip().lower()
        if local_selection_metric not in {"val_loss", "mcc", "downstream_mcc"}:
            raise ValueError(
                "BENCHMARK_LEVEL3_SELECTION_METRIC must be one of: "
                "val_loss, mcc, downstream_mcc"
            )

        # Multi-layer MoLFormer features (optional).
        ml_layers_str = os.getenv("BENCHMARK_LEVEL3_MULTILAYER_LAYERS", "").strip()
        if ml_layers_str:
            ml_layers = [int(x) for x in ml_layers_str.split(",")]
            ml_dirs = _resolve_multilayer_dirs(self.dataset, self.embedding_name)
            tqdm.write(f"  Multi-layer features enabled: layers {ml_layers}")
        else:
            ml_layers = None
            ml_dirs = None

        model, aux_head = _train_attention_pooling(
            train_loader=model_train_loader,
            val_loader=val_loader,
            downstream_fit_loader=feat_extract_loader,
            protein_dim=protein_dim,
            lr=self._config.learning_rate,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            seed=seed,
            model_selection_metric=local_selection_metric,
            multilayer_layers=ml_layers,
            multilayer_dirs=ml_dirs,
        )

        use_aux_channel = os.getenv("BENCHMARK_LEVEL3_USE_AUX_CHANNEL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }

        device = next(model.parameters()).device
        if self.mode == "train":
            fit_loader = train_loader if use_full_train_features else feat_extract_loader
            fit_desc = "full train" if use_full_train_features else "held-out train"
            tqdm.write(f"  Extracting attention-pooled features ({fit_desc} + val)...")
            x_fit, y_fit = _extract_features(
                model,
                fit_loader,
                device,
                desc=f"    Feature extraction (fit — {fit_desc})",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
            )
            x_eval, y_eval = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (eval)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
            )
        else:
            tqdm.write("  Extracting attention-pooled features (val + test)...")
            x_fit, y_fit = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (fit)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
            )
            x_eval, y_eval = _extract_features(
                model,
                test_loader,
                device,
                desc="    Feature extraction (eval)",
                aux_head=aux_head,
                include_aux_channel=use_aux_channel,
            )

        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        tqdm.write("  Training MLP only (KNN skipped)...")
        frozen_selection = None
        if self.mode == "test":
            frozen_selection = _load_frozen_mlp_selection_from_train(
                output_dir=output_dir,
                cache_filename="level3a_mlp_results.json",
            )
            if strict_freeze and frozen_selection is None:
                raise RuntimeError(
                    "Missing frozen train selection for Level 3a test run. "
                    "Run train phase first or set BENCHMARK_REQUIRE_TRAIN_SELECTION=0."
                )

        mlp_metrics = train_mlp_only(
            x_fit,
            y_fit,
            x_eval,
            y_eval,
            seed,
            frozen_mlp_selection=frozen_selection,
        )

        sc_key = "Split by Scaffold"
        result = {sc_key: {"MLP": mlp_metrics}}

        checkpoint_path = os.path.join(output_dir, "level3a_attnpool_model.pt")
        torch.save(
            {
                "encoder": model.state_dict(),
                "aux_head": aux_head.state_dict(),
                "use_aux_channel": use_aux_channel,
            },
            checkpoint_path,
        )

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(f"  Level 3a (seed {seed}): MLP MCC={mlp_metrics['mcc']:.4f}")
        return result
