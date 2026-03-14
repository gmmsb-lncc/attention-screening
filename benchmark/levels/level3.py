"""Level 3 — Embedding matrices + attention pooling + KNN/MLP.

Loads the **same** per-residue ESM-2 protein matrices and per-token
MoLFormer ligand matrices used by Level 2, but applies a
**learned attention pooling** mechanism to produce fixed-size vectors.

Architecture (identical to Level 4 minus cross-attention):
  1. Linear projection  (protein_dim → hidden_dim / ligand_dim → hidden_dim)
  2. Attention pooling   (learnable query, multi-head attention, [B, hidden_dim])
  3. Concatenation       [protein_vec ‖ ligand_vec]  →  [B, 2 × hidden_dim]

Difference from other levels:
  - Level 2: mean pooling (no parameters)
  - **Level 3: projection + attention pooling (light trainable parameters)**

This level isolates the contribution of *learned aggregation* without
the inter-modality interaction that cross-attention provides.

Training protocol (consistent with all levels):
  - Attention pooling model trained on the **training** split
    (validation split for early stopping).
  - Training hyperparameters (epochs, patience, learning rate) are
        taken from the CLI / ``BenchmarkConfig``.
    - In ``train`` mode: fit on held-out train features, evaluate on val features.
    - In ``test`` mode: fit on val features, evaluate on test features.

Classifier note: KNN and MLP are provided by ``benchmark.classifiers``
to guarantee identical hyperparameters across all active levels.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.classifiers import train_knn_mlp
from benchmark.config import (
    PROTEIN_DIMS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    split_loader_for_feature_extraction,
)
from benchmark.levels.protocol import sanitize_features
from benchmark.levels.protein_structure_features import (
    ProteinStructureFeatureLoader,
    build_protein_structure_loader,
)
from benchmark.levels.se3_features import SE3FeatureLoader, build_se3_loader
from benchmark.runtime import (
    batch_size_fallbacks,
    is_cuda_oom_error,
    resolve_effective_batch_size,
)


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
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

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

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.protein_proj, self.ligand_proj):
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

        return torch.cat([protein_vec, ligand_vec], dim=-1)  # [B, 2*hidden]


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

def _train_attention_pooling(
    train_loader: DataLoader,
    val_loader: DataLoader,
    protein_dim: int,
    ligand_dim: int = 768,
    hidden_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int | None = 10,
    seed: int = 42,
    log_prefix: str = "",
) -> _AttentionPoolingModel:
    """Train the projection + attention pooling model.

    Uses a lightweight binary cross-entropy objective just to learn
    *which tokens matter* — the actual classification is delegated to
    the canonical KNN/MLP afterwards.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _AttentionPoolingModel(
        protein_input_dim=protein_dim,
        ligand_input_dim=ligand_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    # Lightweight classification head used ONLY for training the pooling
    # (discarded afterwards — KNN/MLP are the real classifiers)
    aux_head = nn.Linear(hidden_dim * 2, 1).to(device)
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
    best_state = None
    wait = 0

    start_time = time.time()
    log_every = max(1, epochs // 20)
    tqdm.write(
        f"{log_prefix} Training start: epochs={epochs}, patience={patience}, "
        f"device={device}, protein_dim={protein_dim}, ligand_dim={ligand_dim}, "
        f"log_every={log_every}"
    )

    for epoch in range(1, epochs + 1):
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

            features = model(p, l, pm, lm)
            logits = aux_head(features)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # --- validate ---
        model.eval()
        aux_head.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                p = batch["protein_matrix"].to(device)
                l = batch["ligand_matrix"].to(device)
                pm = batch["protein_mask"].to(device)
                lm = batch["ligand_mask"].to(device)
                y = batch["label"].to(device).unsqueeze(1)

                features = model(p, l, pm, lm)
                logits = aux_head(features)
                val_loss += criterion(logits, y).item()
                val_n += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)

        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / max(epoch, 1)) * max(epochs - epoch, 0)
            progress = 100.0 * epoch / max(epochs, 1)
            best_so_far = min(best_val_loss, avg_val)
            tqdm.write(
                f"{log_prefix} Epoch {epoch}/{epochs} ({progress:5.1f}%) "
                f"train={avg_train:.4f} val={avg_val:.4f} best={best_so_far:.4f} "
                f"wait={wait} eta={eta/60:.1f}m"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                tqdm.write(
                    f"{log_prefix} Early stopping at epoch {epoch}/{epochs} "
                    f"(val={avg_val:.4f}, best={best_val_loss:.4f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature extraction with trained attention pooling
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(
    model: _AttentionPoolingModel,
    loader: DataLoader,
    device: torch.device,
    protein_structure_loader: ProteinStructureFeatureLoader | None = None,
    se3_loader: SE3FeatureLoader | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention-pooled features from a dataloader."""
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    esmfold_logged = False
    se3_logged = False

    for batch in loader:
        p = batch["protein_matrix"].to(device)
        l = batch["ligand_matrix"].to(device)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)

        features = model(p, l, pm, lm).cpu().numpy()
        hidden_dim = model.hidden_dim
        protein_part = features[:, :hidden_dim]
        ligand_part = features[:, hidden_dim:]

        if protein_structure_loader is not None:
            prot_struct_batch = protein_structure_loader.get_batch(batch["seq_id"])
            if prot_struct_batch.shape[1] > 0:
                if not esmfold_logged:
                    prot_sem_dim = int(protein_part.shape[1])
                    prot_str_dim = int(prot_struct_batch.shape[1])
                    prot_fused_dim = prot_sem_dim + prot_str_dim
                    tqdm.write(
                        f"  [ESMFold] Active fusion (Level 3): protein_sem_dim={prot_sem_dim}, protein_struct_dim={prot_str_dim}, fused_protein_dim={prot_fused_dim}"
                    )
                    esmfold_logged = True
                protein_part = np.concatenate([protein_part, prot_struct_batch], axis=-1)

        if se3_loader is not None:
            se3_batch = se3_loader.get_batch(batch["chembl_id"])
            if se3_batch.shape[1] > 0:
                if not se3_logged:
                    sem_dim = int(ligand_part.shape[1])
                    se3_dim = int(se3_batch.shape[1])
                    fused_dim = sem_dim + se3_dim
                    tqdm.write(
                        f"  [SE3] Active fusion (Level 3): ligand_sem_dim={sem_dim}, se3_dim={se3_dim}, fused_ligand_dim={fused_dim}"
                    )
                    se3_logged = True
                ligand_part = np.concatenate([ligand_part, se3_batch], axis=-1)

        features = np.concatenate([protein_part, ligand_part], axis=-1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(features)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


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
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 3 results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        tqdm.write(f"  Building Level 3 attention pooling (seed {seed})...")

        se3_loader = build_se3_loader(
            feature_dir=self._config.se3_features_dir,
            use_se3_ligand=self._config.use_se3_ligand,
            default_feature_dirs=self._config.resolved_se3_feature_dirs,
        )

        protein_structure_loader = build_protein_structure_loader(
            feature_dir=self._config.esmfold_features_dir,
            use_esmfold_protein=self._config.use_esmfold_protein,
            default_feature_dirs=self._config.resolved_esmfold_feature_dirs,
        )

        if self._config.use_esmfold_protein and protein_structure_loader is None:
            tqdm.write("  WARNING: ESMFold protein fusion requested, but no valid .npy structure vectors were found. Proceeding with semantic protein features only.")
        elif protein_structure_loader is not None:
            tqdm.write(
                f"  [ESMFold] Structural vectors loaded: dim={protein_structure_loader.dim}. Protein concatenation enabled."
            )

        if self._config.use_se3_ligand and se3_loader is None:
            tqdm.write("  WARNING: SE3 ligand fusion requested, but no valid .npy structural vectors were found. Proceeding with semantic ligand features only.")
        elif se3_loader is not None:
            tqdm.write(
                f"  [SE3] Structural vectors loaded: dim={se3_loader.dim}. Ligand concatenation enabled."
            )

        # Resolve protein dimension from embedding model
        protein_dim = PROTEIN_DIMS.get(self.embedding_name, 640)
        log_prefix = f"[L3|seed {seed}]"

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

                # In train mode, split training data to avoid train-set optimism:
                # model trains on 80% of train, KNN features from held-out 20%.
                feat_extract_loader = None
                if self.mode == "train":
                    model_train_loader, feat_extract_loader = split_loader_for_feature_extraction(
                        train_loader, seed=seed,
                    )
                else:
                    model_train_loader = train_loader

                tqdm.write(f"{log_prefix} Training projection + attention pooling...")
                model = _train_attention_pooling(
                    train_loader=model_train_loader,
                    val_loader=val_loader,
                    protein_dim=protein_dim,
                    ligand_dim=self._config.ligand_dim,
                    lr=self._config.learning_rate,
                    epochs=self._config.epochs,
                    patience=self._config.resolved_patience,
                    seed=seed,
                    log_prefix=log_prefix,
                )

                device = next(model.parameters()).device

                if self.mode == "train":
                    tqdm.write("  Extracting attention-pooled features (held-out train + val)...")
                    x_fit, y_fit = _extract_features(
                        model,
                        feat_extract_loader,
                        device,
                        protein_structure_loader=protein_structure_loader,
                        se3_loader=se3_loader,
                    )
                    x_eval, y_eval = _extract_features(
                        model,
                        val_loader,
                        device,
                        protein_structure_loader=protein_structure_loader,
                        se3_loader=se3_loader,
                    )
                else:
                    tqdm.write("  Extracting attention-pooled features (val + test)...")
                    x_fit, y_fit = _extract_features(
                        model,
                        val_loader,
                        device,
                        protein_structure_loader=protein_structure_loader,
                        se3_loader=se3_loader,
                    )
                    x_eval, y_eval = _extract_features(
                        model,
                        test_loader,
                        device,
                        protein_structure_loader=protein_structure_loader,
                        se3_loader=se3_loader,
                    )

                # Sanitize
                for name, arr in [("fit", x_fit), ("eval", x_eval)]:
                    arr_sanitized, bad = sanitize_features(arr)
                    if bad:
                        tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                        arr[:] = arr_sanitized

                protein_feature_dim = int(model.hidden_dim)
                if protein_structure_loader is not None:
                    protein_feature_dim += protein_structure_loader.dim
                if self._config.ligand_weight != 1.0:
                    tqdm.write(
                        f"  [Fusion] Ligand block weighting enabled: ligand_weight={self._config.ligand_weight:.3f} (protein block weight=1.000)"
                    )

                # Train canonical KNN/MLP (same as all levels)
                tqdm.write("  Training KNN + MLP (canonical classifiers)...")
                models = train_knn_mlp(
                    x_fit,
                    y_fit,
                    x_eval,
                    y_eval,
                    seed,
                    protein_dim=protein_feature_dim,
                    ligand_weight=self._config.ligand_weight,
                )
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
        checkpoint_path = os.path.join(output_dir, "level3_attnpool_model.pt")
        torch.save(model.state_dict(), checkpoint_path)

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 3 (seed {seed}): "
            f"KNN MCC={models['KNN']['mcc']:.4f}, "
            f"MLP MCC={models['MLP']['mcc']:.4f}"
        )
        return result
