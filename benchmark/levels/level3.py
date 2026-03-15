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
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
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
    hidden_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 10,
    seed: int = 42,
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
        ligand_input_dim=MOLFORMER_DIM,
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
    best_val_mcc = -1.0
    best_state = None
    wait = 0

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
        val_probs: list[np.ndarray] = []
        val_targets: list[np.ndarray] = []
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
                val_probs.append(torch.sigmoid(logits).cpu().numpy().ravel())
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
            best_mcc=f"{best_val_mcc:.3f}",
        )

        if (val_mcc > best_val_mcc) or (
            np.isclose(val_mcc, best_val_mcc) and avg_val < best_val_loss
        ):
            best_val_loss = avg_val
            best_val_mcc = val_mcc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience and wait >= patience:
                tqdm.write(
                    f"    Early stopping at epoch {epoch} "
                    f"(val_mcc={val_mcc:.4f}, best_mcc={best_val_mcc:.4f})"
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
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention-pooled features from a dataloader."""
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

    for batch in batch_iter:
        p = batch["protein_matrix"].to(device)
        l = batch["ligand_matrix"].to(device)
        pm = batch["protein_mask"].to(device)
        lm = batch["ligand_mask"].to(device)

        features = model(p, l, pm, lm).cpu().numpy()
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
        model = _train_attention_pooling(
            train_loader=model_train_loader,
            val_loader=val_loader,
            protein_dim=protein_dim,
            lr=self._config.learning_rate,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            seed=seed,
        )

        device = next(model.parameters()).device

        if self.mode == "train":
            tqdm.write("  Extracting attention-pooled features (held-out train + val)...")
            x_fit, y_fit = _extract_features(
                model,
                feat_extract_loader,
                device,
                desc="    Feature extraction (fit)",
            )
            x_eval, y_eval = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (eval)",
            )
        else:
            tqdm.write("  Extracting attention-pooled features (val + test)...")
            x_fit, y_fit = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (fit)",
            )
            x_eval, y_eval = _extract_features(
                model,
                test_loader,
                device,
                desc="    Feature extraction (eval)",
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
            strict_freeze = os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
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
        torch.save(model.state_dict(), checkpoint_path)

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
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 3a results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

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
        if self.mode == "train":
            model_train_loader, feat_extract_loader = split_loader_for_feature_extraction(
                train_loader, seed=seed,
            )
        else:
            model_train_loader = train_loader

        tqdm.write("  Training projection + attention pooling...")
        model = _train_attention_pooling(
            train_loader=model_train_loader,
            val_loader=val_loader,
            protein_dim=protein_dim,
            lr=self._config.learning_rate,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            seed=seed,
        )

        device = next(model.parameters()).device
        if self.mode == "train":
            tqdm.write("  Extracting attention-pooled features (held-out train + val)...")
            x_fit, y_fit = _extract_features(
                model,
                feat_extract_loader,
                device,
                desc="    Feature extraction (fit)",
            )
            x_eval, y_eval = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (eval)",
            )
        else:
            tqdm.write("  Extracting attention-pooled features (val + test)...")
            x_fit, y_fit = _extract_features(
                model,
                val_loader,
                device,
                desc="    Feature extraction (fit)",
            )
            x_eval, y_eval = _extract_features(
                model,
                test_loader,
                device,
                desc="    Feature extraction (eval)",
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
            strict_freeze = os.getenv("BENCHMARK_REQUIRE_TRAIN_SELECTION", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
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
        torch.save(model.state_dict(), checkpoint_path)

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(f"  Level 3a (seed {seed}): MLP MCC={mlp_metrics['mcc']:.4f}")
        return result
