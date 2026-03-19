"""Level 4 — LoRA-Adapted MoLFormer + Attention Pooling + MLP.

Applies Low-Rank Adaptation (LoRA) to MoLFormer's attention layers,
allowing ~0.3% of parameters to specialise for kinase bioactivity
prediction while keeping 99.7% of the pre-trained weights frozen.

**Key difference from Level 3:**
  - Level 3: MoLFormer is completely frozen; ligand matrices are
    pre-computed ``.npy`` files loaded from disk.
  - Level 4: MoLFormer runs **on-the-fly** with LoRA adapters; SMILES
    are tokenised per-batch and passed through the adapted encoder.
    The protein side still uses pre-computed ESM-2 matrices.

Architecture (per batch):
  SMILES → Tokeniser → MoLFormer(❄base + 🔥LoRA) → [B, seq, 768]
                                                          ↓
  Protein .npy → proj + AttnPool ←── Ligand proj + AttnPool
                                                          ↓
                       [prot_vec ‖ lig_vec ‖ interactions] → aux_head (BCE)

Trainable parameters (~663K):
  - LoRA adapters (Q/V in all attention layers): ~120K (rank 8)
  - Protein/Ligand projections + attention pools: ~543K

Training protocol:
  Same as Level 3a — ``downstream_mcc`` selection, early stopping,
  then MLP classifier on extracted features.

Requirements:
  ``pip install peft``  (HuggingFace PEFT library)
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from benchmark.classifiers import train_mlp_only
from benchmark.config import (
    EMBEDDING_BASE_PATH,
    MOLFORMER_DIM,
    PROTEIN_DIMS,
    SUPPORTED_EMBEDDINGS,
    BenchmarkConfig,
)
from benchmark.levels.base import BaseLevelRunner
from benchmark.levels.matrix_utils import (
    build_matrix_dataloaders,
    pad_matrices,
    split_loader_for_feature_extraction,
)


# ── MoLFormer model name on HuggingFace ──────────────────────────────
MOLFORMER_HF_NAME = "ibm/MoLFormer-XL-both-10pct"


# ── SMILES augmentation ──────────────────────────────────────────────

def _randomize_smiles(smiles: str) -> str:
    """Generate a random but chemically equivalent SMILES string.

    Uses RDKit to parse the canonical SMILES and regenerate it with a
    random atom ordering.  Falls back to the original on failure.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smiles


def _augment_smiles_batch(smiles_list: list[str]) -> list[str]:
    """Randomize each SMILES in a batch."""
    return [_randomize_smiles(s) for s in smiles_list]


# ======================================================================
# Dataset: protein matrices + SMILES (no pre-computed ligand matrices)
# ======================================================================


class LoRAMatrixDataset(Dataset):
    """Loads protein matrices from ``.npy`` and carries raw SMILES.

    Unlike :class:`MatrixDataset`, this dataset does **not** load
    pre-computed ligand matrices.  The SMILES string is returned as-is;
    MoLFormer tokenisation happens in the collate function so that the
    tokeniser can batch-pad efficiently.
    """

    def __init__(
        self,
        df: "pd.DataFrame",
        protein_matrix_dirs: Sequence[Path],
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._protein_dirs = list(protein_matrix_dirs)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple:
        row = self._df.iloc[idx]
        seq_id = row["seq_id"]
        chembl_id = row["chembl_id"]
        smiles = row["smiles"]
        label = row["label"]

        protein_mat = self._load_protein(seq_id)
        return protein_mat, smiles, label, seq_id, chembl_id

    def _load_protein(self, seq_id: str) -> np.ndarray:
        for d in self._protein_dirs:
            path = d / f"{seq_id}_matrix.npy"
            if path.exists():
                return np.load(str(path)).astype(np.float32)
        # Fallback zeros
        return np.zeros((100, 320), dtype=np.float32)


def _collate_lora(batch: list) -> dict:
    """Collate function for :class:`LoRAMatrixDataset`.

    Pads protein matrices, bundles raw SMILES strings (tokenisation
    deferred to the training loop where the tokeniser is available).
    """
    protein_mats, smiles_list, labels, seq_ids, chembl_ids = zip(*batch)

    protein_batch, protein_mask = pad_matrices(protein_mats)

    return {
        "protein_matrix": torch.from_numpy(protein_batch),
        "protein_mask": torch.from_numpy(protein_mask),
        "smiles": list(smiles_list),
        "label": torch.tensor(labels, dtype=torch.float32),
        "seq_id": seq_ids,
        "chembl_id": chembl_ids,
    }


# ======================================================================
# Model: LoRA-MoLFormer + Protein projection + Attention Pool
# ======================================================================


class _AttentionPool(nn.Module):
    """Multi-head attention pooling with a learnable query."""

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # [B, embed_dim]


class LoRAAttentionPoolingModel(nn.Module):
    """Joint model: LoRA-MoLFormer encoder + protein/ligand projection + AttnPool.

    The MoLFormer base weights are frozen.  Only the LoRA adapters,
    projection layers, and attention pools are trained.
    """

    def __init__(
        self,
        molformer: nn.Module,
        tokenizer: object,
        protein_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.molformer = molformer
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim

        # Protein branch (same architecture as Level 3)
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.protein_pool = _AttentionPool(hidden_dim, num_heads)

        # Ligand branch (projects MoLFormer output)
        self.ligand_proj = nn.Sequential(
            nn.Linear(MOLFORMER_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.ligand_pool = _AttentionPool(hidden_dim, num_heads)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.protein_proj, self.ligand_proj]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward_ligand(
        self,
        smiles: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise SMILES and run through MoLFormer(LoRA).

        Returns
        -------
        ligand_embeds : torch.Tensor  [B, seq, 768]
        ligand_mask   : torch.Tensor  [B, seq]  (True = valid)
        """
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=202,
        ).to(device)

        outputs = self.molformer(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            embeds = outputs.last_hidden_state  # [B, seq, 768]
        else:
            embeds = outputs[0]

        attention_mask = inputs["attention_mask"].bool()  # [B, seq]
        return embeds, attention_mask

    def forward(
        self,
        protein_matrix: torch.Tensor,
        protein_mask: torch.Tensor,
        smiles: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Full forward: protein proj + ligand LoRA → concat."""
        # Protein branch
        prot_h = self.protein_proj(protein_matrix)
        prot_vec = self.protein_pool(prot_h, protein_mask)

        # Ligand branch (on-the-fly MoLFormer)
        lig_embeds, lig_mask = self.forward_ligand(smiles, device)
        lig_h = self.ligand_proj(lig_embeds)
        lig_vec = self.ligand_pool(lig_h, lig_mask)

        return torch.cat([prot_vec, lig_vec], dim=-1)  # [B, 2*hidden_dim]

    def forward_with_interactions(
        self,
        protein_matrix: torch.Tensor,
        protein_mask: torch.Tensor,
        smiles: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Forward with interaction features (same as Level 3)."""
        prot_h = self.protein_proj(protein_matrix)
        prot_vec = self.protein_pool(prot_h, protein_mask)

        lig_embeds, lig_mask = self.forward_ligand(smiles, device)
        lig_h = self.ligand_proj(lig_embeds)
        lig_vec = self.ligand_pool(lig_h, lig_mask)

        product = prot_vec * lig_vec
        diff = torch.abs(prot_vec - lig_vec)

        return torch.cat([prot_vec, lig_vec, product, diff], dim=-1)  # [B, 4*hidden_dim]


# ======================================================================
# LoRA application helper
# ======================================================================


def _apply_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: int = 16,
    lora_dropout: float = 0.15,
    target_layers: list[int] | None = None,
) -> nn.Module:
    """Wrap a HuggingFace model with LoRA adapters via PEFT.

    Parameters
    ----------
    target_layers : list[int] | None
        Which transformer layers to adapt (0-indexed).  When *None*,
        all layers are adapted.  E.g. ``[4, 5]`` adapts only the last
        two layers of a 6-layer model.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # Discover Q/V projection modules, optionally filtering by layer
    target_modules = _discover_attention_modules(model, target_layers=target_layers)
    if not target_modules:
        # Fallback names common in MoLFormer / transformer architectures
        target_modules = ["q_proj", "v_proj", "query", "value"]

    tqdm.write(
        f"    LoRA config: rank={rank}, alpha={alpha}, dropout={lora_dropout}, "
        f"layers={target_layers or 'all'}, targets={target_modules}"
    )

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(model, config)

    # Report parameter counts
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    tqdm.write(f"    LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return peft_model


def _discover_attention_modules(
    model: nn.Module,
    target_layers: list[int] | None = None,
) -> list[str]:
    """Find linear Q/V layers in attention blocks that can be LoRA-adapted.

    When *target_layers* is given, only modules whose full name contains
    a layer index in that list are returned (using full-path matching).
    """
    qv_keywords = {"q_proj", "v_proj", "query", "value"}
    found: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short = name.split(".")[-1]
        if short not in qv_keywords:
            continue

        if target_layers is not None:
            # Check if this module belongs to one of the target layers.
            # Module names typically look like: encoder.layer.4.attention.query
            # or: transformer.h.4.attn.q_proj — extract digits from path.
            parts = name.split(".")
            layer_idx = None
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            if layer_idx is None or layer_idx not in target_layers:
                continue
            # Use full path for per-layer targeting
            found.append(name)
        else:
            # Target by shortname (applies to ALL layers)
            if short not in {n.split(".")[-1] for n in found}:
                found.append(short)

    return sorted(found)


# ======================================================================
# Training loop
# ======================================================================


def _train_lora_attention_pooling(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    downstream_fit_loader: DataLoader | None,
    protein_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    seed: int,
    lora_rank: int = 4,
    lora_alpha: int = 16,
    model_selection_metric: str = "downstream_mcc",
) -> tuple[LoRAAttentionPoolingModel, nn.Sequential]:
    """Train LoRA-MoLFormer + attention pooling model.

    Mirrors :func:`level3._train_attention_pooling` but with
    on-the-fly MoLFormer inference per batch.
    """
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load MoLFormer + apply LoRA ---
    tqdm.write("    Loading MoLFormer + applying LoRA...")
    base_molformer = AutoModel.from_pretrained(MOLFORMER_HF_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MOLFORMER_HF_NAME, trust_remote_code=True)

    # Freeze all base params
    for p in base_molformer.parameters():
        p.requires_grad = False

    # Apply LoRA (only to selected layers)
    lora_dropout = float(os.getenv("BENCHMARK_LEVEL4_LORA_DROPOUT", "0.15"))
    lora_layers_str = os.getenv("BENCHMARK_LEVEL4_LORA_LAYERS", "4,5").strip()
    if lora_layers_str.lower() == "all":
        target_layers = None  # all layers
    else:
        target_layers = [int(x) for x in lora_layers_str.split(",")]
    lora_molformer = _apply_lora(
        base_molformer, rank=lora_rank, alpha=lora_alpha,
        lora_dropout=lora_dropout, target_layers=target_layers,
    )

    # --- Hidden dim (aligned to num_heads) ---
    hidden_dim = max(64, min(protein_dim, 512))
    num_heads = 8
    if hidden_dim % num_heads != 0:
        hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
    dropout = 0.30
    weight_decay = 0.02

    # Label smoothing: reduces overconfident predictions
    label_smoothing = float(os.getenv("BENCHMARK_LEVEL4_LABEL_SMOOTHING", "0.05"))
    # Differential LR: LoRA adapters get lower LR to avoid overfitting the encoder
    lora_lr_factor = float(os.getenv("BENCHMARK_LEVEL4_LORA_LR_FACTOR", "0.5"))

    tqdm.write(
        f"    L4 config: hidden_dim={hidden_dim}, dropout={dropout:.2f}, "
        f"lr={lr:.1e} (lora_lr={lr*lora_lr_factor:.1e}), wd={weight_decay}, "
        f"protein_dim={protein_dim}, lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
        f"label_smooth={label_smoothing}, lora_dropout={lora_dropout}"
    )

    # --- Build joint model ---
    model = LoRAAttentionPoolingModel(
        molformer=lora_molformer,
        tokenizer=tokenizer,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_heads=num_heads,
    )
    model.to(device)

    # Interaction features: 4*hidden_dim + 1 (cosine sim)
    aux_input_dim = 4 * hidden_dim + 1
    aux_head = nn.Sequential(
        nn.Linear(aux_input_dim, hidden_dim),
        nn.Dropout(dropout),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    # --- Optimizer: differential LR for LoRA vs projection ---
    # LoRA adapters get lower LR to avoid overfitting the pretrained encoder
    lora_params = []
    proj_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            proj_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr * lora_lr_factor},
        {"params": proj_params, "lr": lr},
        {"params": list(aux_head.parameters()), "lr": lr},
    ], weight_decay=weight_decay)

    # All trainable params (for grad clipping)
    all_trainable = lora_params + proj_params + list(aux_head.parameters())

    tqdm.write(
        f"    Param groups: LoRA={sum(p.numel() for p in lora_params):,} (lr={lr*lora_lr_factor:.1e}), "
        f"Proj={sum(p.numel() for p in proj_params):,} (lr={lr:.1e}), "
        f"Aux={sum(p.numel() for p in aux_head.parameters()):,} (lr={lr:.1e})"
    )

    # --- Scheduler: linear warmup + cosine decay ---
    warmup_epochs = max(1, int(epochs * 0.1))
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Loss with label smoothing ---
    criterion = nn.BCEWithLogitsLoss()

    # --- SMILES augmentation ---
    smiles_augment = os.getenv(
        "BENCHMARK_LEVEL4_SMILES_AUGMENT", "1"
    ).strip().lower() not in {"0", "false", "no"}
    if smiles_augment:
        tqdm.write("    SMILES augmentation: ENABLED (random SMILES each epoch)")
    else:
        tqdm.write("    SMILES augmentation: disabled")

    # --- Training loop ---
    best_state = None
    best_aux_state = None
    best_score = -float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        aux_head.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            p = batch["protein_matrix"].to(device)
            pm = batch["protein_mask"].to(device)
            smiles = batch["smiles"]

            # SMILES augmentation: randomize during training only
            if smiles_augment:
                smiles = _augment_smiles_batch(smiles)

            y = batch["label"].to(device)

            # Apply label smoothing: 0 → ε, 1 → 1-ε
            if label_smoothing > 0:
                y_smooth = y * (1.0 - label_smoothing) + (1.0 - y) * label_smoothing
            else:
                y_smooth = y

            features = model.forward_with_interactions(p, pm, smiles, device)

            # Cosine similarity as extra feature
            half = features.shape[-1] // 4
            prot_v = features[:, :half]
            lig_v = features[:, half:2*half]
            cos_sim = nn.functional.cosine_similarity(prot_v, lig_v, dim=-1, eps=1e-8).unsqueeze(-1)
            aux_input = torch.cat([features, cos_sim], dim=-1)

            logits = aux_head(aux_input).squeeze(-1)
            loss = criterion(logits, y_smooth)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        # Step scheduler after each epoch
        scheduler.step()

        avg_loss = running_loss / max(n_batches, 1)

        # --- Validation ---
        model.eval()
        aux_head.eval()
        val_loss_sum = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                p = batch["protein_matrix"].to(device)
                pm = batch["protein_mask"].to(device)
                smiles = batch["smiles"]
                y = batch["label"].to(device)

                features = model.forward_with_interactions(p, pm, smiles, device)
                half = features.shape[-1] // 4
                prot_v = features[:, :half]
                lig_v = features[:, half:2*half]
                cos_sim = nn.functional.cosine_similarity(prot_v, lig_v, dim=-1, eps=1e-8).unsqueeze(-1)
                aux_input = torch.cat([features, cos_sim], dim=-1)

                logits = aux_head(aux_input).squeeze(-1)
                val_loss_sum += criterion(logits, y).item()

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_labels.extend(y.cpu().numpy().tolist())

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)

        # Sweep threshold for best MCC
        best_thr, best_mcc_val = 0.5, 0.0
        for thr in np.arange(0.05, 0.95, 0.01):
            preds_bin = (val_preds_arr >= thr).astype(int)
            if len(np.unique(preds_bin)) < 2:
                continue
            mcc = matthews_corrcoef(val_labels_arr, preds_bin)
            if mcc > best_mcc_val:
                best_mcc_val = mcc
                best_thr = thr

        # Model selection
        if model_selection_metric == "downstream_mcc":
            score = best_mcc_val
        elif model_selection_metric == "val_loss":
            score = -val_loss
        else:
            score = best_mcc_val

        if score > best_score:
            best_score = score
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_aux_state = {k: v.cpu().clone() for k, v in aux_head.state_dict().items()}
            tqdm.write(
                f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_mcc={best_mcc_val:.4f}, thr={best_thr:.3f} ★"
            )
        else:
            no_improve += 1
            tqdm.write(
                f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_mcc={best_mcc_val:.4f}, thr={best_thr:.3f} ({no_improve}/{patience})"
            )
            if no_improve >= patience:
                tqdm.write(
                    f"    Early stopping at epoch {epoch+1} "
                    f"(val_loss={val_loss:.6f}, val_mcc={best_mcc_val:.4f}, "
                    f"best_sel={best_score:.6f}, thr={best_thr:.3f})"
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        aux_head.load_state_dict(best_aux_state)
    model.to(device).eval()
    aux_head.to(device).eval()

    # Clean up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return model, aux_head


# ======================================================================
# Feature extraction
# ======================================================================


@torch.inference_mode()
def _extract_lora_features(
    model: LoRAAttentionPoolingModel,
    aux_head: nn.Sequential,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str = "",
    include_aux_channel: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features using LoRA-adapted MoLFormer."""
    model.eval()
    if aux_head is not None:
        aux_head.eval()

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    batch_iter = loader
    if desc:
        batch_iter = tqdm(loader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)

    for batch in batch_iter:
        p = batch["protein_matrix"].to(device)
        pm = batch["protein_mask"].to(device)
        smiles = batch["smiles"]

        features_t = model.forward_with_interactions(p, pm, smiles, device)

        if include_aux_channel and aux_head is not None:
            half = features_t.shape[-1] // 4
            prot_v = features_t[:, :half]
            lig_v = features_t[:, half:2*half]
            cos_sim = nn.functional.cosine_similarity(prot_v, lig_v, dim=-1, eps=1e-8).unsqueeze(-1)
            aux_input = torch.cat([features_t, cos_sim], dim=-1)

            # aux_head = Sequential(Linear, Dropout, GELU, Linear)
            # Extract hidden representation after first 3 layers (Linear→Dropout→GELU)
            aux_hidden = aux_head[2](aux_head[1](aux_head[0](aux_input)))  # [B, hidden_dim]
            aux_proba = torch.sigmoid(aux_head[3](aux_hidden))  # [B, 1]
            features_t = torch.cat([features_t, aux_hidden, aux_proba], dim=1)

        features = features_t.cpu().numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        all_features.append(features)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


# ======================================================================
# DataLoader builders
# ======================================================================


def _build_lora_dataloaders(
    dataset_type: str,
    embedding_name: str,
    scaffold_split_dir: str,
    batch_size: int = 32,
    dataset_source_filter: str | None = None,
    mode: str = "train",
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build dataloaders that carry SMILES instead of pre-computed ligand matrices.

    Mirrors ``build_matrix_dataloaders`` but creates ``LoRAMatrixDataset``
    instances that return raw SMILES for on-the-fly MoLFormer inference.
    """
    import pandas as pd
    from benchmark.config import EMBEDDING_BASE_PATH, SUPPORTED_EMBEDDINGS

    full_emb = SUPPORTED_EMBEDDINGS.get(embedding_name, embedding_name)
    base_path = Path(EMBEDDING_BASE_PATH)

    # Discover protein matrix dirs and TSV splits
    protein_dirs: list[Path] = []
    datasets = [dataset_type] if dataset_type != "all" else ["human", "non_human"]

    for ds in datasets:
        build_dir = base_path / f"protein_model_benchmark_{ds}" / full_emb / "build"
        prot_dir = build_dir / "protein_matrices"
        if prot_dir.exists():
            protein_dirs.append(prot_dir)

    # Load scaffold split DataFrames
    sc_dir = Path(scaffold_split_dir) / "scenarios" / "Sc"

    def load_split(split_name: str) -> pd.DataFrame | None:
        for suffix in [".tsv.gz", ".tsv"]:
            path = sc_dir / f"universal_{split_name}{suffix}"
            if path.exists():
                df = pd.read_csv(str(path), sep="\t", compression="gzip" if str(path).endswith(".gz") else None)
                # Filter by dataset source if needed
                if dataset_type != "all" and "dataset_source" in df.columns:
                    df = df[df["dataset_source"] == dataset_type]
                if dataset_source_filter and "dataset_source" in df.columns:
                    df = df[df["dataset_source"] == dataset_source_filter]
                # Normalise SMILES column name
                if "smiles" not in df.columns and "canonical_smiles" in df.columns:
                    df = df.rename(columns={"canonical_smiles": "smiles"})
                # Compute label from pchembl_value if missing
                if "label" not in df.columns and "pchembl_value" in df.columns:
                    from benchmark.config import PCHEMBL_ACTIVITY_THRESHOLD
                    df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)
                # Ensure required columns
                required = {"seq_id", "chembl_id", "smiles", "label"}
                missing = required - set(df.columns)
                if missing:
                    tqdm.write(f"    WARNING: Missing columns {missing} in {path}")
                    return None
                return df
        return None

    train_df = load_split("train")
    val_df = load_split("val")

    if train_df is None or val_df is None:
        raise FileNotFoundError(
            f"Could not load scaffold split TSVs from {sc_dir}. "
            "Ensure universal_train.tsv[.gz] and universal_val.tsv[.gz] exist "
            "and contain columns: seq_id, chembl_id, smiles (or canonical_smiles), label (or pchembl_value)."
        )

    train_ds = LoRAMatrixDataset(train_df, protein_dirs)
    val_ds = LoRAMatrixDataset(val_df, protein_dirs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_lora,
        num_workers=0,  # MoLFormer tokeniser not pickle-friendly
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_lora,
        num_workers=0,
        pin_memory=True,
    )

    # Test loader (for test mode)
    test_loader = None
    if mode == "test":
        # Load test split from parent dir (not in scenarios/Sc)
        test_dir = Path(scaffold_split_dir)
        test_df = None
        for suffix in [".tsv.gz", ".tsv"]:
            path = test_dir / f"{dataset_type}_test{suffix}"
            if path.exists():
                test_df = pd.read_csv(str(path), sep="\t", compression="gzip" if str(path).endswith(".gz") else None)
                break
        if test_df is not None:
            test_ds = LoRAMatrixDataset(test_df, protein_dirs)
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_lora,
                num_workers=0,
                pin_memory=True,
            )

    return train_loader, val_loader, test_loader


# ======================================================================
# Level Runner
# ======================================================================


class Level4LoRARunner(BaseLevelRunner):
    """LoRA-adapted MoLFormer + Attention Pooling → MLP."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)

    @property
    def knn_is_deterministic(self) -> bool:
        return False

    @property
    def level_tag(self) -> str:
        return "level4_lora"

    def run_single_seed(
        self,
        seed: int,
        output_dir: str,
        **kwargs: object,
    ) -> Optional[Dict]:
        os.makedirs(output_dir, exist_ok=True)

        cache_path = os.path.join(output_dir, "level4_lora_mlp_results.json")
        if os.path.exists(cache_path) and not self.force:
            tqdm.write(f"  Loading cached Level 4 LoRA results (seed {seed})")
            with open(cache_path) as fh:
                return json.load(fh)

        # --- Resolve protein dim ---
        full_emb = SUPPORTED_EMBEDDINGS.get(self.embedding_name, self.embedding_name)
        protein_dim = PROTEIN_DIMS.get(full_emb, 320)

        # --- LoRA hyperparameters ---
        lora_rank = int(os.getenv("BENCHMARK_LEVEL4_LORA_RANK", "4"))
        lora_alpha = int(os.getenv("BENCHMARK_LEVEL4_LORA_ALPHA", "16"))

        # --- Build dataloaders ---
        tqdm.write(f"  Building Level 4 LoRA dataloaders (seed {seed})...")
        train_loader, val_loader, test_loader = _build_lora_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=self._config.batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self._config.mode,
        )

        # Split for feature extraction (inline because split_loader_for_feature_extraction
        # hardcodes collate_matrices, but we need _collate_lora)
        if self.mode == "train":
            from torch.utils.data import Subset
            ds = train_loader.dataset
            n = len(ds)
            if hasattr(ds, '_df') and 'label' in ds._df.columns:
                labels = ds._df['label'].to_numpy(dtype=int)
            else:
                labels = np.array([ds[i][2] for i in range(n)])
            rng = np.random.RandomState(seed)
            idx_pos = np.where(labels == 1)[0]
            idx_neg = np.where(labels == 0)[0]
            rng.shuffle(idx_pos); rng.shuffle(idx_neg)
            sp, sn = int(len(idx_pos) * 0.8), int(len(idx_neg) * 0.8)
            model_idx = np.concatenate([idx_pos[:sp], idx_neg[:sn]])
            feat_idx = np.concatenate([idx_pos[sp:], idx_neg[sn:]])
            rng.shuffle(model_idx); rng.shuffle(feat_idx)
            bs = train_loader.batch_size or 32
            model_train_loader = DataLoader(
                Subset(ds, model_idx.tolist()), batch_size=bs, shuffle=True,
                collate_fn=_collate_lora, num_workers=0, pin_memory=True,
            )
            feat_extract_loader = DataLoader(
                Subset(ds, feat_idx.tolist()), batch_size=bs, shuffle=False,
                collate_fn=_collate_lora, num_workers=0, pin_memory=True,
            )
        else:
            model_train_loader = train_loader

        local_selection_metric = os.getenv(
            "BENCHMARK_LEVEL4_SELECTION_METRIC",
            self._config.model_selection_metric,
        ).strip().lower()
        if local_selection_metric not in {"val_loss", "mcc", "downstream_mcc"}:
            local_selection_metric = "downstream_mcc"

        # --- Train ---
        tqdm.write(f"  Training Level 4 LoRA (seed {seed})...")
        model, aux_head = _train_lora_attention_pooling(
            train_loader=model_train_loader,
            val_loader=val_loader,
            downstream_fit_loader=None,
            protein_dim=protein_dim,
            lr=self._config.learning_rate,
            epochs=self._config.epochs,
            patience=self._config.resolved_patience or 10,
            seed=seed,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            model_selection_metric=local_selection_metric,
        )

        use_aux_channel = os.getenv("BENCHMARK_LEVEL4_USE_AUX_CHANNEL", "1").strip().lower() not in {
            "0", "false", "no",
        }

        device = next(model.parameters()).device

        # --- Extract features ---
        if self.mode == "train":
            tqdm.write("  Extracting LoRA-adapted features (held-out train + val)...")
            x_fit, y_fit = _extract_lora_features(
                model, aux_head, feat_extract_loader, device,
                desc="    Feature extraction (fit)",
                include_aux_channel=use_aux_channel,
            )
            x_eval, y_eval = _extract_lora_features(
                model, aux_head, val_loader, device,
                desc="    Feature extraction (eval)",
                include_aux_channel=use_aux_channel,
            )
        else:
            tqdm.write("  Extracting LoRA-adapted features (val + test)...")
            x_fit, y_fit = _extract_lora_features(
                model, aux_head, val_loader, device,
                desc="    Feature extraction (fit)",
                include_aux_channel=use_aux_channel,
            )
            if test_loader is not None:
                x_eval, y_eval = _extract_lora_features(
                    model, aux_head, test_loader, device,
                    desc="    Feature extraction (eval)",
                    include_aux_channel=use_aux_channel,
                )
            else:
                tqdm.write("  WARNING: No test loader available.")
                x_eval, y_eval = x_fit, y_fit

        # Sanitise
        for name, arr in [("fit", x_fit), ("eval", x_eval)]:
            bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
            if bad:
                tqdm.write(f"  WARNING: {name} has {bad} NaN/Inf values -> replaced with 0")
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Train MLP ---
        tqdm.write("  Training MLP only (KNN skipped)...")
        mlp_metrics = train_mlp_only(x_fit, y_fit, x_eval, y_eval, seed)

        sc_key = "Split by Scaffold"
        result = {sc_key: {"MLP": mlp_metrics}}

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(f"  Level 4 LoRA (seed {seed}): MLP MCC={mlp_metrics['mcc']:.4f}")
        return result
