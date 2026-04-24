"""Level 4 CNN v8 — Multi-Source Enrichment.

Extends InteractionMapCNN (v7) with external features that carry
information beyond what is derivable from the SMILES / amino-acid
sequence alone:

  Ligand  (concat globally, 100% coverage)
    - ChemBERTa-77M-MTR attention-pooled  (384-d)  [supervised MoleculeNet]
    - ADMET-AI 41-d                                [experimental TDC]
    - ClassyFire Superclass+Class one-hot          [curated taxonomy]

  Protein (concat globally, >=95% coverage)
    - BioBERT attention-pooled (768-d) over UniProt functional text
    - Pfam domain OHE (top-K kinome-relevant)
    - Taxonomy lineage OHE (NCBI)

Fusion:
  (a) Pre-attention token injection — ChemBERTa/BioBERT per-token caches
      are attention-pooled to a single vector per modality, projected to
      the backbone dimension (320 protein, 768 ligand), and concatenated
      as an extra token BEFORE the respective embedding adapter.
  (b) Post-pool concat — ADMET, ClassyFire, Pfam, Taxonomy are concatenated
      to the hierarchical-attention-pooled vector before the final classifier.

All pooling uses attention pooling with a single learnable query
(never mean pooling, never CLS).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from benchmark.levels.level4_cnn import (
    InteractionMapCNN,
    FocalLoss,
    _best_mcc_threshold,
    _platt_calibrate,
    _temperature_calibrate,
    _evaluate,
    _save_training_checkpoint,
)
from benchmark.levels.matrix_utils import (
    MatrixDataset,
    _loader_kwargs,
    _resolve_matrix_dirs,
    read_split_file,
    _validate_matrix_coverage,
    collate_matrices,
    PCHEMBL_ACTIVITY_THRESHOLD,
)


# ======================================================================
# Attention pooling primitive (replaces mean/CLS pooling everywhere in v8)
# ======================================================================


class AttentionPool1D(nn.Module):
    """Attention pooling with a single learnable query.

    Reduces a (B, L, D) sequence to a (B, D) vector via softmax-weighted
    sum of token representations, where the weights come from a
    learnable query attending to each token.

    Args:
        dim:    token embedding dimensionality (D).
        hidden: optional hidden size for the key projection. Defaults to D.
    """

    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        h = hidden or dim
        self.key = nn.Linear(dim, h)
        self.query = nn.Parameter(torch.empty(h))
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L, D); mask: (B, L) with 1=valid, 0=pad
        k = torch.tanh(self.key(tokens))
        scores = (k * self.query).sum(dim=-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)


# ======================================================================
# v8 model: InteractionMapCNN + multi-source injection
# ======================================================================


class InteractionMapCNNv8(InteractionMapCNN):
    """v8 model. Subclasses InteractionMapCNN and overrides forward.

    Accepts additional kwargs via forward. Missing features are treated
    as zero vectors (configurable via constructor flags), enabling
    ablation (v8-lig, v8-prot, v8-full) without separate classes.
    """

    def __init__(
        self,
        *args,
        enable_chemberta: bool = False,
        enable_admet: bool = False,
        enable_classyfire: bool = False,
        enable_biobert: bool = False,
        enable_pfam: bool = False,
        enable_taxonomy: bool = False,
        chemberta_dim: int = 384,
        biobert_dim: int = 768,
        admet_dim: int = 41,
        classyfire_dim: int = 230,
        pfam_dim: int = 50,
        taxonomy_dim: int = 10,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        cnn_channels: int = 64,
        **kwargs,
    ) -> None:
        # Parent uses protein_dim/ligand_dim/cnn_channels, forward them.
        super().__init__(
            *args,
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            cnn_channels=cnn_channels,
            **kwargs,
        )

        self.enable_chemberta = enable_chemberta
        self.enable_admet = enable_admet
        self.enable_classyfire = enable_classyfire
        self.enable_biobert = enable_biobert
        self.enable_pfam = enable_pfam
        self.enable_taxonomy = enable_taxonomy

        # ---- Pre-attention injection (attention-pooled + projection) ----
        if enable_chemberta:
            self.chemberta_pool = AttentionPool1D(chemberta_dim)
            self.chemberta_proj = nn.Sequential(
                nn.Linear(chemberta_dim, ligand_dim),
                nn.LayerNorm(ligand_dim),
            )
        if enable_biobert:
            self.biobert_pool = AttentionPool1D(biobert_dim)
            self.biobert_proj = nn.Sequential(
                nn.Linear(biobert_dim, protein_dim),
                nn.LayerNorm(protein_dim),
            )

        # ---- Post-pool concat ----
        extra_post = 0
        if enable_admet:
            extra_post += admet_dim
        if enable_classyfire:
            extra_post += classyfire_dim
        if enable_pfam:
            extra_post += pfam_dim
        if enable_taxonomy:
            extra_post += taxonomy_dim
        self.extra_post_dim = extra_post

        # Parent built self.classifier with clf_in = cnn_channels (+1 if cosine_feat).
        # Replace it to accept the extra post-pool features.
        if extra_post > 0:
            # Read parent's classifier input dim by probing state_dict
            base_in = cnn_channels + (1 if self.cosine_feat else 0)
            new_in = base_in + extra_post
            # Match v7 style: single Linear head (mlp_head=False default)
            if isinstance(self.classifier, nn.Sequential):
                # Parent built MLP head — rebuild respecting dropout/d_model
                # Find the last Linear to get its out dim
                out_dim = None
                dropout_p = 0.0
                for m in self.classifier.modules():
                    if isinstance(m, nn.Dropout):
                        dropout_p = m.p
                    if isinstance(m, nn.Linear):
                        d_model = m.out_features  # last wins; classifier ends with Linear(_, 1)
                        out_dim = m.out_features
                # Rebuild an MLP head matching the pattern [Linear(in, d_model), GELU, Dropout, Linear(d_model, 1)]
                # but d_model here is 1 on the last layer; we need the hidden dim of the first Linear.
                first_linear = next(m for m in self.classifier.modules() if isinstance(m, nn.Linear))
                hidden = first_linear.out_features
                self.classifier = nn.Sequential(
                    nn.Linear(new_in, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden, 1),
                )
                for m in self.classifier.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            else:
                self.classifier = nn.Linear(new_in, 1)
                nn.init.xavier_uniform_(self.classifier.weight)
                nn.init.zeros_(self.classifier.bias)

    def forward(  # type: ignore[override]
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        chemberta_tokens: Optional[torch.Tensor] = None,
        chemberta_mask: Optional[torch.Tensor] = None,
        biobert_tokens: Optional[torch.Tensor] = None,
        biobert_mask: Optional[torch.Tensor] = None,
        admet: Optional[torch.Tensor] = None,
        classyfire: Optional[torch.Tensor] = None,
        pfam: Optional[torch.Tensor] = None,
        taxonomy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Model dtype drives all non-mask extras (feature tensors come in as
        # float32 from the .npy cache; when the model is configured with
        # `double: true` in v7.yaml, weights are float64 and we must cast
        # features to match — otherwise F.linear raises dtype mismatch).
        _dtype = next(self.parameters()).dtype
        if chemberta_tokens is not None and chemberta_tokens.dtype != _dtype:
            chemberta_tokens = chemberta_tokens.to(_dtype)
        if biobert_tokens is not None and biobert_tokens.dtype != _dtype:
            biobert_tokens = biobert_tokens.to(_dtype)
        if admet is not None and admet.dtype != _dtype:
            admet = admet.to(_dtype)
        if classyfire is not None and classyfire.dtype != _dtype:
            classyfire = classyfire.to(_dtype)
        if pfam is not None and pfam.dtype != _dtype:
            pfam = pfam.to(_dtype)
        if taxonomy is not None and taxonomy.dtype != _dtype:
            taxonomy = taxonomy.to(_dtype)

        # ---- Pre-attention token injection ----
        if self.enable_chemberta and chemberta_tokens is not None:
            cb_vec = self.chemberta_pool(chemberta_tokens, chemberta_mask)
            cb_vec = self.chemberta_proj(cb_vec)  # (B, ligand_dim)
            ligand_matrix = torch.cat([cb_vec.unsqueeze(1), ligand_matrix], dim=1)
            ligand_mask = torch.cat(
                [torch.ones(ligand_mask.size(0), 1, dtype=ligand_mask.dtype, device=ligand_mask.device),
                 ligand_mask], dim=1,
            )
        if self.enable_biobert and biobert_tokens is not None:
            bb_vec = self.biobert_pool(biobert_tokens, biobert_mask)
            bb_vec = self.biobert_proj(bb_vec)  # (B, protein_dim)
            protein_matrix = torch.cat([bb_vec.unsqueeze(1), protein_matrix], dim=1)
            protein_mask = torch.cat(
                [torch.ones(protein_mask.size(0), 1, dtype=protein_mask.dtype, device=protein_mask.device),
                 protein_mask], dim=1,
            )

        # ---- Run parent's forward up to the pooled vector ----
        # Replicate parent's logic, stop before self.classifier, append extras.

        if self.use_adapter:
            protein_matrix = self.prot_adapter(protein_matrix)
            ligand_matrix = self.lig_adapter(ligand_matrix)

        if self.contrastive_dim > 0 or self.cosine_feat:
            _pm = protein_mask.unsqueeze(-1).float()
            _lm = ligand_mask.unsqueeze(-1).float()
            _prot_mean = (protein_matrix * _pm).sum(1) / _pm.sum(1).clamp(min=1)
            _lig_mean = (ligand_matrix * _lm).sum(1) / _lm.sum(1).clamp(min=1)
            if self.contrastive_dim > 0:
                self._z_prot = F.normalize(self.prot_contrast_proj(_prot_mean), dim=-1)
                self._z_lig = F.normalize(self.lig_contrast_proj(_lig_mean), dim=-1)
            if self.cosine_feat:
                if self.contrastive_dim > 0:
                    p_proj = self._z_prot
                    l_proj = self._z_lig
                else:
                    p_proj = self.prot_cos_proj(_prot_mean)
                    l_proj = self.lig_cos_proj(_lig_mean)
                self._cos_sim_feat = F.cosine_similarity(p_proj, l_proj, dim=-1).unsqueeze(1)

        # v7 / v7_gated only (v8 variant = v7 in configs/v8.yaml)
        p_feat = protein_matrix
        l_feat = ligand_matrix
        if self.variant == "v7_gated":
            p_feat = protein_matrix * self.prot_gate(protein_matrix)
            l_feat = ligand_matrix * self.lig_gate(ligand_matrix)

        maps = []
        for ph, lh in zip(self.prot_heads, self.lig_heads):
            p = ph(p_feat)
            l = lh(l_feat)
            if self.cosine_sim:
                p = F.normalize(p, dim=-1)
                l = F.normalize(l, dim=-1)
            m = torch.bmm(p, l.transpose(1, 2)) * self.scale
            maps.append(m)
        interaction = torch.stack(maps, dim=1)

        mask_2d = protein_mask.unsqueeze(2) * ligand_mask.unsqueeze(1)
        interaction = interaction * mask_2d.unsqueeze(1)

        features = self.cnn(interaction)
        features = features * mask_2d.unsqueeze(1)

        pooled = self.pool(features, protein_mask, ligand_mask)
        pooled = self.dropout(pooled)

        if self.cosine_feat and self._cos_sim_feat is not None:
            pooled = torch.cat([pooled, self._cos_sim_feat], dim=-1)

        # ---- Post-pool concat ----
        extras = []
        if self.enable_admet and admet is not None:
            extras.append(admet)
        if self.enable_classyfire and classyfire is not None:
            extras.append(classyfire)
        if self.enable_pfam and pfam is not None:
            extras.append(pfam)
        if self.enable_taxonomy and taxonomy is not None:
            extras.append(taxonomy)
        if extras:
            pooled = torch.cat([pooled] + extras, dim=-1)

        logits = self.classifier(pooled)
        return logits


# ======================================================================
# Dataset + collate that loads v7 matrices + v8 caches together
# ======================================================================


class MatrixDatasetV8(Dataset):
    """Wraps v7 MatrixDataset and optionally loads per-sample v8 caches.

    MatrixDataset.__getitem__ returns a 5-tuple
    (protein_mat, ligand_mat, label, seq_id, chembl_id). We re-wrap into
    a dict that also carries the requested v8 feature tensors.

    Cache layout (per corpus):
      data/embeddings/v8/{feature}_{corpus}/{key}.npy
    where {key} is chembl_id (ligand) or seq_id (protein).

    mmap_mode="r" is used for large per-token caches (ChemBERTa/BioBERT)
    to avoid loading the entire cache into RAM. We copy into a writable
    contiguous array before `torch.from_numpy` to silence PyTorch's
    non-writable-tensor warning.
    """

    PER_TOKEN_FEATURES = {"chemberta", "biobert"}

    def __init__(
        self,
        base: MatrixDataset,
        cache_root: Optional[Path],
        corpus: str,
        enabled: dict[str, bool],
    ) -> None:
        self.base = base
        self.enabled = enabled
        self.cache_dirs: dict[str, Path] = {}
        if cache_root is not None:
            for feature in ("chemberta", "admet", "classyfire", "biobert", "pfam", "taxonomy"):
                if enabled.get(feature, False):
                    self.cache_dirs[feature] = cache_root / f"{feature}_{corpus}"

    def __len__(self) -> int:
        return len(self.base)

    def _load_feature(self, feature: str, key: str) -> np.ndarray:
        path = self.cache_dirs[feature] / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"v8 cache missing: {path}")
        if feature in self.PER_TOKEN_FEATURES:
            # mmap'd read → force WRITABLE copy (np.array with copy=True).
            # np.ascontiguousarray with matching dtype returns a view that
            # inherits the mmap's read-only flag, which PyTorch warns about.
            arr = np.load(path, mmap_mode="r")
            return np.array(arr, dtype=np.float32, copy=True)
        return np.load(path).astype(np.float32, copy=True)

    def __getitem__(self, idx: int) -> dict:
        # Base dataset returns a 5-tuple
        protein_mat, ligand_mat, label, seq_id, chembl_id = self.base[idx]
        chembl_id = str(chembl_id)
        seq_id = str(seq_id)
        sample: dict = {
            "protein_mat": protein_mat,
            "ligand_mat": ligand_mat,
            "label": label,
            "seq_id": seq_id,
            "chembl_id": chembl_id,
        }
        if "chemberta" in self.cache_dirs:
            sample["chemberta_tokens"] = torch.from_numpy(self._load_feature("chemberta", chembl_id))
        if "admet" in self.cache_dirs:
            sample["admet"] = torch.from_numpy(self._load_feature("admet", chembl_id))
        if "classyfire" in self.cache_dirs:
            sample["classyfire"] = torch.from_numpy(self._load_feature("classyfire", chembl_id))
        if "biobert" in self.cache_dirs:
            sample["biobert_tokens"] = torch.from_numpy(self._load_feature("biobert", seq_id))
        if "pfam" in self.cache_dirs:
            sample["pfam"] = torch.from_numpy(self._load_feature("pfam", seq_id))
        if "taxonomy" in self.cache_dirs:
            sample["taxonomy"] = torch.from_numpy(self._load_feature("taxonomy", seq_id))
        return sample


def _pad_token_batch(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length (L, D) tensors into (B, L_max, D) with mask."""
    max_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.float32)
    for i, t in enumerate(tensors):
        L = t.shape[0]
        padded[i, :L] = t
        mask[i, :L] = 1.0
    return padded, mask


def collate_v8(batch: list[dict]) -> dict:
    """Collate dict samples from MatrixDatasetV8.

    Replicates the v7 collate behavior on (protein_mat, ligand_mat, label)
    using pad_matrices, then stacks v8 fixed-size features and pads
    per-token features along the sequence axis with mask.
    """
    from benchmark.levels.matrix_utils import pad_matrices  # local import

    protein_mats = [s["protein_mat"] for s in batch]
    ligand_mats = [s["ligand_mat"] for s in batch]
    labels = [s["label"] for s in batch]
    seq_ids = [s["seq_id"] for s in batch]
    chembl_ids = [s["chembl_id"] for s in batch]

    protein_batch, protein_mask = pad_matrices(protein_mats)
    ligand_batch, ligand_mask = pad_matrices(ligand_mats)

    out = {
        "protein_matrix": torch.from_numpy(protein_batch),
        "ligand_matrix": torch.from_numpy(ligand_batch),
        "protein_mask": torch.from_numpy(protein_mask),
        "ligand_mask": torch.from_numpy(ligand_mask),
        "label": torch.tensor(labels, dtype=torch.float32),
        "seq_id": seq_ids,
        "chembl_id": chembl_ids,
    }

    # Per-token v8 features
    for key, mask_key in (("chemberta_tokens", "chemberta_mask"),
                         ("biobert_tokens", "biobert_mask")):
        if key in batch[0]:
            padded, mask = _pad_token_batch([s[key] for s in batch])
            out[key] = padded
            out[mask_key] = mask
    # Fixed-size v8 features
    for key in ("admet", "classyfire", "pfam", "taxonomy"):
        if key in batch[0]:
            out[key] = torch.stack([s[key] for s in batch], dim=0)
    return out


# ======================================================================
# Dataloader builder (mirrors build_matrix_dataloaders but with v8 dataset)
# ======================================================================


def build_v8_dataloaders(
    dataset_type: str,
    embedding_name: str,
    scaffold_split_dir: str,
    batch_size: int,
    enabled: dict[str, bool],
    cache_root: Path,
    dataset_source_filter: Optional[str] = None,
    cache_corpus: Optional[str] = None,
    mode: str = "test",
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """dataset_type controls protein/ligand matrix directory resolution
    ('all' searches both Human and Non-Human .npy dirs). cache_corpus
    controls which v8 cache directory is read ('non_human' for
    precompute_*_ligand.py --corpus non_human). Defaults to dataset_type.
    """
    cache_c = cache_corpus or dataset_type
    protein_dirs, ligand_dirs = _resolve_matrix_dirs(dataset_type, embedding_name)

    train_df = read_split_file(os.path.join(scaffold_split_dir, "scenarios/Sc", "universal_train.tsv"))
    val_df = read_split_file(os.path.join(scaffold_split_dir, "scenarios/Sc", "universal_val.tsv"))
    test_df = read_split_file(os.path.join(scaffold_split_dir, "universal_test.tsv")) if mode == "test" else None

    if dataset_source_filter is not None:
        train_df = train_df[train_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)
        val_df = val_df[val_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)
        if test_df is not None:
            test_df = test_df[test_df["dataset_source"] == dataset_source_filter].reset_index(drop=True)

    for df in (train_df, val_df, test_df):
        if df is not None and "label" not in df.columns:
            df["label"] = (df["pchembl_value"] >= PCHEMBL_ACTIVITY_THRESHOLD).astype(int)

    def _make(df, shuffle):
        _validate_matrix_coverage(df, protein_dirs, ligand_dirs)
        base = MatrixDataset(df, protein_dirs, ligand_dirs)
        v8 = MatrixDatasetV8(base, cache_root=cache_root, corpus=cache_c, enabled=enabled)
        return DataLoader(
            v8, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_v8, **_loader_kwargs(),
        )

    train_loader = _make(train_df, shuffle=True)
    val_loader = _make(val_df, shuffle=False)
    test_loader = _make(test_df, shuffle=False) if test_df is not None else None
    return train_loader, val_loader, test_loader
