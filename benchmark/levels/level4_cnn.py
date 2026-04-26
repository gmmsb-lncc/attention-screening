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

Scope note: this is the canonical DT-Kinase v7 reference path and is
pure classification (single logit + Focal/BCE loss). Multi-task
regression heads (MultiTaskLoss, regression_head) exist in separate
modules under src/classifier/ and src/attention_matrix/ for exploratory
variants only; they are NOT used by the thesis benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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

import torch.nn.functional as F


# ======================================================================
# Focal Loss — penalises easy examples, focuses on hard negatives
# ======================================================================

class FocalLoss(nn.Module):
    """Focal Loss for binary classification (from logits).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma=0 this reduces to standard BCE.
    Higher gamma (e.g. 2) down-weights easy examples and focuses
    learning on hard, misclassified examples (false positives).

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha      # per-sample alpha (None = use pos_weight)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE without reduction
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Class weighting (like pos_weight in BCE)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.pos_weight * targets + 1.0 * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Margin-based contrastive loss for drug-target co-embedding.

    Inspired by ConPLex (Sledzieski et al., PNAS 2023).
    Positive pairs (binding): maximize cosine similarity.
    Negative pairs (non-binding): push cosine similarity below margin.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z_prot: torch.Tensor,
        z_lig: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(z_prot, z_lig, dim=-1)
        labels = labels.view(-1).float()
        pos_loss = labels * (1 - cos_sim).pow(2)
        neg_loss = (1 - labels) * F.relu(cos_sim - self.margin).pow(2)
        return (pos_loss + neg_loss).mean()


# ======================================================================
# Model components
# ======================================================================


class _GradientReversalFn(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, ICML 2015).

    Forward: identity. Backward: multiplies gradient by -lambd.
    Used to flip gradient sign on the domain branch so the encoder
    learns to PRODUCE features that confuse the domain classifier
    (i.e., domain-invariant features).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class DomainAdversarialHead(nn.Module):
    """Small MLP that predicts domain (corpus) from pooled features.

    The encoder side passes through a Gradient Reversal Layer, so
    optimising the domain CE loss makes the encoder learn features
    that are INDISTINGUISHABLE between corpora — encouraging
    cross-corpus transferability (CDAN-style, Long et al. 2018).

    Designed to be plugged onto the HierPool output (~64-d vector)
    of InteractionMapCNN. Predicts one of {human, non_human, all}
    by default (3 classes).
    """

    def __init__(self, in_dim: int, n_domains: int = 3, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        x_rev = _GradientReversalFn.apply(x, lambd)
        return self.head(x_rev)


class _AxisAttentionPool(nn.Module):
    """Learnable-query attention pool along seq dimension.

    Input:  [B, L, D]
    Output: [B, D]

    Uses dot-product attention with a learnable query vector.
    Mask convention: ``True = padding`` (will be ignored).
    """

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Multi-head learnable queries. Each head learns to attend to a
        # different aspect of the input sequence (e.g. one head focuses on
        # high-affinity residues, another on aromatic substructures, etc.).
        # Single-head reduces to original v7 behaviour.
        self.queries = nn.Parameter(torch.randn(1, num_heads, dim) * 0.02)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        # When multi-head, concat outputs (H × D) and project back to D.
        # Identity in single-head mode keeps backward compatibility.
        if num_heads > 1:
            self.head_proj = nn.Linear(num_heads * dim, dim)
        else:
            self.head_proj = nn.Identity()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool [B, L, D] → [B, D]."""
        B = x.size(0)
        q = self.queries.expand(B, -1, -1)                # [B, H, D]
        scores = torch.bmm(q, x.transpose(1, 2)) * self.scale  # [B, H, L]

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)                # [B, H, L]
        pooled = torch.bmm(attn, x)                         # [B, H, D]
        if self.num_heads > 1:
            pooled = pooled.reshape(B, -1)                   # [B, H*D]
            pooled = self.head_proj(pooled)                  # [B, D]
        else:
            pooled = pooled.squeeze(1)                       # [B, D]
        return self.norm(pooled)                              # [B, D]


class _HierarchicalPool(nn.Module):
    """Pool 2D CNN output [B, C, H, W] → [B, C] via two-stage attention.

    Stage 1 — ligand axis:  for each protein position h, pool across
              W (ligand) to get a single vector → [B, H, C].
    Stage 2 — protein axis: pool across H (protein) → [B, C].

    This lets the model learn:
      (1) which ligand atoms matter for each protein residue
      (2) which protein residues matter overall
    """

    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.lig_pool = _AxisAttentionPool(channels, num_heads=num_heads)
        self.prot_pool = _AxisAttentionPool(channels, num_heads=num_heads)

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


class EmbeddingAdapter(nn.Module):
    """Adapter to refine frozen embeddings for a specific task.

    Supports:
      - Stacked MLP bottleneck blocks (num_layers ≥ 1)
      - Optional per-residue self-attention before the MLP blocks

    Each MLP block:  Linear → GELU → Dropout → Linear + Skip → LayerNorm
    Self-attention:  MultiheadAttention(4 heads) + Skip → LayerNorm

    References:
      Houlsby et al., "Parameter-Efficient Transfer Learning", ICML 2019.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        dropout: float = 0.3,
        num_layers: int = 1,
        use_self_attn: bool = False,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        # Legacy flag inverted to default ON after lição 17 §6.9.1 isolation
        # confirmed §6.5 (pre-norm + LoRA gates + zero-init self_attn) regresses
        # v7+F by -0.053 MCC in short-training regime. Setting LEGACY=0 opts in
        # to §6.5 fixes (use only with longer training / additional stochastic
        # regularisation that breaks the zero-cascade trap).
        # Default ("1") reproduces v7+F = 0.5266 ± 0.010 on diamante-01.
        self._legacy = os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY", "1") == "1"

        # --- Optional self-attention (context-aware adaptation) --------
        self.use_self_attn = use_self_attn
        if use_self_attn:
            assert dim % num_heads == 0, (
                f"adapter num_heads={num_heads} must divide dim={dim}")
            self.self_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True,
            )
            if not self._legacy:
                # §6.5: zero-init out_proj so attn_out = 0 at t=0
                nn.init.zeros_(self.self_attn.out_proj.weight)
                nn.init.zeros_(self.self_attn.out_proj.bias)
            # else: leave PyTorch default (Xavier-uniform) — pre-§6.5
            self.attn_norm = nn.LayerNorm(dim)
            if not self._legacy:
                # §6.5: LoRA-style scalar gate, init zero
                self.attn_scale = nn.Parameter(torch.zeros(1))

        # --- Stacked MLP bottleneck blocks ----------------------------
        self.mlp_blocks = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        if not self._legacy:
            self.mlp_scales = nn.ParameterList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(dim, bottleneck),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, dim),
            )
            # Zero-init output layer → starts as identity (kept in BOTH
            # regimes — predates §6.5, lesson 2)
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            self.mlp_blocks.append(block)
            self.mlp_norms.append(nn.LayerNorm(dim))
            if not self._legacy:
                # §6.5: per-block LoRA scalar gate, init zero
                self.mlp_scales.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._legacy:
            # Pre-§6.5: post-norm residual without scalar gates.
            #   self-attn:  x = LayerNorm(x + attn_out)
            #   MLP block:  x = LayerNorm(x + block(x))
            if self.use_self_attn:
                attn_out, _ = self.self_attn(x, x, x)
                x = self.attn_norm(x + attn_out)
            for block, norm in zip(self.mlp_blocks, self.mlp_norms):
                x = norm(x + block(x))
            return x

        # §6.5: pre-norm with LoRA scalar gates (Xiong et al., 2020).
        if self.use_self_attn:
            x_n = self.attn_norm(x)
            attn_out, _ = self.self_attn(x_n, x_n, x_n)
            x = x + self.attn_scale * attn_out
        for block, norm, scale in zip(
            self.mlp_blocks, self.mlp_norms, self.mlp_scales
        ):
            x = x + scale * block(norm(x))
        return x


def apply_rope_1d(x: torch.Tensor) -> torch.Tensor:
    """Apply 1D Rotary Positional Embedding to the last dim of x [B, L, D].

    Standard RoPE (Su et al. 2021, RoFormer): for each position p ∈ [0, L),
    rotate consecutive pair (x[2i], x[2i+1]) by angle p · θ_i where
    θ_i = 10000^(-2i/D). After rotation, the dot product q·k between two
    positions encodes only their relative offset, injecting positional
    structure without learnable parameters.

    Used per-modality (protein and ligand independently) before the
    cross-modal dot product M_k = P_proj_k · L_proj_k^T, which becomes
    aware of each position's location along its own sequence axis.
    head_dim must be even.
    """
    B, L, D = x.shape
    half = D // 2
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, half, dtype=torch.float32, device=x.device) / half
    ))
    pos = torch.arange(L, dtype=torch.float32, device=x.device)
    freqs = torch.outer(pos, inv_freq)                # [L, half]
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    x_first, x_second = x[..., :half], x[..., half:]
    rotated_first = x_first * cos - x_second * sin
    rotated_second = x_first * sin + x_second * cos
    return torch.cat([rotated_first, rotated_second], dim=-1)


class InteractionMapCNN(nn.Module):
    """2D CNN on protein–ligand interaction maps.

    Architecture variants:
      v7 (original): K pairs of linear projections (prot_dim→head_dim,
          lig_dim→head_dim) then dot-product → K interaction maps.
      v7_gated:      Same as v7, but applies an MLP-based Sigmoid gate
          to the original embeddings before projection to filter out noise.
      v8 (BAN):      Full Bilinear Attention Network — each head has a
          weight matrix W_k [prot_dim, lig_dim] that computes interaction
          directly in the original embedding spaces without any projection
          bottleneck.  Preserves 100% of ligand information.
      v9 (CrossAttn): Bidirectional Cross-Attention — protein attends to
          ligand and vice-versa through stacked Transformer-style layers.
          Uses attention pooling + MLP classifier (no CNN).
      v10 (CrossAttn+CNN): Same cross-attention as v9, but creates
          enriched interaction maps fed into the CNN+HierPool pipeline.
          Combines long-range awareness with spatial pattern detection.

    Pipeline (v7/v7_gated/v8):
      1. Interaction maps           [B, K, seq_p, seq_l]
      2. 4-layer 2D CNN with dilated convolution
      3. Hierarchical attention pooling
      4. Linear classifier → BCEWithLogitsLoss

    Pipeline (v9):
      1. Project protein & ligand to shared d_model space
      2. Bidirectional cross-attention (N layers)
      3. Attention pooling per modality
      4. MLP classifier → BCEWithLogitsLoss

    Pipeline (v10):
      1. Project protein & ligand to shared d_model space
      2. Bidirectional cross-attention (N layers)
      3. Multi-head dot-product → enriched interaction maps
      4. 4-layer 2D CNN with dilated convolution
      5. Hierarchical attention pooling
      6. Linear classifier → BCEWithLogitsLoss
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
        num_cross_layers: int = 2,
        mlp_head: bool = False,
        cosine_sim: bool = False,
        use_adapter: bool = False,
        adapter_bottleneck_prot: int = 256,
        adapter_bottleneck_lig: int = 512,
        adapter_layers: int = 1,
        adapter_self_attn: bool = False,
        contrastive_dim: int = 128,
        cosine_feat: bool = False,
        pool_num_heads: int = 1,
        # Asymmetric adapter knobs. None → fall back to symmetric values
        # (adapter_layers, num_heads=4) to preserve backward compatibility.
        # Used to give the ligand branch more capacity than the protein
        # branch, motivated by kinase-domain conservation: ATP-binding
        # pockets are highly similar across kinases, so most discriminative
        # signal lives on the ligand side.
        adapter_layers_prot: int | None = None,
        adapter_layers_lig: int | None = None,
        adapter_attn_heads_prot: int = 4,
        adapter_attn_heads_lig: int = 4,
        # BAN-residual (lição 12 reformulação): adds, on top of v7's K
        # dot-product heads, a residual term M_k += α_k · P W_k Lᵀ with
        # α_k init zero and W_k Xavier. At t=0 the model is identical to
        # v7 (gate=0); gradient activates the bilinear term where useful.
        # Avoids the §6.5 zero-cascade trap because gradient ∂L/∂α
        # depends on (P W_k Lᵀ) which is non-zero at t=0 (W_k Xavier).
        use_ban_residual: bool = False,
        # 2D RoPE (Su et al. 2021): applies 1D Rotary Positional Embedding
        # per-modality (protein and ligand independently) to the head
        # projections before the cross-modal dot product. Injects relative
        # position structure into M_k without learnable parameters. NOTE:
        # not identity-init (rotates from t=0); marginal positional inductive
        # bias for the CNN that processes M_k. head_dim must be even.
        use_rope: bool = False,
        # Morgan fingerprint topological auxiliary feature (DrugBAN/GraphBAN
        # GCN proxy). When morgan_n_bits > 0, model expects a `morgan_fp`
        # tensor of shape [B, n_bits] in forward() and projects it to a
        # `morgan_proj_dim`-d vector that is CONCATENATED to the HierPool
        # output before the classifier head.
        morgan_n_bits: int = 0,
        morgan_proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.num_heads = num_heads
        self.mlp_head = mlp_head
        self.cosine_sim = cosine_sim
        self.use_ban_residual = use_ban_residual and variant in ("v7", "v7_gated")
        self.use_rope = use_rope and variant in ("v7", "v7_gated")
        if self.use_rope:
            assert head_dim % 2 == 0, (
                f"use_rope=True requires even head_dim, got {head_dim}")
        self.use_adapter = use_adapter
        self.contrastive_dim = contrastive_dim
        self.cosine_feat = cosine_feat
        self.morgan_n_bits = int(morgan_n_bits) if morgan_n_bits else 0
        self.morgan_proj_dim = int(morgan_proj_dim) if morgan_proj_dim else 0
        if self.morgan_n_bits > 0 and self.morgan_proj_dim > 0:
            self.morgan_proj = nn.Sequential(
                nn.Linear(self.morgan_n_bits, self.morgan_proj_dim),
                nn.GELU(),
                nn.LayerNorm(self.morgan_proj_dim),
            )
        self._z_prot: torch.Tensor | None = None
        self._z_lig: torch.Tensor | None = None
        self._cos_sim_feat: torch.Tensor | None = None
        self._morgan_feat: torch.Tensor | None = None

        # --- Embedding adapters (optional, asymmetric capacity) -------
        # Resolve per-side overrides. None falls back to symmetric value.
        _prot_layers = adapter_layers if adapter_layers_prot is None else adapter_layers_prot
        _lig_layers  = adapter_layers if adapter_layers_lig  is None else adapter_layers_lig
        if use_adapter:
            self.prot_adapter = EmbeddingAdapter(
                dim=protein_dim,
                bottleneck=adapter_bottleneck_prot,
                dropout=dropout,
                num_layers=_prot_layers,
                use_self_attn=adapter_self_attn,
                num_heads=adapter_attn_heads_prot,
            )
            self.lig_adapter = EmbeddingAdapter(
                dim=ligand_dim,
                bottleneck=adapter_bottleneck_lig,
                dropout=dropout,
                num_layers=_lig_layers,
                use_self_attn=adapter_self_attn,
                num_heads=adapter_attn_heads_lig,
            )

        if variant in ("v7", "v7_gated"):
            # Original: project both sides to head_dim, then dot-product
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            
            if variant == "v7_gated":
                # MLP gates for non-linear local feature selection
                self.prot_gate = nn.Sequential(
                    nn.Linear(protein_dim, protein_dim // 2),
                    nn.GELU(),
                    nn.Linear(protein_dim // 2, protein_dim),
                    nn.Sigmoid()
                )
                self.lig_gate = nn.Sequential(
                    nn.Linear(ligand_dim, ligand_dim // 2),
                    nn.GELU(),
                    nn.Linear(ligand_dim // 2, ligand_dim),
                    nn.Sigmoid()
                )

            self.prot_heads = nn.ModuleList([
                nn.Linear(protein_dim, head_dim) for _ in range(num_heads)
            ])
            self.lig_heads = nn.ModuleList([
                nn.Linear(ligand_dim, head_dim) for _ in range(num_heads)
            ])
            # BAN-residual heads (lição 12 reformulação). One scalar gate
            # α_k per head (init zero) + one bilinear matrix W_k per head
            # (Xavier). Gradient flows from the start because P W_k Lᵀ ≠ 0
            # at t=0; α_k absorbs the contribution magnitude as training
            # progresses.
            if self.use_ban_residual:
                self.ban_alphas = nn.ParameterList([
                    nn.Parameter(torch.zeros(1)) for _ in range(num_heads)
                ])
                self.ban_weights = nn.ParameterList([
                    nn.Parameter(torch.empty(protein_dim, ligand_dim))
                    for _ in range(num_heads)
                ])
                # Xavier-uniform on each W_k for non-zero gradient on α_k.
                # Scale down by sqrt(ligand_dim) so the residual is in the
                # same numerical range as the dot-product term (which is
                # divided by sqrt(head_dim)).
                _ban_scale = (protein_dim * ligand_dim) ** -0.25
                for w in self.ban_weights:
                    nn.init.xavier_uniform_(w, gain=_ban_scale)
        elif variant == "v8":
            # Full Bilinear Attention: W_k[prot_dim, lig_dim] per head.
            # score(i,j) = protein[i] @ W_k @ ligand[j]
            # No projection bottleneck — uses full embeddings.
            self.head_dim = 0  # not used in v8
            self.scale = ligand_dim ** -0.5
            self.W_ban = nn.Parameter(
                torch.empty(num_heads, protein_dim, ligand_dim)
            )
        elif variant == "v9":
            # Pure Cross-Attention: bidirectional Transformer-style attention
            # with attention pooling + MLP classifier. No CNN.
            self.head_dim = head_dim  # reused as d_model
            d_model = head_dim

            self.prot_proj = nn.Linear(protein_dim, d_model)
            self.lig_proj  = nn.Linear(ligand_dim, d_model)

            self.num_cross_layers = num_cross_layers
            self.cross_prot_to_lig = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                      batch_first=True)
                for _ in range(num_cross_layers)
            ])
            self.cross_lig_to_prot = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                      batch_first=True)
                for _ in range(num_cross_layers)
            ])
            self.prot_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_cross_layers)
            ])
            self.lig_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_cross_layers)
            ])

            # Attention pooling: learn which tokens matter
            self.prot_pool_attn = nn.Linear(d_model, 1)
            self.lig_pool_attn  = nn.Linear(d_model, 1)

        elif variant == "v10":
            # Cross-Attention + CNN hybrid: cross-attention enriches
            # representations, then multi-head dot-product creates
            # interaction maps for the CNN+HierPool pipeline.
            self.head_dim = head_dim
            d_model = head_dim
            self.scale = head_dim ** -0.5

            self.prot_proj = nn.Linear(protein_dim, d_model)
            self.lig_proj  = nn.Linear(ligand_dim, d_model)

            self.num_cross_layers = num_cross_layers
            self.cross_prot_to_lig = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                      batch_first=True)
                for _ in range(num_cross_layers)
            ])
            self.cross_lig_to_prot = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                      batch_first=True)
                for _ in range(num_cross_layers)
            ])
            self.prot_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_cross_layers)
            ])
            self.lig_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_cross_layers)
            ])

            # Multi-head projections for interaction maps (post cross-attn)
            self.prot_heads = nn.ModuleList([
                nn.Linear(d_model, head_dim) for _ in range(num_heads)
            ])
            self.lig_heads = nn.ModuleList([
                nn.Linear(d_model, head_dim) for _ in range(num_heads)
            ])
        else:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose 'v7', 'v7_gated', 'v8', 'v9', or 'v10'."
            )

        if variant in ("v7", "v7_gated", "v8", "v10"):
            # 2D CNN on interaction maps [B, num_heads, seq_p, seq_l]
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
            self.pool = _HierarchicalPool(cnn_channels, num_heads=pool_num_heads)
            self.dropout = nn.Dropout(dropout)
            clf_in = cnn_channels + (1 if cosine_feat else 0) + (
                self.morgan_proj_dim if (self.morgan_n_bits > 0 and self.morgan_proj_dim > 0) else 0
            )
            if mlp_head:
                self.classifier = nn.Sequential(
                    nn.Linear(clf_in, cnn_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(cnn_channels * 2, 1),
                )
            else:
                self.classifier = nn.Linear(clf_in, 1)
        else:
            # v9: MLP classifier on concatenated pooled vectors
            d_model = head_dim
            self.dropout = nn.Dropout(dropout)
            clf_in_v9 = d_model * 2 + (1 if cosine_feat else 0)
            self.classifier = nn.Sequential(
                nn.Linear(clf_in_v9, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        # --- Contrastive projection heads (ConPLex-inspired) ----------
        if contrastive_dim > 0:
            self.prot_contrast_proj = nn.Sequential(
                nn.Linear(protein_dim, contrastive_dim),
                nn.ReLU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
            self.lig_contrast_proj = nn.Sequential(
                nn.Linear(ligand_dim, contrastive_dim),
                nn.ReLU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )

        # --- Cosine feature projection (shared space) -----------------
        if cosine_feat and contrastive_dim <= 0:
            # Need separate projections if no contrastive heads exist
            _cos_dim = 128
            self.prot_cos_proj = nn.Linear(protein_dim, _cos_dim)
            self.lig_cos_proj = nn.Linear(ligand_dim, _cos_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.variant in ("v7", "v7_gated"):
            for heads in (self.prot_heads, self.lig_heads):
                for h in heads:
                    nn.init.xavier_uniform_(h.weight)
                    nn.init.zeros_(h.bias)
            if self.variant == "v7_gated":
                for seq in (self.prot_gate, self.lig_gate):
                    for layer in seq:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight)
                            nn.init.zeros_(layer.bias)
        elif self.variant == "v8":
            # Xavier uniform on each head's W_k[prot_dim, lig_dim]
            for k in range(self.num_heads):
                nn.init.xavier_uniform_(self.W_ban[k])
        elif self.variant == "v9":
            # Init projections and pooling attention
            for lin in (self.prot_proj, self.lig_proj,
                        self.prot_pool_attn, self.lig_pool_attn):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
        elif self.variant == "v10":
            # Init projections and post-cross-attn heads
            for lin in (self.prot_proj, self.lig_proj):
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            for heads in (self.prot_heads, self.lig_heads):
                for h in heads:
                    nn.init.xavier_uniform_(h.weight)
                    nn.init.zeros_(h.bias)

        # Init classifier (Linear or Sequential)
        if isinstance(self.classifier, nn.Sequential):
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
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
        # --- Apply embedding adapters (if enabled) --------------------
        if self.use_adapter:
            protein_matrix = self.prot_adapter(protein_matrix)
            ligand_matrix = self.lig_adapter(ligand_matrix)

        # --- Mean-pooled embeddings (contrastive & cosine feat) -------
        if self.contrastive_dim > 0 or self.cosine_feat:
            _pm = protein_mask.unsqueeze(-1).float()
            _lm = ligand_mask.unsqueeze(-1).float()
            _prot_mean = (protein_matrix * _pm).sum(1) / _pm.sum(1).clamp(min=1)
            _lig_mean = (ligand_matrix * _lm).sum(1) / _lm.sum(1).clamp(min=1)

            if self.contrastive_dim > 0:
                self._z_prot = F.normalize(self.prot_contrast_proj(_prot_mean), dim=-1)
                self._z_lig = F.normalize(self.lig_contrast_proj(_lig_mean), dim=-1)

            if self.cosine_feat:
                # Project to common space before cosine similarity
                if self.contrastive_dim > 0:
                    p_proj = self._z_prot
                    l_proj = self._z_lig
                else:
                    p_proj = self.prot_cos_proj(_prot_mean)
                    l_proj = self.lig_cos_proj(_lig_mean)
                self._cos_sim_feat = F.cosine_similarity(
                    p_proj, l_proj, dim=-1,
                ).unsqueeze(1)  # [B, 1]

        # --- Build multi-head interaction maps -----------------------
        if self.variant == "v9":
            # Pure Cross-Attention + Attention Pooling + MLP
            prot = self.prot_proj(protein_matrix)  # [B, sp, d_model]
            lig  = self.lig_proj(ligand_matrix)    # [B, sl, d_model]

            prot_key_mask = (protein_mask == 0)
            lig_key_mask  = (ligand_mask == 0)

            for i in range(self.num_cross_layers):
                prot_att, _ = self.cross_prot_to_lig[i](
                    query=prot, key=lig, value=lig,
                    key_padding_mask=lig_key_mask,
                )
                prot = self.prot_norms[i](prot + prot_att)

                lig_att, _ = self.cross_lig_to_prot[i](
                    query=lig, key=prot, value=prot,
                    key_padding_mask=prot_key_mask,
                )
                lig = self.lig_norms[i](lig + lig_att)

            # Attention pooling
            p_scores = self.prot_pool_attn(prot).squeeze(-1)
            p_scores = p_scores.masked_fill(prot_key_mask, -1e9)
            p_weights = torch.softmax(p_scores, dim=-1).unsqueeze(-1)
            prot_pooled = (prot * p_weights).sum(dim=1)  # [B, d_model]

            l_scores = self.lig_pool_attn(lig).squeeze(-1)
            l_scores = l_scores.masked_fill(lig_key_mask, -1e9)
            l_weights = torch.softmax(l_scores, dim=-1).unsqueeze(-1)
            lig_pooled = (lig * l_weights).sum(dim=1)  # [B, d_model]

            combined = torch.cat([prot_pooled, lig_pooled], dim=-1)
            combined = self.dropout(combined)
            logits = self.classifier(combined)  # [B, 1]
            return logits

        if self.variant == "v10":
            # Cross-Attention + CNN hybrid
            prot = self.prot_proj(protein_matrix)
            lig  = self.lig_proj(ligand_matrix)

            prot_key_mask = (protein_mask == 0)
            lig_key_mask  = (ligand_mask == 0)

            for i in range(self.num_cross_layers):
                prot_att, _ = self.cross_prot_to_lig[i](
                    query=prot, key=lig, value=lig,
                    key_padding_mask=lig_key_mask,
                )
                prot = self.prot_norms[i](prot + prot_att)

                lig_att, _ = self.cross_lig_to_prot[i](
                    query=lig, key=prot, value=prot,
                    key_padding_mask=prot_key_mask,
                )
                lig = self.lig_norms[i](lig + lig_att)

            # Multi-head dot-product on ENRICHED representations
            maps: list[torch.Tensor] = []
            for ph, lh in zip(self.prot_heads, self.lig_heads):
                p = ph(prot)
                l = lh(lig)
                if self.cosine_sim:
                    p = F.normalize(p, dim=-1)
                    l = F.normalize(l, dim=-1)
                m = torch.bmm(p, l.transpose(1, 2)) * self.scale
                maps.append(m)
            interaction = torch.stack(maps, dim=1)  # [B, K, sp, sl]

        elif self.variant in ("v7", "v7_gated"):
            # If gated, squelch noise with local MLP gate BEFORE projection
            p_feat = protein_matrix * self.prot_gate(protein_matrix) if self.variant == "v7_gated" else protein_matrix
            l_feat = ligand_matrix * self.lig_gate(ligand_matrix) if self.variant == "v7_gated" else ligand_matrix

            maps: list[torch.Tensor] = []
            for k, (ph, lh) in enumerate(zip(self.prot_heads, self.lig_heads)):
                p = ph(p_feat)
                l = lh(l_feat)
                if self.use_rope:
                    # 1D RoPE per modality before the cross dot product:
                    # M_k[i,j] = (rope(P)_i) · (rope(L)_j) carries relative
                    # position info along each axis. Cosine_sim, if enabled,
                    # is applied AFTER rotation (rotation preserves norm).
                    p = apply_rope_1d(p)
                    l = apply_rope_1d(l)
                if self.cosine_sim:
                    p = F.normalize(p, dim=-1)
                    l = F.normalize(l, dim=-1)
                m = torch.bmm(p, l.transpose(1, 2)) * self.scale
                if self.use_ban_residual:
                    # BAN residual: M_k += α_k · P_feat W_k L_featᵀ
                    # Compute via einsum to avoid intermediate (B, sp, lig_dim).
                    # Final shape matches dot-product map: [B, sp, sl].
                    m_ban = torch.einsum(
                        'bip,pq,bjq->bij',
                        p_feat, self.ban_weights[k], l_feat,
                    )
                    m = m + self.ban_alphas[k] * m_ban
                maps.append(m)
            interaction = torch.stack(maps, dim=1)

        elif self.variant == "v8":
            p_proj = torch.einsum(
                'bip, kpd -> bkid', protein_matrix, self.W_ban
            )
            interaction = torch.matmul(
                p_proj,
                ligand_matrix.unsqueeze(1).transpose(-1, -2),
            ) * self.scale

        # Mask padding positions
        mask_2d = protein_mask.unsqueeze(2) * ligand_mask.unsqueeze(1)
        interaction = interaction * mask_2d.unsqueeze(1)

        # --- 2D CNN ---------------------------------------------------
        features = self.cnn(interaction)
        features = features * mask_2d.unsqueeze(1)

        # --- Hierarchical attention pool → classification ------------
        pooled = self.pool(features, protein_mask, ligand_mask)
        pooled = self.dropout(pooled)

        # Concatenate global cosine similarity feature
        if self.cosine_feat and self._cos_sim_feat is not None:
            pooled = torch.cat([pooled, self._cos_sim_feat], dim=-1)

        # Concatenate Morgan FP topological feature (DrugBAN/GraphBAN
        # GCN proxy). Self._morgan_feat is set externally via the train
        # loop before forward() when morgan_fp is in the batch dict.
        if self.morgan_n_bits > 0 and self._morgan_feat is not None:
            morgan_proj = self.morgan_proj(self._morgan_feat)  # [B, morgan_proj_dim]
            pooled = torch.cat([pooled, morgan_proj], dim=-1)

        # Stash pooled vector so external loops (e.g., adversarial DA)
        # can hook into the pre-classifier representation without a
        # second forward pass.
        self._last_pooled = pooled

        logits = self.classifier(pooled)  # [B, 1]

        return logits


# ======================================================================
# Threshold sweep
# ======================================================================

def _best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "mcc",
) -> tuple[float, float]:
    """Two-pass threshold sweep that maximises `metric` ∈ {"mcc","f1"}.

    Pass 1: coarse grid (100 points) over [0.01, 0.99]
    Pass 2: fine grid (100 points) in ±0.05 around the best

    F1 mode mirrors DrugBAN/GraphBAN native criterion (val-F1-optimal).
    """
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    metric = str(metric).lower()
    if metric == "f1":
        score_fn = lambda yt, yp: float(f1_score(yt, yp, zero_division=0))
    else:
        score_fn = lambda yt, yp: float(matthews_corrcoef(yt, yp))

    grid = np.linspace(0.01, 0.99, 100)
    anchors = np.unique(np.clip(y_proba, 0.01, 0.99))
    thresholds = np.unique(np.concatenate([grid, anchors]))

    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        s = score_fn(y_true, pred)
        if s > best_score:
            best_score = s
            best_thr = float(thr)

    lo = max(0.01, best_thr - 0.05)
    hi = min(0.99, best_thr + 0.05)
    fine_grid = np.linspace(lo, hi, 100)
    for thr in fine_grid:
        pred = (y_proba >= thr).astype(int)
        s = score_fn(y_true, pred)
        if s > best_score:
            best_score = s
            best_thr = float(thr)

    return best_thr, best_score


def _threshold_metric_env() -> str:
    """Read BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC; default 'mcc'.

    Controls which metric the val-set threshold sweep maximises. The
    chosen threshold is then applied to the test set for reporting.
    """
    return str(os.getenv("BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC", "mcc")).lower()


def _selection_metric_env() -> str:
    """Read BENCHMARK_LEVEL4CNN_SELECTION_METRIC; defaults to threshold metric.

    Controls which metric the checkpoint-selection composite score
    maximises during training. Decouples model selection from the
    reporting threshold (lesson 15 in licoes_aprendidas.md §6.7).
    Typical use: THRESHOLD=f1 (matches DrugBAN/GraphBAN reporting) +
    SELECTION=mcc (preserves discriminative epoch picking).
    """
    val = os.getenv("BENCHMARK_LEVEL4CNN_SELECTION_METRIC")
    if val is None:
        return _threshold_metric_env()
    return str(val).lower()


def _best_mcc_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """Back-compat wrapper. Honours BENCHMARK_LEVEL4CNN_THRESHOLD_METRIC.

    Returns (threshold, score_in_chosen_metric). When metric=="f1", the
    second element is val-F1 at that threshold; when metric=="mcc",
    it is val-MCC. Callers that explicitly need MCC at the chosen
    threshold should recompute it from the returned threshold.
    """
    return _best_threshold(y_true, y_proba, metric=_threshold_metric_env())


# ======================================================================
# Platt Scaling — probability calibration
# ======================================================================

@torch.inference_mode()
def _platt_calibrate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> LogisticRegression:
    """Fit Platt scaling on raw logits from the validation set.

    Returns a fitted LogisticRegression that maps raw logits → calibrated
    probabilities.  Usage: calibrator.predict_proba(logits)[:, 1]
    """
    model.eval()
    _raw = getattr(model, '_orig_mod', model)
    model_dtype = next(_raw.parameters()).dtype
    eval_amp = device.type == "cuda" and model_dtype != torch.float64

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in val_loader:
        p = batch["protein_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        l = batch["ligand_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        pm = batch["protein_mask"].to(device, non_blocking=True)
        lm = batch["ligand_mask"].to(device, non_blocking=True)
        y = batch["label"].numpy()

        with torch.amp.autocast(device_type=device.type, enabled=eval_amp):
            _set_aux_features(model, batch, device, model_dtype)
            logits = model(p, l, pm, lm)

        all_logits.append(logits.float().cpu().numpy().ravel())
        all_targets.append(y.ravel())

    logits_np = np.concatenate(all_logits).reshape(-1, 1)
    targets_np = np.concatenate(all_targets).astype(int)

    # Platt scaling = logistic regression on raw logits (no regularisation)
    calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits_np, targets_np)

    # Report calibration shift
    a = float(calibrator.coef_[0, 0])
    b = float(calibrator.intercept_[0])
    tqdm.write(f"    Platt calibration: a={a:.4f}, b={b:.4f}")

    return calibrator


@torch.inference_mode()
def _temperature_calibrate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Fit temperature scaling T on validation logits (Guo et al. 2017).

    Temperature scaling divides logits by a single scalar T before sigmoid.
    Minimises negative log-likelihood on val set — 1 parameter, so it
    generalises better than Platt (2 parameters) across scaffold splits.

    T > 1  →  softer probabilities (model was overconfident)
    T < 1  →  sharper probabilities (model was underconfident)
    T = 1  →  no change (model is already well-calibrated)
    """
    from scipy.optimize import minimize_scalar

    model.eval()
    _raw = getattr(model, '_orig_mod', model)
    model_dtype = next(_raw.parameters()).dtype
    eval_amp = device.type == "cuda" and model_dtype != torch.float64

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in val_loader:
        p = batch["protein_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        l = batch["ligand_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        pm = batch["protein_mask"].to(device, non_blocking=True)
        lm = batch["ligand_mask"].to(device, non_blocking=True)
        y = batch["label"].numpy()

        with torch.amp.autocast(device_type=device.type, enabled=eval_amp):
            _set_aux_features(model, batch, device, model_dtype)
            logits = model(p, l, pm, lm)

        all_logits.append(logits.float().cpu().numpy().ravel())
        all_targets.append(y.ravel())

    logits_np = np.concatenate(all_logits)
    targets_np = np.concatenate(all_targets).astype(float)

    def _nll(T: float) -> float:
        """Binary cross-entropy after temperature scaling."""
        scaled = logits_np / max(float(T), 1e-4)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(
            targets_np * np.log(probs) + (1.0 - targets_np) * np.log(1.0 - probs)
        ))

    result = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
    T = float(result.x)
    tqdm.write(
        f"    Temperature scaling: T={T:.4f} "
        f"({'underconfident → softened' if T > 1 else 'overconfident → sharpened' if T < 1 else 'well-calibrated'})"
    )
    return T


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
    # Unwrap compiled model to get clean state_dict keys (no _orig_mod. prefix)
    _raw_model = getattr(model, '_orig_mod', model)
    payload = {
        "epoch": epoch,
        "model_state_dict": _raw_model.state_dict(),
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

def _set_aux_features(model: nn.Module, batch: dict, device, dtype) -> None:
    """Stash auxiliary per-batch features onto the model BEFORE forward.

    InteractionMapCNN.forward() does NOT take aux features as args; it
    reads `self._morgan_feat` (and similar future hooks) just before the
    classifier head. This helper extracts those features from `batch`,
    moves to device, and assigns to the model. Should be called once
    per batch in train/val/eval loops if Morgan FP injection is enabled.
    """
    _orig = getattr(model, "_orig_mod", model)
    if "morgan_fp" in batch and getattr(_orig, "morgan_n_bits", 0) > 0:
        _orig._morgan_feat = batch["morgan_fp"].to(
            device=device, dtype=dtype, non_blocking=True)


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
    num_cross_layers: int = 2,
    mlp_head: bool = False,
    cosine_sim: bool = False,
    use_adapter: bool = False,
    adapter_bottleneck_prot: int = 256,
    adapter_bottleneck_lig: int = 512,
    adapter_layers: int = 1,
    adapter_self_attn: bool = False,
    adapter_lr_mult: float = 1.0,
    adapter_layers_prot: int | None = None,
    adapter_layers_lig: int | None = None,
    adapter_attn_heads_prot: int = 4,
    adapter_attn_heads_lig: int = 4,
    label_smooth: float = 0.0,
    mixup_alpha: float = 0.0,
    contrastive_weight: float = 0.0,
    cosine_feat: bool = False,
    contrastive_dim: int = 128,
    pool_num_heads: int = 1,
    train_to_zero: bool = False,
    train_to_zero_threshold: float = 0.01,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 50,
    swa_start: int = 0,
    use_ban_residual: bool = False,
    use_rope: bool = False,
    morgan_n_bits: int = 0,
    morgan_proj_dim: int = 32,
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

    # --- Precision flags ----------------------------------------------
    use_double = os.getenv("BENCHMARK_LEVEL4CNN_DOUBLE", "1") == "1"
    no_amp = os.getenv("BENCHMARK_LEVEL4CNN_NO_AMP", "0") == "1"
    deterministic = os.getenv("BENCHMARK_LEVEL4CNN_DETERMINISTIC", "0") == "1"

    dtype = torch.float64 if use_double else torch.float32
    use_amp = device.type == "cuda" and not no_amp and not use_double
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # TF32: OK with AMP (fp16 path dominates). With AMP off + fp32, TF32
    # truncates matmul mantissa to ~10 bits (fp16-like precision) which
    # kills accuracy in deep cross-attention stacks. Disable in pure-fp32.
    if device.type == "cuda":
        pure_fp32 = (not use_amp) and (not use_double)
        tf32_on = not pure_fp32
        torch.backends.cuda.matmul.allow_tf32 = tf32_on
        torch.backends.cudnn.allow_tf32 = tf32_on

    # cuDNN disable knob: diamante-02 driver 12.4 + cuDNN 9.x ABI mismatch
    # triggers CUDNN_STATUS_NOT_INITIALIZED on Conv2d regardless of dtype.
    # Also auto-disabled when double=true (cuDNN fp64 Conv2d unreliable).
    disable_cudnn = os.getenv("BENCHMARK_LEVEL4CNN_DISABLE_CUDNN", "0") == "1"
    if device.type == "cuda":
        torch.backends.cudnn.enabled = (not disable_cudnn) and (not use_double)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        tqdm.write("    Deterministic mode: ON (cudnn.benchmark=False)")
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = torch.backends.cudnn.enabled

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
        num_cross_layers=num_cross_layers,
        mlp_head=mlp_head,
        cosine_sim=cosine_sim,
        use_adapter=use_adapter,
        adapter_bottleneck_prot=adapter_bottleneck_prot,
        adapter_bottleneck_lig=adapter_bottleneck_lig,
        adapter_layers=adapter_layers,
        adapter_self_attn=adapter_self_attn,
        adapter_layers_prot=adapter_layers_prot,
        adapter_layers_lig=adapter_layers_lig,
        adapter_attn_heads_prot=adapter_attn_heads_prot,
        adapter_attn_heads_lig=adapter_attn_heads_lig,
        contrastive_dim=contrastive_dim if contrastive_weight > 0 else 0,
        cosine_feat=cosine_feat,
        pool_num_heads=pool_num_heads,
        use_ban_residual=use_ban_residual,
        use_rope=use_rope,
        morgan_n_bits=morgan_n_bits,
        morgan_proj_dim=morgan_proj_dim,
    ).to(device=device, dtype=dtype)

    # Optional adversarial domain head (CDAN-style). Only constructed when
    # BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA > 0. Attaches to the pooled
    # feature vector (HierPool output, dim=cnn_channels[+1 if cosine_feat]).
    adversarial_lambda = float(os.getenv(
        "BENCHMARK_LEVEL4CNN_ADVERSARIAL_LAMBDA", "0.0"))
    n_domains = int(os.getenv("BENCHMARK_LEVEL4CNN_ADVERSARIAL_N_DOMAINS", "2"))
    pool_out_dim = cnn_channels + (1 if cosine_feat else 0)
    adversarial_head = None
    if adversarial_lambda > 0:
        adversarial_head = DomainAdversarialHead(
            in_dim=pool_out_dim, n_domains=n_domains,
        ).to(device=device, dtype=dtype)
        tqdm.write(
            f"    Adversarial DA: λ_max={adversarial_lambda}, n_domains={n_domains}, "
            f"pool_dim={pool_out_dim} (linear ramp first half of training)"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _cross_layers = int(os.getenv('BENCHMARK_LEVEL4CNN_CROSS_LAYERS', '2'))
    variant_info = (
        f"head_dim={head_dim}" if variant == "v7"
        else f"head_dim={head_dim}, MLP-gated" if variant == "v7_gated"
        else f"BAN W[{num_heads}x{protein_dim}x{MOLFORMER_DIM}]" if variant == "v8"
        else f"CrossAttn d_model={head_dim}, layers={_cross_layers}" if variant == "v9"
        else f"CrossAttn+CNN d_model={head_dim}, layers={_cross_layers}"
    )
    head_tag = ", mlp_head" if mlp_head else ""
    adapter_tag = f", adapter({adapter_bottleneck_prot}/{adapter_bottleneck_lig})" if use_adapter else ""
    tqdm.write(
        f"    InteractionModel: variant={variant}, heads={num_heads}, "
        f"{variant_info}{head_tag}{adapter_tag}, dropout={dropout:.2f}\n"
        f"    Trainable params: {trainable:,} / {total:,}"
    )

    # --- torch.compile for fused training kernels (PyTorch 2.x) -------
    # Disable via BENCHMARK_LEVEL4CNN_NO_COMPILE=1 (default on diamante-02:
    # dynamic cross-attention shapes trigger noisy symbolic_shapes warnings
    # and repeated recompiles; gain is marginal without cuDNN).
    compiled = False
    no_compile = os.getenv("BENCHMARK_LEVEL4CNN_NO_COMPILE", "0") == "1"
    if (hasattr(torch, 'compile') and device.type == 'cuda'
            and not use_double and not no_compile):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            compiled = True
            tqdm.write("    torch.compile: enabled (reduce-overhead)")
        except Exception as e:
            tqdm.write(f"    torch.compile: unavailable ({e})")
    elif no_compile:
        tqdm.write("    torch.compile: disabled via BENCHMARK_LEVEL4CNN_NO_COMPILE=1")

    # --- Optimiser, loss, scheduler -----------------------------------
    weight_decay = float(os.getenv("BENCHMARK_LEVEL4CNN_WEIGHT_DECAY", "0.02"))
    # Per-side adapter LR multipliers (Direção D — ligand-heavy optimisation
    # bias). Defaults inherit adapter_lr_mult; override individually via
    # BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_{PROT,LIG}. Hypothesis (kinase
    # ATP-pocket conserved → ligand carries most discriminative signal):
    # giving lig_adapter a higher LR than prot_adapter accelerates the
    # side that needs more capacity.
    lr_mult_prot = float(os.getenv(
        "BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_PROT", str(adapter_lr_mult)))
    lr_mult_lig = float(os.getenv(
        "BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT_LIG", str(adapter_lr_mult)))

    # Optional dedicated LR multiplier for BAN-residual params (W_k + α_k):
    # complements Lição 19 (capacity↔LR atomicidade) for BAN-residual case.
    ban_lr_mult = float(os.getenv("BENCHMARK_LEVEL4CNN_BAN_LR_MULT", "1.0"))

    use_per_side = use_adapter and (lr_mult_prot != 1.0 or lr_mult_lig != 1.0)
    use_ban_lr   = use_ban_residual and ban_lr_mult != 1.0

    if use_per_side or use_ban_lr:
        param_groups = []
        claimed_ids = set()

        # Per-side adapter param groups (Direção D).
        if use_per_side:
            if lr_mult_prot == lr_mult_lig:
                adapter_params = (
                    list(model.prot_adapter.parameters())
                    + list(model.lig_adapter.parameters())
                )
                claimed_ids.update(id(p) for p in adapter_params)
                param_groups.append({"params": adapter_params, "lr": lr * lr_mult_prot})
            else:
                prot_params = list(model.prot_adapter.parameters())
                lig_params = list(model.lig_adapter.parameters())
                claimed_ids.update(id(p) for p in prot_params + lig_params)
                param_groups.append({"params": prot_params, "lr": lr * lr_mult_prot})
                param_groups.append({"params": lig_params,  "lr": lr * lr_mult_lig})

        # BAN-residual params (W_k bilinear matrices + α_k scalar gates).
        if use_ban_lr:
            ban_params = list(model.ban_alphas.parameters()) + list(model.ban_weights.parameters())
            ban_params = [p for p in ban_params if id(p) not in claimed_ids]
            claimed_ids.update(id(p) for p in ban_params)
            param_groups.append({"params": ban_params, "lr": lr * ban_lr_mult})

        # Everything else (including adversarial_head if present).
        other_params = [
            p for p in model.parameters()
            if id(p) not in claimed_ids and p.requires_grad
        ]
        if adversarial_head is not None:
            other_params += list(adversarial_head.parameters())
        param_groups.append({"params": other_params, "lr": lr})

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay,
            fused=(device.type == 'cuda' and not use_double),
        )
        msg = ["    Differential LR:"]
        if use_per_side:
            if lr_mult_prot == lr_mult_lig:
                msg.append(f"adapter={lr * lr_mult_prot:.2e} ({lr_mult_prot}x)")
            else:
                msg.append(f"prot={lr*lr_mult_prot:.2e}({lr_mult_prot}x), lig={lr*lr_mult_lig:.2e}({lr_mult_lig}x)")
        if use_ban_lr:
            msg.append(f"ban={lr*ban_lr_mult:.2e}({ban_lr_mult}x)")
        msg.append(f"other={lr:.2e}")
        tqdm.write(", ".join(msg))
    else:
        all_params = list(model.parameters())
        if adversarial_head is not None:
            all_params += list(adversarial_head.parameters())
        optimizer = torch.optim.AdamW(
            all_params, lr=lr, weight_decay=weight_decay,
            fused=(device.type == 'cuda' and not use_double),
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_labels = _extract_labels(train_loader)
    pos_weight = _compute_pos_weight(train_labels)

    # --- Loss function: Focal Loss or BCE -----------------------------
    focal_gamma = float(os.getenv("BENCHMARK_LEVEL4CNN_FOCAL_GAMMA", "2.0"))
    use_focal = os.getenv("BENCHMARK_LEVEL4CNN_FOCAL", "1") == "1"

    if use_focal:
        criterion = FocalLoss(
            gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        tqdm.write(
            f"    Loss: FocalLoss(gamma={focal_gamma}, pos_weight={pos_weight:.2f}), "
            f"lr={lr:.1e}, wd={weight_decay}"
        )
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=dtype, device=device),
        )
        tqdm.write(f"    Loss: BCE(pos_weight={pos_weight:.2f}), lr={lr:.1e}, wd={weight_decay}")

    if cosine_sim:
        tqdm.write("    Interaction maps: cosine similarity (L2-normalized)")
    if contrastive_weight > 0:
        tqdm.write(f"    Contrastive loss: weight={contrastive_weight}, dim={contrastive_dim} (ConPLex-inspired)")
    if cosine_feat:
        tqdm.write("    Cosine similarity: added as extra classifier feature")
    if label_smooth > 0:
        tqdm.write(f"    Label smoothing: eps={label_smooth}")
    if mixup_alpha > 0:
        tqdm.write(f"    Mixup: alpha={mixup_alpha}")
    if train_to_zero:
        tqdm.write(
            f"    *** TRAIN-TO-ZERO mode: early stopping DISABLED, "
            f"training until train_loss & val_loss < {train_to_zero_threshold:.4f} "
            f"(max {epochs} epochs) ***"
        )

    # --- Checkpoint paths ---------------------------------------------
    ckpt_path = os.path.join(checkpoint_dir, "training_checkpoint.pt") if checkpoint_dir else None

    # --- Stochastic Weight Averaging (SWA) setup ----------------------
    # When swa_start > 0, an AveragedModel is maintained in parallel and
    # updated each epoch starting at swa_start. After training, BN
    # running stats are recomputed via update_bn over train_loader, and
    # the SWA model substitutes the best-checkpoint model for Platt /
    # threshold / test evaluation. Izmailov et al., UAI 2018.
    swa_enabled = swa_start > 0
    swa_model = None
    if swa_enabled:
        from torch.optim.swa_utils import AveragedModel
        _raw_for_swa = getattr(model, '_orig_mod', model)
        swa_model = AveragedModel(_raw_for_swa).to(device)
        tqdm.write(f"    SWA enabled: averaging weights from epoch {swa_start} onward")

    # --- Training loop ------------------------------------------------
    best_score = -float("inf")
    best_state = None
    no_improve = 0
    start_epoch = 1

    # --- Resume from checkpoint if available --------------------------
    if ckpt_path and os.path.exists(ckpt_path):
        tqdm.write(f"    Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix if checkpoint was saved by compiled model
        raw_sd = ckpt["model_state_dict"]
        _clean_sd = {
            k.removeprefix("_orig_mod."): v for k, v in raw_sd.items()
        }
        _raw_for_load = getattr(model, '_orig_mod', model)
        _raw_for_load.load_state_dict(_clean_sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_score = ckpt.get("best_score", -float("inf"))
        no_improve = ckpt.get("no_improve", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if ckpt.get("best_state") is not None:
            _bs = ckpt["best_state"]
            best_state = {k.removeprefix("_orig_mod."): v for k, v in _bs.items()}
        tqdm.write(
            f"    Resumed at epoch {start_epoch}, "
            f"best_val_mcc={best_score:.4f}, no_improve={no_improve}"
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            p = batch["protein_matrix"].to(device=device, dtype=dtype, non_blocking=True)
            l = batch["ligand_matrix"].to(device=device, dtype=dtype, non_blocking=True)
            pm = batch["protein_mask"].to(device, non_blocking=True)
            lm = batch["ligand_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(1)

            # --- Mixup (only during training) ---------------------------
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(p.size(0), device=device)
                p = lam * p + (1 - lam) * p[idx]
                l = lam * l + (1 - lam) * l[idx]
                pm = pm | pm[idx]   # union of valid positions from both samples
                lm = lm | lm[idx]
                y = lam * y + (1 - lam) * y[idx]

            # --- Label smoothing ----------------------------------------
            if label_smooth > 0:
                y = y * (1 - label_smooth) + label_smooth * 0.5

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _set_aux_features(model, batch, device, dtype)
                logits = model(p, l, pm, lm)
                loss = criterion(logits, y)

                # Contrastive regularization (ConPLex-inspired)
                _orig = getattr(model, '_orig_mod', model)
                if contrastive_weight > 0 and _orig.contrastive_dim > 0:
                    c_loss = ContrastiveLoss(margin=0.5)(
                        _orig._z_prot, _orig._z_lig, y,
                    )
                    loss = loss + contrastive_weight * c_loss

                # Adversarial domain alignment (CDAN-style, Ganin 2015):
                # only active when adversarial_lambda > 0 AND batch has
                # both H+NH samples (joint training on `all` corpus).
                if (adversarial_head is not None and adversarial_lambda > 0
                        and "domain" in batch):
                    domain = batch["domain"].to(device)
                    valid_dom = domain >= 0
                    if valid_dom.any():
                        # λ schedule: linearly ramp from 0 to lambda_max
                        # over the first half of training to avoid early
                        # gradient noise dominating the classification.
                        progress = min(1.0, epoch / max(epochs * 0.5, 1.0))
                        lambd = adversarial_lambda * progress
                        pooled = _orig._last_pooled
                        domain_logits = adversarial_head(pooled[valid_dom], lambd)
                        adv_loss = F.cross_entropy(
                            domain_logits, domain[valid_dom])
                        loss = loss + lambd * adv_loss

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
                p = batch["protein_matrix"].to(device=device, dtype=dtype, non_blocking=True)
                l = batch["ligand_matrix"].to(device=device, dtype=dtype, non_blocking=True)
                pm = batch["protein_mask"].to(device, non_blocking=True)
                lm = batch["ligand_mask"].to(device, non_blocking=True)
                y = batch["label"].to(device=device, dtype=dtype, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _set_aux_features(model, batch, device, dtype)
                    logits = model(p, l, pm, lm)
                    val_loss += criterion(logits, y).item()

                val_n += 1
                val_probs.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
                val_targets.append(y.cpu().numpy().ravel())

        avg_val = val_loss / max(val_n, 1)
        probs = np.concatenate(val_probs) if val_probs else np.array([])
        targets = np.concatenate(val_targets) if val_targets else np.array([])

        # Threshold for reporting (test eval will reuse): chosen by
        # THRESHOLD_METRIC env (mcc|f1). Selection score is computed
        # independently by SELECTION_METRIC env (defaults to threshold
        # metric). See licoes_aprendidas §6.7 — lesson 15.
        threshold_metric = _threshold_metric_env()
        selection_metric = _selection_metric_env()

        thr, _thr_score = _best_threshold(
            targets.astype(int), probs, metric=threshold_metric)
        if selection_metric == threshold_metric:
            val_selection_score = _thr_score
        else:
            _, val_selection_score = _best_threshold(
                targets.astype(int), probs, metric=selection_metric)

        # Real val MCC and val F1 at the reporting threshold (display).
        _val_preds = (probs >= thr).astype(int)
        val_mcc = float(matthews_corrcoef(targets.astype(int), _val_preds))
        val_f1 = float(f1_score(targets.astype(int), _val_preds, zero_division=0))

        # --- Composite selection criterion -----------------------------
        # score = val_<selection_metric> - λ · val_loss
        # λ=0 recovers pure val_<selection_metric> behaviour.
        _lambda_loss = float(os.getenv("BENCHMARK_LEVEL4CNN_SELECTION_LAMBDA_LOSS", "0.0"))
        composite_score = float(val_selection_score) - _lambda_loss * float(avg_val)

        improved = composite_score > best_score
        marker = " ★" if improved else ""

        if train_to_zero:
            tqdm.write(
                f"    Epoch {epoch:3d}: loss={avg_train:.6f}, "
                f"val_loss={avg_val:.6f}, val_mcc={val_mcc:.4f}, "
                f"val_f1={val_f1:.4f}, thr={thr:.3f}{marker}"
            )
        else:
            tqdm.write(
                f"    Epoch {epoch:3d}: loss={avg_train:.4f}, "
                f"val_loss={avg_val:.4f}, val_mcc={val_mcc:.4f}, "
                f"val_f1={val_f1:.4f}, thr={thr:.3f} "
                f"[sel:{selection_metric}={val_selection_score:.4f}] "
                f"({no_improve}/{patience}){marker}"
            )
        sys.stdout.flush()
        sys.stderr.flush()

        if improved:
            best_score = composite_score
            # Save raw (unwrapped) state_dict for portability
            _raw_best = getattr(model, '_orig_mod', model)
            best_state = {k: v.cpu().clone() for k, v in _raw_best.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # --- SWA update (after epoch >= swa_start) -------------------
        if swa_enabled and epoch >= swa_start:
            _raw_for_swa = getattr(model, '_orig_mod', model)
            swa_model.update_parameters(_raw_for_swa)

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

    # --- Restore best model OR finalize SWA ---------------------------
    if swa_enabled and swa_model is not None:
        # SWA path: recompute BN running stats with averaged weights
        # (critical — without this the BN-CNN diverges from the new
        # averaged Conv2d weights and predictions become unreliable).
        # torch.optim.swa_utils.update_bn assumes loader yields tensors
        # or tuples; our train_loader yields dict batches. Custom impl
        # below mirrors the official function's logic.
        tqdm.write("    SWA: recomputing BN running statistics over train_loader …")
        _bn_momenta = {}
        for _m in swa_model.modules():
            if isinstance(_m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                _m.reset_running_stats()
                _bn_momenta[_m] = _m.momentum
                _m.momentum = None  # accumulate full statistics
        if _bn_momenta:
            swa_model.train()
            _swa_dtype = next(swa_model.parameters()).dtype
            with torch.no_grad():
                for batch in train_loader:
                    p = batch["protein_matrix"].to(device=device, dtype=_swa_dtype, non_blocking=True)
                    l = batch["ligand_matrix"].to(device=device, dtype=_swa_dtype, non_blocking=True)
                    pm = batch["protein_mask"].to(device, non_blocking=True)
                    lm = batch["ligand_mask"].to(device, non_blocking=True)
                    swa_model(p, l, pm, lm)
            # Restore original momenta
            for _m, _mom in _bn_momenta.items():
                _m.momentum = _mom
        # Replace the model used for downstream Platt / threshold / test
        # with the SWA-averaged weights. Wrapper unwrap matches existing
        # convention.
        _raw_restore = getattr(model, '_orig_mod', model)
        _raw_restore.load_state_dict(swa_model.module.state_dict())
        tqdm.write("    SWA: averaged weights loaded into model")
    elif best_state is not None:
        _raw_restore = getattr(model, '_orig_mod', model)
        _raw_restore.load_state_dict(best_state)
    # Call .to().eval() on the wrapper (delegates to _orig_mod internally)
    model.to(device)
    model.eval()

    # NOTE: torch.compile was already applied before training (if available).
    # Re-compiling after load_state_dict on an already-compiled model
    # produces a plain function wrapper that loses nn.Module API.

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
    calibrator: LogisticRegression | None = None,
    temperature: float = 1.0,
) -> dict:
    """Evaluate model and return full metrics dict.

    If `threshold` is provided (e.g., from val-optimized), use it directly.
    Otherwise, sweep to find the MCC-optimal threshold on this data.
    If `calibrator` is provided, Platt-calibrated probabilities are used;
    otherwise sigmoid(logit / temperature) is used (temperature scaling).

    Always computes MCC at fixed threshold=0.5 as a scaffold-agnostic
    sanity check (reported as mcc_at_05 in the returned dict).

    Returns dict with: mcc, mcc_at_05, threshold, accuracy, f1, precision,
    recall, auroc, plus raw y_true and y_prob arrays.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    # Auto-detect model dtype for double precision support
    _raw = getattr(model, '_orig_mod', model)
    model_dtype = next(_raw.parameters()).dtype
    eval_amp = device.type == "cuda" and model_dtype != torch.float64

    for batch in loader:
        p = batch["protein_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        l = batch["ligand_matrix"].to(device=device, dtype=model_dtype, non_blocking=True)
        pm = batch["protein_mask"].to(device, non_blocking=True)
        lm = batch["ligand_mask"].to(device, non_blocking=True)
        y = batch["label"].numpy()

        with torch.amp.autocast(device_type=device.type, enabled=eval_amp):
            logits = model(p, l, pm, lm)

        all_logits.append(logits.float().cpu().numpy().ravel())
        all_targets.append(y)

    raw_logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets).astype(int)

    # Calibrated or temperature-scaled probabilities
    if calibrator is not None:
        # Platt scaling takes priority when explicitly provided
        probs = calibrator.predict_proba(raw_logits.reshape(-1, 1))[:, 1]
    else:
        # Temperature scaling: sigmoid(logit / T); T=1.0 → plain sigmoid
        scaled_logits = raw_logits / max(float(temperature), 1e-4)
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))

    if threshold is None:
        # Pick threshold by active metric (mcc|f1) but always report
        # MCC computed via matthews_corrcoef at that threshold.
        thr, _ = _best_mcc_threshold(targets, probs)
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

    # MCC at fixed threshold 0.5 — scaffold-agnostic sanity check
    # (not optimised on any split, so valid even across distribution shifts)
    _has_both_classes = len(np.unique(targets)) > 1
    if _has_both_classes:
        preds_at_05 = (probs >= 0.5).astype(int)
        mcc_at_05 = float(matthews_corrcoef(targets, preds_at_05))
    else:
        mcc_at_05 = 0.0

    if desc:
        tqdm.write(
            f"    {desc}: MCC={mcc:.4f} (thr={thr:.3f}), "
            f"MCC@0.5={mcc_at_05:.4f}, AUROC={auroc:.4f}, "
            f"F1={f1:.4f}, acc={acc:.4f}"
        )

    return {
        "mcc": mcc,
        "mcc_at_05": mcc_at_05,
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
        num_cross_layers = int(os.getenv("BENCHMARK_LEVEL4CNN_CROSS_LAYERS", "2"))
        mlp_head = os.getenv("BENCHMARK_LEVEL4CNN_MLP_HEAD", "0") == "1"
        cosine_sim = os.getenv("BENCHMARK_LEVEL4CNN_COSINE_SIM", "0") == "1"
        use_adapter = os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER", "0") == "1"
        adapter_bottleneck_prot = int(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_PROT_DIM", "256"))
        adapter_bottleneck_lig = int(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LIG_DIM", "512"))
        adapter_layers = int(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS", "1"))
        adapter_self_attn = os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_SELF_ATTN", "0") == "1"
        adapter_lr_mult = float(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LR_MULT", "1.0"))
        # Asymmetric adapter capacity per side. Empty/missing → fall back
        # to symmetric (adapter_layers, attn_heads=4). Motivation: ligand
        # carries more discriminative info in kinase domain (conserved
        # ATP-binding pocket).
        _alp = os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS_PROT", "")
        _all = os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_LAYERS_LIG", "")
        adapter_layers_prot = int(_alp) if _alp else None
        adapter_layers_lig = int(_all) if _all else None
        adapter_attn_heads_prot = int(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_PROT", "4"))
        adapter_attn_heads_lig = int(os.getenv("BENCHMARK_LEVEL4CNN_ADAPTER_ATTN_HEADS_LIG", "4"))
        pool_num_heads = int(os.getenv("BENCHMARK_LEVEL4CNN_POOL_HEADS", "1"))
        swa_start = int(os.getenv("BENCHMARK_LEVEL4CNN_SWA_START", "0"))

        # --- Build dataloaders ----------------------------------------
        train_loader, val_loader, test_loader = build_matrix_dataloaders(
            dataset_type=self.dataset,
            embedding_name=self.embedding_name,
            scaffold_split_dir=self.scaffold_split_dir,
            batch_size=batch_size,
            dataset_source_filter=self._config.dataset_source_filter,
            mode=self.mode,
        )

        # CNN training is end-to-end (no separate feature-extraction stage),
        # so the full train loader is used for gradient steps.
        model_train_loader = train_loader

        # --- Train-to-zero mode ----------------------------------------
        train_to_zero = os.getenv("BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO", "0") == "1"
        train_to_zero_thr = float(os.getenv("BENCHMARK_LEVEL4CNN_TRAIN_TO_ZERO_THR", "0.01"))

        # --- Checkpoint frequency --------------------------------------
        checkpoint_every = int(os.getenv("BENCHMARK_LEVEL4CNN_CHECKPOINT_EVERY", "50"))

        # --- Regularization techniques --------------------------------
        label_smooth = float(os.getenv("BENCHMARK_LEVEL4CNN_LABEL_SMOOTH", "0.0"))
        mixup_alpha = float(os.getenv("BENCHMARK_LEVEL4CNN_MIXUP_ALPHA", "0.0"))
        contrastive_weight = float(os.getenv("BENCHMARK_LEVEL4CNN_CONTRASTIVE_WEIGHT", "0.0"))
        cosine_feat = os.getenv("BENCHMARK_LEVEL4CNN_COSINE_FEAT", "0") == "1"
        contrastive_dim = int(os.getenv("BENCHMARK_LEVEL4CNN_CONTRASTIVE_DIM", "128"))
        # Lição 12 reformulação: BAN-residual com α-gate aprendível.
        use_ban_residual = os.getenv("BENCHMARK_LEVEL4CNN_BAN_RESIDUAL", "0") == "1"
        # 2D RoPE (per-modality 1D RoPE applied before cross dot product).
        use_rope = os.getenv("BENCHMARK_LEVEL4CNN_USE_ROPE", "0") == "1"
        # Morgan FP topological feature (DrugBAN/GraphBAN GCN proxy).
        # Activate via BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR (cache path).
        # n_bits + proj_dim resolved here from env so model gets right dims.
        morgan_n_bits = (
            int(os.getenv("BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_BITS", "1024"))
            if os.getenv("BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_DIR") else 0
        )
        morgan_proj_dim = int(os.getenv("BENCHMARK_LEVEL4CNN_LIGAND_MORGAN_PROJ", "32"))

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
            num_cross_layers=num_cross_layers,
            mlp_head=mlp_head,
            cosine_sim=cosine_sim,
            use_adapter=use_adapter,
            adapter_bottleneck_prot=adapter_bottleneck_prot,
            adapter_bottleneck_lig=adapter_bottleneck_lig,
            adapter_layers=adapter_layers,
            adapter_self_attn=adapter_self_attn,
            adapter_lr_mult=adapter_lr_mult,
            adapter_layers_prot=adapter_layers_prot,
            adapter_layers_lig=adapter_layers_lig,
            adapter_attn_heads_prot=adapter_attn_heads_prot,
            adapter_attn_heads_lig=adapter_attn_heads_lig,
            label_smooth=label_smooth,
            mixup_alpha=mixup_alpha,
            contrastive_weight=contrastive_weight,
            cosine_feat=cosine_feat,
            contrastive_dim=contrastive_dim,
            pool_num_heads=pool_num_heads,
            swa_start=swa_start,
            train_to_zero=train_to_zero,
            train_to_zero_threshold=train_to_zero_thr,
            checkpoint_dir=output_dir,
            checkpoint_every=checkpoint_every,
            use_ban_residual=use_ban_residual,
            use_rope=use_rope,
            morgan_n_bits=morgan_n_bits,
            morgan_proj_dim=morgan_proj_dim,
        )

        # Unwrap torch.compile wrapper (if any) to access nn.Module API
        _raw = getattr(model, '_orig_mod', model)
        device = next(_raw.parameters()).device

        # ----------------------------------------------------------------
        # Calibration strategy (controlled by env vars):
        #
        #  Platt Scaling (default ON):
        #    Fits a 2-parameter logistic regression on val logits
        #    Corrects both scale and bias — empirically superior under
        #    scaffold splits (Human: +0.021 MCC vs Temperature, 4/5 seeds)
        #    Reference: Platt, "Probabilistic Outputs for SVMs", 1999.
        #
        #  Temperature Scaling (default OFF, opt-in via _TEMPERATURE=1):
        #    Fits a single scalar T on val logits → sigmoid(logit/T)
        #    1 parameter ⇒ lower risk of overfitting but corrects scale only
        #    Reference: Guo et al., ICML 2017.
        #
        #  Protocol: calibrate(val) → threshold(val) → evaluate(test)
        # ----------------------------------------------------------------
        use_platt = os.getenv("BENCHMARK_LEVEL4CNN_PLATT", "1") == "1"
        use_temperature = os.getenv("BENCHMARK_LEVEL4CNN_TEMPERATURE", "0") == "1"

        temperature = 1.0
        calibrator = None

        if use_platt:
            tqdm.write("  Fitting Platt calibration on val set...")
            calibrator = _platt_calibrate(model, val_loader, device)

        if use_temperature:
            tqdm.write("  Fitting temperature scaling on val set...")
            temperature = _temperature_calibrate(model, val_loader, device)

        # --- Evaluate -------------------------------------------------
        # Step 1: Evaluate on val to get val-optimized threshold
        val_result = _evaluate(
            model, val_loader, device, desc="Eval (val)",
            calibrator=calibrator, temperature=temperature,
        )
        val_threshold = val_result["threshold"]

        if self.mode == "train":
            eval_result = val_result
        else:
            # Step 2: Apply val threshold to test set (fair protocol)
            eval_result = _evaluate(
                model, test_loader, device,
                threshold=val_threshold, desc="Eval (test)",
                calibrator=calibrator, temperature=temperature,
            )

        # --- Save results in standard format --------------------------
        sc_key = "Split by Scaffold"
        cnn_metrics = {
            "mcc": round(eval_result["mcc"], 6),
            "mcc_at_05": round(eval_result["mcc_at_05"], 6),
            "threshold": round(eval_result["threshold"], 4),
            "accuracy": round(eval_result["accuracy"], 6),
            "f1": round(eval_result["f1"], 6),
            "precision": round(eval_result["precision"], 6),
            "recall": round(eval_result["recall"], 6),
            "auc": round(eval_result["auroc"], 6),
            "best_val_mcc": round(train_info["best_val_mcc"], 6),
            "val_threshold": round(val_threshold, 4),
            "temperature": round(temperature, 4),
        }
        result = {sc_key: {"MLP": cnn_metrics}}

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, "level4_cnn_model.pt")
        torch.save(_raw.state_dict(), ckpt_path)

        # Save raw predictions for reproducibility
        np.savez(
            os.path.join(output_dir, "raw_predictions.npz"),
            y_true=eval_result["y_true"],
            y_prob=eval_result["y_prob"],
        )

        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        tqdm.write(
            f"  Level 4 CNN (seed {seed}): "
            f"MCC={eval_result['mcc']:.4f} (thr={val_threshold:.3f}), "
            f"MCC@0.5={eval_result['mcc_at_05']:.4f}, "
            f"AUROC={eval_result['auroc']:.4f}, F1={eval_result['f1']:.4f}"
        )
        return result

    # ------------------------------------------------------------------
    # Ensemble: average probabilities across seeds → single prediction
    # ------------------------------------------------------------------

    def run(self, **kwargs: object) -> Optional[Dict]:
        """Override base run to add probability ensemble after all seeds."""
        base_result = super().run(**kwargs)

        # Attempt ensemble of raw predictions
        use_ensemble = os.getenv("BENCHMARK_LEVEL4CNN_ENSEMBLE", "1") == "1"
        if use_ensemble and self.mode != "train":
            self._ensemble_predictions()

        return base_result

    def _ensemble_predictions(self) -> None:
        """Load raw predictions from all seeds, average probs, compute MCC.

        The threshold is the mean of per-seed val-optimized thresholds
        (NOT swept on test) to maintain the fair evaluation protocol.
        """
        level_dir = self.output_dir_for_level()
        all_probs = []
        val_thresholds = []
        y_true = None

        for seed in self.seeds:
            seed_dir = os.path.join(level_dir, f"seed_{seed}")
            npz_path = os.path.join(seed_dir, "raw_predictions.npz")
            res_path = os.path.join(seed_dir, "level4_cnn_results.json")
            if not os.path.exists(npz_path) or not os.path.exists(res_path):
                tqdm.write(f"  Ensemble: skipping seed {seed} (missing files)")
                return
            data = np.load(npz_path)
            all_probs.append(data["y_prob"])
            if y_true is None:
                y_true = data["y_true"]

            # Load val threshold from per-seed results
            with open(res_path) as fh:
                seed_res = json.load(fh)
            sc_key = next(iter(seed_res), None)
            if sc_key:
                mlp = seed_res[sc_key].get("MLP", {})
                val_thresholds.append(mlp.get("val_threshold", 0.5))

        if y_true is None or len(all_probs) < 2:
            return

        # Average probabilities across seeds
        ensemble_probs = np.mean(all_probs, axis=0)

        # Use mean of val thresholds (fair protocol: no test optimization)
        thr = float(np.mean(val_thresholds)) if val_thresholds else 0.5

        preds = (ensemble_probs >= thr).astype(int)
        acc = float(np.mean(preds == y_true))
        mcc = float(matthews_corrcoef(y_true.astype(int), preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        try:
            auroc = float(roc_auc_score(y_true, ensemble_probs))
        except ValueError:
            auroc = 0.0

        preds_at_05 = (ensemble_probs >= 0.5).astype(int)
        mcc_at_05 = float(matthews_corrcoef(y_true.astype(int), preds_at_05)) if len(np.unique(y_true)) > 1 else 0.0

        tqdm.write(
            f"\n  ┌── Ensemble ({len(all_probs)} seeds) ──────────────────\n"
            f"  │ MCC={mcc:.4f} (thr={thr:.3f})  MCC@0.5={mcc_at_05:.4f}\n"
            f"  │ AUROC={auroc:.4f}  F1={f1:.4f}\n"
            f"  │ Precision={prec:.4f}  Recall={rec:.4f}  Acc={acc:.4f}\n"
            f"  │ Threshold={thr:.4f} (mean of val thresholds)\n"
            f"  └─────────────────────────────────────────"
        )

        # Save ensemble results
        ensemble_path = os.path.join(level_dir, "ensemble_results.json")
        ensemble_data = {
            "n_seeds": len(all_probs),
            "mcc": round(mcc, 6),
            "mcc_at_05": round(mcc_at_05, 6),
            "auroc": round(auroc, 6),
            "f1": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "accuracy": round(acc, 6),
            "threshold": round(thr, 4),
            "threshold_source": "mean_val_thresholds",
        }
        with open(ensemble_path, "w") as fh:
            json.dump(ensemble_data, fh, indent=2)
        tqdm.write(f"  Ensemble results saved: {ensemble_path}")

