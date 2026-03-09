"""Level 5b-DA model: Attention Pooling + GRL + Domain Discriminator.

Architecture
------------
Identical to Level 3 (benchmark ``_AttentionPoolingModel``) for the
feature extraction pathway — linear projection followed by learnable
attention pooling, NO cross-attention.  Two components are added:

  1. **GradientReversalLayer** — identity on forward, negates gradient
     on backward (forces scaffold-invariant features).
  2. **DomainDiscriminator** — MLP predicting scaffold-cluster id.

The ``forward()`` returns a dict compatible with the experiment.py
trainer: ``{'classification', 'regression' (None), 'domain_logits'}``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..level5_lite.attention import AttentionPooling
from ..level5_da.domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
)


class Level5bDAModel(nn.Module):
    """Level 5b-DA: AttnPool + Adversarial Domain Adaptation.

    Parameters
    ----------
    protein_input_dim : int
        Dimension of per-residue protein embeddings (e.g. 320 for 8M).
    ligand_input_dim : int
        Dimension of per-token ligand embeddings (768 for MoLFormer).
    hidden_dim : int
        Shared hidden dimension after projection.
    num_heads : int
        Attention heads in the pooling layers.
    dropout : float
        Dropout in projection and pooling.
    classifier_dropout : float
        Dropout in the classification head.
    num_domains : int
        Number of scaffold clusters (domain labels).
    domain_hidden_dim : int
        Hidden size of the domain discriminator MLP.
    domain_dropout : float
        Dropout in the domain discriminator.
    grl_lambda : float
        Initial GRL strength (updated externally each epoch).
    """

    def __init__(
        self,
        protein_input_dim: int = 320,
        ligand_input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        classifier_dropout: float = 0.2,
        num_domains: int = 16,
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.3,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()

        self._protein_input_dim = protein_input_dim
        self._ligand_input_dim = ligand_input_dim
        self._hidden_dim = hidden_dim
        self.num_domains = num_domains

        # --- Projection layers (identical to Level 3 / Level 4) ---
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

        # --- Attention pooling (same as Level 3 and Level 4) ---
        self.protein_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.ligand_pool = AttentionPooling(hidden_dim, num_heads, dropout)

        # --- Classification head (BCE-trainable) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 1),
        )

        # --- Domain adaptation branch ---
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=hidden_dim * 2,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            dropout=domain_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.protein_proj, self.ligand_proj):
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.xavier_uniform_(self.classifier[3].weight)
        nn.init.zeros_(self.classifier[3].bias)

    # ---- properties (for compatibility with runner) ----

    @property
    def protein_input_dim(self) -> int:
        return self._protein_input_dim

    @property
    def ligand_input_dim(self) -> int:
        return self._ligand_input_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def set_grl_lambda(self, lam: float) -> None:
        """Update GRL strength (call once per epoch)."""
        self.grl.lam = lam

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor | None = None,
        ligand_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
        """Forward pass.

        Returns
        -------
        dict with keys:
          - ``'classification'``: ``[B, 1]`` logits
          - ``'regression'``: ``None`` (classification-only model)
          - ``'domain_logits'``: ``[B, num_domains]``
          - ``'features'``: ``[B, 2*hidden_dim]`` (only if *return_features*)
        """
        # Invert masks for PyTorch attention (True = padding)
        p_mask = (protein_mask == 0) if protein_mask is not None else None
        l_mask = (ligand_mask == 0) if ligand_mask is not None else None

        # Project
        protein = self.protein_proj(protein_matrix)  # [B, prot_len, hidden]
        ligand = self.ligand_proj(ligand_matrix)      # [B, lig_len, hidden]

        # Attention pool (NO cross-attention — this is the L3 path)
        protein_vec = self.protein_pool(protein, p_mask)  # [B, hidden]
        ligand_vec = self.ligand_pool(ligand, l_mask)      # [B, hidden]

        combined = torch.cat([protein_vec, ligand_vec], dim=-1)  # [B, 2*hidden]

        # Classification
        classification = self.classifier(combined)  # [B, 1]

        # Domain adversarial branch (GRL applied to same combined vector)
        reversed_features = self.grl(combined)
        domain_logits = self.domain_discriminator(reversed_features)

        out: dict = {
            "classification": classification,
            "regression": None,
            "domain_logits": domain_logits,
        }
        if return_features:
            out["features"] = combined

        return out
