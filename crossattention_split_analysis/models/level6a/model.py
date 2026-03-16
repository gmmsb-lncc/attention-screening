"""Level 6a model: CrossAttn + BAN fusion + GRL.

Architecture
------------
  1. Projection (protein_dim → hidden, ligand_dim → hidden)
  2. Bidirectional Cross-Attention (N layers) — enriches representations
  3. **BANLayer** — bilinear attention-weighted pooling that fuses
     protein and ligand sequences into a single interaction vector.
     Replaces the separate AttentionPooling + concatenation of L5a.
  4. Classification head
  5. GRL + Domain Discriminator (scaffold-invariance)

The BAN operates on the full sequence representations [B, seq_len, hidden]
produced by cross-attention, simultaneously computing all pairwise
protein-ligand attention scores and pooling them into a fixed-size vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from ..level5_lite.encoders import ProteinEncoder, LigandEncoder
from ..level5_lite.attention import BidirectionalCrossAttention
from ..ban import BANLayer
from ..level5_da.domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
)


class Level6aModel(nn.Module):
    """Level 6a: CrossAttn + BAN + GRL.

    Parameters
    ----------
    protein_input_dim : int
        Per-residue protein embedding dimension.
    ligand_input_dim : int
        Per-token ligand embedding dimension.
    hidden_dim : int
        Shared hidden dimension after projection.
    num_cross_attn_layers : int
        Number of bidirectional cross-attention layers.
    num_heads : int
        Attention heads in cross-attention.
    dropout : float
        Dropout in encoders and attention.
    ban_heads : int
        Number of bilinear attention heads in BAN.
    ban_k : int
        Low-rank factorisation rank for BAN.
    classifier_dropout : float
        Dropout in classification head.
    num_domains : int
        Number of scaffold clusters.
    domain_hidden_dim : int
        Hidden size of domain discriminator.
    domain_dropout : float
        Dropout in domain discriminator.
    grl_lambda : float
        Initial GRL strength.
    """

    def __init__(
        self,
        protein_input_dim: int = 320,
        ligand_input_dim: int = 768,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2,
        ban_heads: int = 2,
        ban_k: int = 3,
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

        # --- Encoders (same as Level 4/5) ---
        self.protein_encoder = ProteinEncoder(protein_input_dim, hidden_dim, dropout)
        self.ligand_encoder = LigandEncoder(ligand_input_dim, hidden_dim, dropout)

        # --- Cross-attention layers (same as Level 4/5) ---
        self.cross_attention_layers = nn.ModuleList([
            BidirectionalCrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # --- BAN fusion (replaces AttnPool + concat) ---
        self.ban = weight_norm(
            BANLayer(
                v_dim=hidden_dim,
                q_dim=hidden_dim,
                h_dim=hidden_dim,
                h_out=ban_heads,
                act="ReLU",
                dropout=dropout,
                k=ban_k,
            ),
            name="h_mat",
            dim=None,
        )

        # --- Classification head ---
        # BAN outputs [B, hidden_dim] (not 2*hidden_dim like concat)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Domain adaptation branch ---
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=hidden_dim,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            dropout=domain_dropout,
        )

    # ---- properties ----
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
          - ``'classification'``: [B, 1] logits
          - ``'regression'``: None
          - ``'domain_logits'``: [B, num_domains]
          - ``'features'``: [B, hidden_dim] (only if return_features)
        """
        # Invert masks for PyTorch attention (True = padding)
        p_attn_mask = (protein_mask == 0) if protein_mask is not None else None
        l_attn_mask = (ligand_mask == 0) if ligand_mask is not None else None

        # Encode
        protein = self.protein_encoder(protein_matrix)  # [B, prot_len, hidden]
        ligand = self.ligand_encoder(ligand_matrix)      # [B, lig_len, hidden]

        # Cross-attention enrichment
        for cross_attn in self.cross_attention_layers:
            protein, ligand = cross_attn(
                protein, ligand, p_attn_mask, l_attn_mask
            )

        # BAN fusion (replaces AttnPool + concat)
        # BAN expects masks as 1=real, 0=pad (same as input masks)
        fused, _att_maps = self.ban(
            protein, ligand, softmax=True,
            v_mask=protein_mask, q_mask=ligand_mask,
        )  # fused: [B, hidden_dim]

        # Classification
        classification = self.classifier(fused)  # [B, 1]

        # Domain adversarial branch
        reversed_features = self.grl(fused)
        domain_logits = self.domain_discriminator(reversed_features)

        out: dict = {
            "classification": classification,
            "regression": None,
            "domain_logits": domain_logits,
        }
        if return_features:
            out["features"] = fused

        return out
