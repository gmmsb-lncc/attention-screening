"""Level 6b model: AttnPool + BAN fusion + GRL (no cross-attention).

Architecture
------------
  1. Projection (protein_dim → hidden, ligand_dim → hidden)
  2. **AttentionPooling** — per-modality learned pooling (seq → vec).
  3. **BANLayer** — bilinear attention directly on projected sequences,
     producing a cross-modal interaction vector.
  4. Concatenation of AttnPool vectors + BAN vector → [B, 3*hidden]
  5. Classification head
  6. GRL + Domain Discriminator (scaffold-invariance)

Comparison with Level 6a:
  - Level 6a: Proj → CrossAttn → BAN → classifier + GRL
  - Level 6b: Proj → AttnPool + BAN → classifier + GRL (no cross-attention)

The AttnPool captures uni-modal summaries while BAN captures the
bilinear cross-modal interaction, giving a richer representation
(3*hidden_dim) than either alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from ..ban import BANLayer
from ..level5_lite.attention import AttentionPooling
from ..level5_da.domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
)


class Level6bModel(nn.Module):
    """Level 6b: AttnPool + BAN + GRL (no cross-attention).

    Parameters
    ----------
    protein_input_dim : int
        Per-residue protein embedding dimension.
    ligand_input_dim : int
        Per-token ligand embedding dimension.
    hidden_dim : int
        Shared hidden dimension after projection.
    num_heads : int
        Attention heads in AttentionPooling layers.
    dropout : float
        Dropout in projections, pooling, and BAN.
    ban_heads : int
        Number of bilinear attention heads.
    ban_k : int
        Low-rank factorisation rank.
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

        # Feature dim: AttnPool gives 2*hidden (concat), BAN gives hidden
        self._feature_dim = hidden_dim * 3

        # --- Projection layers ---
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

        # --- Attention pooling (uni-modal summaries) ---
        self.protein_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.ligand_pool = AttentionPooling(hidden_dim, num_heads, dropout)

        # --- BAN fusion (cross-modal bilinear interaction) ---
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
        # Input: cat(protein_vec, ligand_vec, ban_vec) = 3 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self._feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 1),
        )

        # --- Domain adaptation branch ---
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=self._feature_dim,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            dropout=domain_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.protein_proj, self.ligand_proj):
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.zeros_(proj[0].bias)

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
          - ``'features'``: [B, 3*hidden_dim] (only if return_features)
        """
        # Invert masks for PyTorch attention (True = padding)
        p_attn_mask = (protein_mask == 0) if protein_mask is not None else None
        l_attn_mask = (ligand_mask == 0) if ligand_mask is not None else None

        # Project
        protein = self.protein_proj(protein_matrix)  # [B, prot_len, hidden]
        ligand = self.ligand_proj(ligand_matrix)      # [B, lig_len, hidden]

        # Path A: Attention pooling (uni-modal summaries)
        protein_vec = self.protein_pool(protein, p_attn_mask)  # [B, hidden]
        ligand_vec = self.ligand_pool(ligand, l_attn_mask)      # [B, hidden]

        # Path B: BAN fusion (cross-modal bilinear interaction)
        ban_vec, _att_maps = self.ban(
            protein, ligand, softmax=True,
            v_mask=protein_mask, q_mask=ligand_mask,
        )  # ban_vec: [B, hidden]

        # Combine uni-modal + cross-modal representations
        combined = torch.cat([protein_vec, ligand_vec, ban_vec], dim=-1)  # [B, 3*hidden]

        # Classification
        classification = self.classifier(combined)  # [B, 1]

        # Domain adversarial branch
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
