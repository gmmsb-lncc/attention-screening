"""Level 6b model: BAN fusion + GRL (no cross-attention).

Architecture
------------
  1. Projection (protein_dim → hidden, ligand_dim → hidden)
  2. **BANLayer** — bilinear attention directly on projected sequences.
     No cross-attention enrichment — isolates the BAN contribution.
  3. Classification head
  4. GRL + Domain Discriminator (scaffold-invariance)

Comparison with Level 6a:
  - Level 6a: Proj → CrossAttn → BAN → classifier + GRL
  - Level 6b: Proj → BAN → classifier + GRL (no cross-attention)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from ..ban import BANLayer
from ..level5_da.domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
)


class Level6bModel(nn.Module):
    """Level 6b: BAN + GRL (no cross-attention).

    Parameters
    ----------
    protein_input_dim : int
        Per-residue protein embedding dimension.
    ligand_input_dim : int
        Per-token ligand embedding dimension.
    hidden_dim : int
        Shared hidden dimension after projection.
    num_heads : int
        (Unused — kept for interface consistency with other levels.)
    dropout : float
        Dropout in projections and BAN.
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

        # --- BAN fusion (replaces AttnPool + concat of L5b) ---
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
          - ``'features'``: [B, hidden_dim] (only if return_features)
        """
        # Project
        protein = self.protein_proj(protein_matrix)  # [B, prot_len, hidden]
        ligand = self.ligand_proj(ligand_matrix)      # [B, lig_len, hidden]

        # BAN fusion (directly on projected sequences — no cross-attention)
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
