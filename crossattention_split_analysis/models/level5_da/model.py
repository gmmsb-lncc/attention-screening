"""Level 5-DA model: Level5LiteModel + Gradient Reversal + Domain Discriminator.

Architecture
------------
Identical to Level5LiteModel (Level 4 in the benchmark) for all
components up to and including the pooled ``combined`` vector.
Two new components are added on top of that vector:

  1. **GradientReversalLayer** — identity on forward, negates gradient
     on backward.
  2. **DomainDiscriminator** — MLP predicting scaffold cluster id.

The ``forward()`` method returns the same dict as Level5LiteModel
plus an extra ``'domain_logits'`` key when domain labels are available.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..level5_lite.model import Level5LiteModel
from .domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
)


class Level5DAModel(nn.Module):
    """Level 5-DA: Cross-Attention + Adversarial Domain Adaptation.

    Wraps a frozen-architecture Level5LiteModel and adds a GRL +
    domain-discriminator branch.  The backbone is *not* shared via
    inheritance to keep Level 4 code untouched; instead we
    instantiate a Level5LiteModel internally and delegate.

    Parameters
    ----------
    num_domains : int
        Number of scaffold clusters (domain labels).
    domain_hidden_dim : int
        Hidden size of the domain discriminator MLP.
    domain_dropout : float
        Dropout in the domain discriminator.
    grl_lambda : float
        Initial GRL strength (updated externally each epoch).
    **backbone_kwargs
        Forwarded to ``Level5LiteModel.__init__()``.
    """

    def __init__(
        self,
        num_domains: int = 16,
        domain_hidden_dim: int = 256,
        domain_dropout: float = 0.3,
        grl_lambda: float = 1.0,
        # --- backbone kwargs (same as Level5LiteModel) ---
        protein_input_dim: int = 320,
        ligand_input_dim: int = 768,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_domains = num_domains

        # --- Backbone (identical to Level 4) ---
        self.backbone = Level5LiteModel(
            protein_input_dim=protein_input_dim,
            ligand_input_dim=ligand_input_dim,
            hidden_dim=hidden_dim,
            num_cross_attn_layers=num_cross_attn_layers,
            num_heads=num_heads,
            dropout=dropout,
            classifier_dropout=classifier_dropout,
        )

        # --- Domain adaptation branch ---
        self.grl = GradientReversalLayer(lam=grl_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=hidden_dim * 2,  # same as classifier input
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            dropout=domain_dropout,
        )

    # ---- delegate common properties to backbone ----
    @property
    def protein_input_dim(self) -> int:
        return self.backbone.protein_input_dim

    @property
    def ligand_input_dim(self) -> int:
        return self.backbone.ligand_input_dim

    @property
    def hidden_dim(self) -> int:
        return self.backbone.hidden_dim

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

        Returns the same dict as ``Level5LiteModel.forward()`` plus:
          - ``'domain_logits'``: ``[B, num_domains]`` scaffold-cluster
            prediction (always present).

        The backbone is called with ``return_features=True`` so we
        always have access to the ``combined`` vector for the GRL.
        """
        out = self.backbone(
            protein_matrix,
            ligand_matrix,
            protein_mask,
            ligand_mask,
            return_features=True,
        )

        # Override regression to None — L5a is classification-only.
        # The backbone (Level5LiteModel) returns a regression head for
        # historical reasons, but the DA variant must not use it so that
        # the trainer follows the classification-only BCE path.
        out["regression"] = None

        features = out["features"]  # [B, 2*hidden_dim]

        # Domain adversarial branch
        reversed_features = self.grl(features)
        domain_logits = self.domain_discriminator(reversed_features)
        out["domain_logits"] = domain_logits

        if not return_features:
            out.pop("features", None)

        return out
