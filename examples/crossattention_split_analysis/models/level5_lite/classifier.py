"""Classifier head for Level 5-Lite."""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP for binary classification.

    OPTIMIZED Architecture:
    - Two hidden layers for better representation (512 → 128)
    - BatchNorm for stability and faster convergence
    - Moderate dropout (0.2) to prevent overfitting
    - ReLU activation (simple and effective)
    """

    def __init__(
        self,
        input_dim: int = 512,  # 256 + 256 = 512 (hidden_dim * 2)
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize ClassifierHead.

        Args:
            input_dim: Input dimension (protein_dim + ligand_dim)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, input_dim]

        Returns:
            [batch, 1] logits (not probabilities)
        """
        return self.classifier(x)
