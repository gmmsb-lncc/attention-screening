"""Classifier head for Level 5-Lite."""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP for binary classification.
    
    SIMPLIFIED: Reduced dropout and made shallower.
    
    Architecture:
    - Single hidden layer (not two)
    - Moderate dropout (0.2, not 0.3)
    - ReLU activation (simpler, works well for final layers)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        """Initialize ClassifierHead.
        
        Args:
            input_dim: Input dimension (protein_dim + ligand_dim)
            hidden_dim: Hidden layer dimension (single layer)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, input_dim]
        
        Returns:
            [batch, 1] logits (not probabilities)
        """
        return self.classifier(x)
