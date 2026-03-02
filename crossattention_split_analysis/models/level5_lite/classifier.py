"""Classifier head for Level 5-Lite."""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP for binary classification.
    
    Architecture with strong regularization:
    - LayerNorm after each layer (stability)
    - Dropout 0.3 (prevents overfitting)
    - GELU activation (better than ReLU for transformers)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        """Initialize ClassifierHead.
        
        Args:
            input_dim: Input dimension (protein_dim + ligand_dim)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        
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
