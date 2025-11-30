"""
Cross-Attention Models for Protein-Ligand Affinity Prediction.

Single Responsibility: Model architecture definitions only.
Open/Closed: Easy to extend with new architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CrossAttentionModel(nn.Module):
    """
    Basic Cross-Attention model for protein-ligand affinity prediction.
    
    Architecture:
    1. Protein projection: Linear -> LayerNorm -> ReLU
    2. Ligand projection: Linear -> LayerNorm -> ReLU
    3. Cross-attention: Protein attends to Ligand
    4. Pooling: Mean pooling over sequence
    5. Head: Concatenated features -> MLP -> (affinity, activity)
    
    Args:
        protein_dim: Input dimension of protein embeddings
        ligand_dim: Input dimension of ligand embeddings
        hidden_dim: Hidden dimension for projections and attention
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers (unused in basic model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,  # Unused in basic model
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        
        # Projections
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Separate prediction heads
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            protein: Protein embeddings (batch, seq_len, protein_dim)
            ligand: Ligand embeddings (batch, seq_len, ligand_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with:
                - regression: Predicted pChEMBL values (batch,)
                - classification: Predicted activity logits (batch,)
                - attention: Attention weights if return_attention=True
        """
        # Project to common space
        protein_proj = self.protein_proj(protein)
        ligand_proj = self.ligand_proj(ligand)
        
        # Cross-attention: protein attends to ligand
        attn_output, attn_weights = self.cross_attn(
            protein_proj, ligand_proj, ligand_proj,
            need_weights=return_attention
        )
        
        # Residual + norm
        protein_out = self.norm(protein_proj + attn_output)
        
        # Pool over sequence dimension
        protein_pooled = protein_out.mean(dim=1)
        ligand_pooled = ligand_proj.mean(dim=1)
        
        # Concatenate and predict
        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)
        
        regression = self.regression_head(combined).squeeze(-1)
        classification = self.classification_head(combined).squeeze(-1)
        
        outputs = {
            'regression': regression,
            'classification': classification
        }
        
        if return_attention:
            outputs['attention'] = attn_weights
        
        return outputs


class ImprovedCrossAttentionModel(nn.Module):
    """
    Improved Cross-Attention model with deeper architecture.
    
    Improvements over basic model:
    - Deeper projection layers with GELU activation
    - Multiple cross-attention layers
    - Gradient clipping friendly architecture
    - Better regularization
    
    Args:
        protein_dim: Input dimension of protein embeddings
        ligand_dim: Input dimension of ligand embeddings
        hidden_dim: Hidden dimension for projections and attention
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Deep protein projection
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Deep ligand projection
        self.ligand_proj = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multiple cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout * 0.5
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers after attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout * 0.5)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Regression head (pChEMBL prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classification head (active/inactive)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple attention layers.
        
        Args:
            protein: Protein embeddings (batch, seq_len, protein_dim)
            ligand: Ligand embeddings (batch, seq_len, ligand_dim)
            return_attention: Whether to return last layer attention weights
            
        Returns:
            Dictionary with:
                - regression: Predicted pChEMBL values (batch,)
                - classification: Predicted activity logits (batch,)
                - attention: Last layer attention weights if return_attention=True
        """
        # Project to common space
        protein = self.protein_proj(protein)
        ligand = self.ligand_proj(ligand)
        
        # Multiple cross-attention layers
        last_attn_weights = None
        for i, (attn, norm, ffn, ffn_norm) in enumerate(zip(
            self.cross_attn_layers,
            self.layer_norms,
            self.ffn_layers,
            self.ffn_norms
        )):
            # Cross-attention
            attn_output, attn_weights = attn(
                protein, ligand, ligand,
                need_weights=(return_attention and i == self.num_layers - 1)
            )
            protein = norm(protein + attn_output)
            
            # Feed-forward
            ffn_output = ffn(protein)
            protein = ffn_norm(protein + ffn_output)
            
            if return_attention and i == self.num_layers - 1:
                last_attn_weights = attn_weights
        
        # Pool over sequence dimension
        protein_pooled = protein.mean(dim=1)
        ligand_pooled = ligand.mean(dim=1)
        
        # Concatenate and predict
        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)
        
        regression = self.regression_head(combined).squeeze(-1)
        classification = self.classification_head(combined).squeeze(-1)
        
        outputs = {
            'regression': regression,
            'classification': classification
        }
        
        if return_attention:
            outputs['attention'] = last_attn_weights
        
        return outputs
    
    def get_attention_weights(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            protein: Protein embeddings (batch, seq_len, protein_dim)
            ligand: Ligand embeddings (batch, seq_len, ligand_dim)
            
        Returns:
            attention: Attention weights (batch, num_heads, protein_len, ligand_len)
        """
        outputs = self.forward(protein, ligand, return_attention=True)
        return outputs.get('attention')
