"""Level 5-Lite: Complete model architecture.

Cross-Attention with Pre-calculated Embeddings.

Architecture:
1. Protein Encoder (ESM-2 matrices → Transformer)
2. Ligand Encoder (MoLFormer matrices → Transformer)
3. Bidirectional Cross-Attention
4. Attention Pooling
5. Classifier Head
"""

import torch
import torch.nn as nn

from .encoders import ProteinEncoder, LigandEncoder
from .attention import BidirectionalCrossAttention, AttentionPooling
from .classifier import ClassifierHead


class Level5LiteModel(nn.Module):
    """Level 5-Lite: Cross-Attention with Pre-calculated Embeddings.
    
    This model combines:
    - Pre-calculated ESM-2 protein embeddings (per-residue)
    - Pre-calculated MoLFormer ligand embeddings (per-token)
    - Simple projection layers (no redundant Transformers!)
    - Bidirectional cross-attention for interaction modeling
    - Attention pooling for sequence-to-vector aggregation
    - MLP classifier for binary prediction
    
    FIXED: Removed redundant Transformer encoders after pre-trained models.
    
    Target performance: MCC 0.45-0.52 (vs Level 1 baseline: 0.428)
    Reduced parameters: ~8M (vs previous 22M)
    """

    def __init__(
        self,
        protein_input_dim: int = 320,
        ligand_input_dim: int = 768,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.2,
        classifier_dropout: float = 0.2,
    ):
        """Initialize Level5LiteModel.

        Args:
            protein_input_dim: ESM-2 embedding dimension (320/640/1280)
            ligand_input_dim: MoLFormer embedding dimension (768)
            hidden_dim: Hidden dimension for all layers
            num_cross_attn_layers: Number of cross-attention blocks
            num_heads: Number of attention heads
            dropout: Dropout for projections and attention
            classifier_dropout: Dropout for classifier head
        """
        super().__init__()
        
        self.protein_input_dim = protein_input_dim
        self.ligand_input_dim = ligand_input_dim
        self.hidden_dim = hidden_dim
        
        # Simple projection encoders (no redundant Transformers)
        self.protein_encoder = ProteinEncoder(
            input_dim=protein_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        self.ligand_encoder = LigandEncoder(
            input_dim=ligand_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Attention pooling
        self.protein_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.ligand_pool = AttentionPooling(hidden_dim, num_heads, dropout)

        # Classifier (optimized 2-layer MLP with BatchNorm)
        self.classifier = ClassifierHead(
            input_dim=hidden_dim * 2,  # concat protein + ligand (256 + 256 = 512)
            hidden_dim=128,
            dropout=classifier_dropout,
        )

        # Regression head (for compatibility with existing training loop)
        self.regression_head = nn.Linear(hidden_dim * 2, 1)
        
    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor = None,
        ligand_mask: torch.Tensor = None,
    ) -> dict:
        """Forward pass.
        
        Args:
            protein_matrix: [batch, protein_len, protein_input_dim] ESM-2 embeddings
            ligand_matrix: [batch, ligand_len, ligand_input_dim] MoLFormer embeddings
            protein_mask: [batch, protein_len] mask where 1=real token, 0=padding
            ligand_mask: [batch, ligand_len] mask where 1=real token, 0=padding
        
        Returns:
            dict with:
                - 'classification': [batch, 1] classification logits
                - 'regression': [batch, 1] regression predictions (for compatibility)
        """
        # Convert masks: PyTorch attention expects True for padding, False for real tokens
        # Input masks are 1=real, 0=padding, so we need to invert
        if protein_mask is not None:
            # Convert 0/1 float mask to boolean: True where padding (mask == 0)
            protein_attn_mask = (protein_mask == 0)
        else:
            protein_attn_mask = None
            
        if ligand_mask is not None:
            ligand_attn_mask = (ligand_mask == 0)
        else:
            ligand_attn_mask = None
        
        # Encode each modality
        protein = self.protein_encoder(protein_matrix, protein_attn_mask)
        ligand = self.ligand_encoder(ligand_matrix, ligand_attn_mask)
        
        # Apply cross-attention layers
        for cross_attn in self.cross_attn_layers:
            protein, ligand = cross_attn(
                protein, ligand, protein_attn_mask, ligand_attn_mask
            )
        
        # Pool to fixed-size vectors
        protein_vec = self.protein_pool(protein, protein_attn_mask)  # [B, hidden_dim]
        ligand_vec = self.ligand_pool(ligand, ligand_attn_mask)      # [B, hidden_dim]
        
        # Concatenate
        combined = torch.cat([protein_vec, ligand_vec], dim=-1)  # [B, 2*hidden_dim]
        
        # Classification and regression
        logits = self.classifier(combined)
        regression = self.regression_head(combined)
        
        return {
            'classification': logits,
            'regression': regression,
        }
    
    def get_attention_weights(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: torch.Tensor = None,
        ligand_mask: torch.Tensor = None,
    ) -> dict:
        """Get attention weights for interpretability.
        
        Returns a dict with pooling attention weights for protein and ligand.
        Useful for visualizing which residues/atoms are most important.
        
        Args:
            protein_matrix: [batch, protein_len, protein_input_dim] ESM-2 embeddings
            ligand_matrix: [batch, ligand_len, ligand_input_dim] MoLFormer embeddings
            protein_mask: [batch, protein_len] mask where 1=real token, 0=padding
            ligand_mask: [batch, ligand_len] mask where 1=real token, 0=padding
        """
        # Convert masks (same as forward())
        if protein_mask is not None:
            protein_attn_mask = (protein_mask == 0)
        else:
            protein_attn_mask = None
            
        if ligand_mask is not None:
            ligand_attn_mask = (ligand_mask == 0)
        else:
            ligand_attn_mask = None
        
        # Encode
        protein = self.protein_encoder(protein_matrix, protein_attn_mask)
        ligand = self.ligand_encoder(ligand_matrix, ligand_attn_mask)
        
        # Cross-attention
        for cross_attn in self.cross_attn_layers:
            protein, ligand = cross_attn(
                protein, ligand, protein_attn_mask, ligand_attn_mask
            )
        
        # Get pooling attention weights
        batch_size = protein.size(0)
        
        # Protein pooling attention
        p_query = self.protein_pool.query.expand(batch_size, -1, -1)
        _, p_attn = self.protein_pool.attention(
            query=p_query,
            key=protein,
            value=protein,
            key_padding_mask=protein_attn_mask,
            average_attn_weights=True,
        )
        
        # Ligand pooling attention
        l_query = self.ligand_pool.query.expand(batch_size, -1, -1)
        _, l_attn = self.ligand_pool.attention(
            query=l_query,
            key=ligand,
            value=ligand,
            key_padding_mask=ligand_attn_mask,
            average_attn_weights=True,
        )
        
        return {
            'protein_attention': p_attn.squeeze(1),  # [B, protein_len]
            'ligand_attention': l_attn.squeeze(1),   # [B, ligand_len]
        }
    
    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            'protein_encoder': sum(p.numel() for p in self.protein_encoder.parameters()),
            'ligand_encoder': sum(p.numel() for p in self.ligand_encoder.parameters()),
            'cross_attention': sum(p.numel() for p in self.cross_attn_layers.parameters()),
            'protein_pool': sum(p.numel() for p in self.protein_pool.parameters()),
            'ligand_pool': sum(p.numel() for p in self.ligand_pool.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }
        
        return {
            'total': total,
            'trainable': trainable,
            'breakdown': breakdown,
        }
