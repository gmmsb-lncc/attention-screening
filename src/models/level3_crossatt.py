"""Level 3: Cross-Attention Model with Strong Regularization.

Improved architecture with:
- Stronger dropout (0.3 encoder, 0.5 classifier)
- Weight decay (1e-4)
- Label smoothing (0.1)
- Gradient clipping (1.0)
- Layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProteinEncoder(nn.Module):
    """Simplified encoder for protein embeddings - single layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        return self.encoder(x)


class LigandEncoder(nn.Module):
    """Simplified encoder for ligand embeddings - single layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        return self.encoder(x)


class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention between protein and ligand."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Protein attends to ligand (P→L)
        self.prot_to_lig = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Ligand attends to protein (L→P)
        self.lig_to_prot = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_prot = nn.LayerNorm(hidden_dim)
        self.norm_lig = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        prot: torch.Tensor,
        lig: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            prot: [batch, prot_len, hidden_dim]
            lig: [batch, lig_len, hidden_dim]
            prot_mask: [batch, prot_len] (True = valid, False = padding)
            lig_mask: [batch, lig_len] (True = valid, False = padding)
        
        Returns:
            prot_out: [batch, prot_len, hidden_dim]
            lig_out: [batch, lig_len, hidden_dim]
        """
        # Convert masks for MultiheadAttention (expects key_padding_mask: True=ignore)
        prot_key_mask = ~prot_mask if prot_mask is not None else None
        lig_key_mask = ~lig_mask if lig_mask is not None else None
        
        # Protein attends to ligand (query=prot, key/value=lig)
        prot_attn, _ = self.prot_to_lig(
            query=prot,
            key=lig,
            value=lig,
            key_padding_mask=lig_key_mask,
            need_weights=False
        )
        prot_out = self.norm_prot(prot + prot_attn)  # Residual connection
        
        # Ligand attends to protein (query=lig, key/value=prot)
        lig_attn, _ = self.lig_to_prot(
            query=lig,
            key=prot,
            value=prot,
            key_padding_mask=prot_key_mask,
            need_weights=False
        )
        lig_out = self.norm_lig(lig + lig_attn)  # Residual connection
        
        return prot_out, lig_out


class Level3CrossAttModel(nn.Module):
    """Level 3: Cross-Attention model with strong regularization."""
    
    def __init__(
        self,
        protein_dim: int = 320,
        ligand_dim: int = 768,
        hidden_dim: int = 256,  # Reduced from 512
        num_heads: int = 8,
        encoder_dropout: float = 0.4,  # Increased from 0.3
        attention_dropout: float = 0.2,  # Increased from 0.1
        classifier_dropout: float = 0.5
    ):
        super().__init__()
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.protein_encoder = ProteinEncoder(protein_dim, hidden_dim, encoder_dropout)
        self.ligand_encoder = LigandEncoder(ligand_dim, hidden_dim, encoder_dropout)
        
        # Cross-attention
        self.cross_attention = CrossAttentionLayer(hidden_dim, num_heads, attention_dropout)
        
        # Classifier head with strong dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            protein: [batch, prot_len, protein_dim]
            ligand: [batch, lig_len, ligand_dim]
            protein_mask: [batch, prot_len] (True = valid)
            ligand_mask: [batch, lig_len] (True = valid)
        
        Returns:
            logits: [batch, 1]
        """
        # Encode
        prot_encoded = self.protein_encoder(protein)  # [batch, prot_len, hidden]
        lig_encoded = self.ligand_encoder(ligand)     # [batch, lig_len, hidden]
        
        # Cross-attention
        prot_attended, lig_attended = self.cross_attention(
            prot_encoded, lig_encoded, protein_mask, ligand_mask
        )
        
        # Masked mean pooling
        if protein_mask is not None:
            prot_mask_expanded = protein_mask.unsqueeze(-1).float()  # [batch, prot_len, 1]
            prot_sum = (prot_attended * prot_mask_expanded).sum(dim=1)
            prot_count = prot_mask_expanded.sum(dim=1).clamp(min=1e-9)
            prot_pooled = prot_sum / prot_count  # [batch, hidden]
        else:
            prot_pooled = prot_attended.mean(dim=1)
        
        if ligand_mask is not None:
            lig_mask_expanded = ligand_mask.unsqueeze(-1).float()
            lig_sum = (lig_attended * lig_mask_expanded).sum(dim=1)
            lig_count = lig_mask_expanded.sum(dim=1).clamp(min=1e-9)
            lig_pooled = lig_sum / lig_count
        else:
            lig_pooled = lig_attended.mean(dim=1)
        
        # Concatenate and classify
        combined = torch.cat([prot_pooled, lig_pooled], dim=-1)  # [batch, hidden*2]
        logits = self.classifier(combined)  # [batch, 1]
        
        return logits


# Alias for backward compatibility
CrossAttnLite = Level3CrossAttModel
