"""Attention modules for Level 5-Lite.

Contains:
- BidirectionalCrossAttention: Cross-attention between protein and ligand
- AttentionPooling: Learnable query pooling
"""

import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between protein and ligand.
    
    Scientific justification:
    - Protein → Ligand: which chemical groups the binding pocket "sees"
    - Ligand → Protein: which residues the pharmacophore "sees"
    - Bidirectional captures the complementarity of the interaction
    
    Reference: TargetFormer (Zhang et al., Nature Comm 2023)
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize BidirectionalCrossAttention.
        
        Args:
            hidden_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Protein queries ligand
        self.protein_to_ligand = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Ligand queries protein
        self.ligand_to_protein = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms (Pre-LN style)
        self.norm_p_pre = nn.LayerNorm(hidden_dim)
        self.norm_l_pre = nn.LayerNorm(hidden_dim)
        
        # Feed-forward after cross-attention
        self.ffn_p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm_p_ffn = nn.LayerNorm(hidden_dim)
        self.norm_l_ffn = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: torch.Tensor = None,
        ligand_mask: torch.Tensor = None,
    ) -> tuple:
        """Forward pass.
        
        Args:
            protein: [batch, protein_len, hidden_dim]
            ligand: [batch, ligand_len, hidden_dim]
            protein_mask: [batch, protein_len] padding mask (True = pad)
            ligand_mask: [batch, ligand_len] padding mask (True = pad)
        
        Returns:
            protein_out: [batch, protein_len, hidden_dim] - protein enriched with ligand info
            ligand_out: [batch, ligand_len, hidden_dim] - ligand enriched with protein info
        """
        # Protein attends to ligand
        p_norm = self.norm_p_pre(protein)
        p_cross, _ = self.protein_to_ligand(
            query=p_norm,
            key=ligand,
            value=ligand,
            key_padding_mask=ligand_mask,
        )
        protein = protein + p_cross  # Residual connection
        protein = protein + self.ffn_p(self.norm_p_ffn(protein))
        
        # Ligand attends to protein
        l_norm = self.norm_l_pre(ligand)
        l_cross, _ = self.ligand_to_protein(
            query=l_norm,
            key=protein,
            value=protein,
            key_padding_mask=protein_mask,
        )
        ligand = ligand + l_cross  # Residual connection
        ligand = ligand + self.ffn_l(self.norm_l_ffn(ligand))
        
        return protein, ligand


class AttentionPooling(nn.Module):
    """Pooling with learnable query.
    
    Scientific justification:
    - Mean pooling treats all tokens equally (suboptimal)
    - Attention pooling learns which tokens are important
    - For proteins: binding pocket residues receive more weight
    - For ligands: pharmacophores receive more weight
    
    Reference: Set Transformer (Lee et al., ICML 2019)
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize AttentionPooling.
        
        Args:
            hidden_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable query (1 token that "asks" for the summary)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] padding mask (True = pad)
        
        Returns:
            [batch, hidden_dim] pooled representation
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        # Attention pooling
        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=mask,
        )
        
        pooled = self.norm(pooled)
        
        return pooled.squeeze(1)  # [B, D]
