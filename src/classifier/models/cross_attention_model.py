"""
CNN + Cross-Attention Model for Protein-Ligand Affinity Prediction.

This model uses per-token embedding matrices from protein and ligand encoders
to learn interaction patterns through cross-attention mechanisms.

Architecture:
    1. CNN encoders: Extract local patterns from protein/ligand sequences
    2. Cross-Attention: Model protein-ligand interactions
    3. Multi-task heads: Classification + Regression outputs

Author: DockTKinase Team
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Literal
import logging
import math

try:
    from .base_model import BaseClassifier
except ImportError:
    from base_model import BaseClassifier

try:
    from .positional_encoding import (
        SinusoidalPositionalEncoding,
        RotaryPositionalEmbedding,
        create_positional_encoding
    )
except ImportError:
    from positional_encoding import (
        SinusoidalPositionalEncoding,
        RotaryPositionalEmbedding,
        create_positional_encoding
    )

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence data.
    
    Adds positional information to embedding matrices to preserve
    spatial relationships in the sequence.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Conv1DBlock(nn.Module):
    """
    1D Convolutional block with residual connection.
    
    Extracts local patterns from sequence embeddings.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection for residual if dimensions don't match
        if use_residual and in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, 1)
            self.use_residual = True
        else:
            self.proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        residual = x
        out = self.conv(x)
        
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        
        return self.dropout(self.activation(out))


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting local patterns from embedding matrices.
    
    Takes [batch, seq_len, embed_dim] input and produces
    [batch, seq_len, hidden_dim] output with local pattern features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            self.conv_layers.append(
                Conv1DBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] padding mask (1 for valid, 0 for padding)
        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Transpose for conv1d: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return self.layer_norm(x)


class LinearEncoder(nn.Module):
    """
    Lightweight token-wise encoder with linear projections only.

    Keeps sequence length unchanged while projecting inputs to hidden_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.proj(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for protein-ligand interaction modeling.
    
    Allows protein representations to attend to ligand representations
    and vice versa, capturing interaction patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, query_len, hidden_dim] - e.g., protein
            key: [batch, key_len, hidden_dim] - e.g., ligand
            value: [batch, key_len, hidden_dim] - e.g., ligand
            key_mask: [batch, key_len] padding mask
        Returns:
            [batch, query_len, hidden_dim]
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # [batch, query_len, hidden_dim]
        K = self.k_proj(key)    # [batch, key_len, hidden_dim]
        V = self.v_proj(value)  # [batch, key_len, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch, num_heads, query_len, key_len]
        
        # Apply mask
        if key_mask is not None:
            # key_mask: [batch, key_len] -> [batch, 1, 1, key_len]
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        # [batch, num_heads, query_len, head_dim]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        
        return self.out_proj(context)


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention block with feed-forward network.
    
    Performs:
    1. Protein attends to ligand
    2. Ligand attends to protein
    3. Feed-forward transformation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Cross-attention layers
        self.protein_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.ligand_cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward networks
        self.protein_ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ligand_ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.protein_norm1 = nn.LayerNorm(hidden_dim)
        self.protein_norm2 = nn.LayerNorm(hidden_dim)
        self.ligand_norm1 = nn.LayerNorm(hidden_dim)
        self.ligand_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            protein: [batch, protein_len, hidden_dim]
            ligand: [batch, ligand_len, hidden_dim]
            protein_mask: [batch, protein_len]
            ligand_mask: [batch, ligand_len]
        Returns:
            Updated (protein, ligand) representations
        """
        # Bidirectional cross-attention in parallel from the same input states.
        protein_in = protein
        ligand_in = ligand

        protein_attn = self.protein_cross_attn(protein_in, ligand_in, ligand_in, ligand_mask)
        ligand_attn = self.ligand_cross_attn(ligand_in, protein_in, protein_in, protein_mask)

        protein = self.protein_norm1(protein_in + protein_attn)
        ligand = self.ligand_norm1(ligand_in + ligand_attn)

        protein = self.protein_norm2(protein + self.protein_ff(protein))
        ligand = self.ligand_norm2(ligand + self.ligand_ff(ligand))

        # Keep padded positions neutral to avoid representation leakage into pooling.
        if protein_mask is not None:
            protein = protein * protein_mask.unsqueeze(-1).to(protein.dtype)
        if ligand_mask is not None:
            ligand = ligand * ligand_mask.unsqueeze(-1).to(ligand.dtype)

        return protein, ligand


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for classification and regression.
    
    Takes pooled representations and produces:
    - Classification logits (binding/non-binding)
    - Regression output (binding affinity / pIC50)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Affinity value
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] pooled representation
        Returns:
            classification_logits: [batch, 1]
            regression_output: [batch, 1]
        """
        shared = self.shared(x)
        
        cls_logits = self.classifier(shared)
        reg_output = self.regressor(shared)
        
        return cls_logits, reg_output


class CrossAttentionAffinityModel(BaseClassifier):
    """
    Cross-Attention model for protein-ligand affinity prediction.
    
    Architecture:
        1. Token encoders process protein/ligand embedding matrices
           (CNN or linear projection)
        2. Cross-attention blocks model interactions
        3. Multi-task heads output classification + regression
    
    Input format:
        protein_matrix: [batch, protein_len, protein_dim]
        ligand_matrix: [batch, ligand_len, ligand_dim]
    
    Output format:
        classification_logits: [batch, 1]
        regression_output: [batch, 1]
    
    Positional Encoding Options:
        - 'sinusoidal': Classical Vaswani et al. (2017) - fixed max length
        - 'rope': Rotary Position Embedding (Su et al., 2021) - unlimited length
        - None: No positional encoding
    
    References:
        - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.
        - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary PE". arXiv:2104.09864.
    """
    
    def __init__(
        self,
        protein_dim: int = 2560,      # ESM-2 3B dimension
        ligand_dim: int = 768,        # SMI-TED dimension
        hidden_dim: int = 256,        # Internal hidden dimension
        encoder_type: Literal['cnn', 'linear'] = 'cnn',
        num_cnn_layers: int = 3,      # CNN encoder layers
        num_cross_attn_layers: int = 2,  # Cross-attention blocks
        num_heads: int = 8,           # Attention heads
        ff_dim: int = 1024,           # Feed-forward dimension
        dropout: float = 0.1,
        max_protein_len: int = 2048,  # Max protein sequence length (sinusoidal only)
        max_ligand_len: int = 512,    # Max ligand token length (sinusoidal only)
        use_positional_encoding: bool = True,
        positional_encoding_type: Literal['sinusoidal', 'rope'] = 'sinusoidal'
    ):
        """
        Initialize CrossAttentionAffinityModel.
        
        Args:
            protein_dim: Input dimension for protein embeddings (e.g., 2560 for ESM-2 3B)
            ligand_dim: Input dimension for ligand embeddings (e.g., 768 for SMI-TED)
            hidden_dim: Internal hidden dimension for all layers
            encoder_type: Token encoder type ('cnn' or 'linear')
            num_cnn_layers: Number of CNN encoder layers
            num_cross_attn_layers: Number of cross-attention blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout probability
            max_protein_len: Maximum protein sequence length (only used for sinusoidal)
            max_ligand_len: Maximum ligand token length (only used for sinusoidal)
            use_positional_encoding: Whether to use positional encoding
            positional_encoding_type: Type of positional encoding:
                - 'sinusoidal': Classical Vaswani (2017), fixed max length
                - 'rope': Rotary Position Embedding (Su et al., 2021), unlimited length
        
        Note:
            RoPE is recommended for sequences longer than max_len or when you need
            variable-length support without truncation.
        """
        # BaseClassifier expects input_dim, we'll use hidden_dim * 2 (pooled concatenation)
        super().__init__(input_dim=hidden_dim * 2)
        
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.num_cross_attn_layers = num_cross_attn_layers
        self.positional_encoding_type = positional_encoding_type

        if self.encoder_type not in {'cnn', 'linear'}:
            raise ValueError(f"Unknown encoder_type={self.encoder_type!r}. Use 'cnn' or 'linear'.")
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            if positional_encoding_type == 'rope':
                # RoPE: Rotary Position Embedding (Su et al., 2021)
                # Applied to Q/K in attention, handles any sequence length
                self.protein_pos_enc = RotaryPositionalEmbedding(
                    dim=hidden_dim,
                    max_seq_len_cached=max_protein_len  # Initial cache, auto-extends
                )
                self.ligand_pos_enc = RotaryPositionalEmbedding(
                    dim=hidden_dim,
                    max_seq_len_cached=max_ligand_len
                )
                logger.info(f"Using RoPE positional encoding (unlimited sequence length)")
            else:
                # Sinusoidal: Classical Vaswani (2017)
                self.protein_pos_enc = PositionalEncoding(hidden_dim, max_protein_len, dropout)
                self.ligand_pos_enc = PositionalEncoding(hidden_dim, max_ligand_len, dropout)
                logger.info(f"Using Sinusoidal positional encoding (max_len: protein={max_protein_len}, ligand={max_ligand_len})")
        
        # Token encoders
        if self.encoder_type == 'cnn':
            self.protein_encoder = CNNEncoder(
                input_dim=protein_dim,
                hidden_dim=hidden_dim,
                num_layers=num_cnn_layers,
                dropout=dropout
            )
            self.ligand_encoder = CNNEncoder(
                input_dim=ligand_dim,
                hidden_dim=hidden_dim,
                num_layers=num_cnn_layers,
                dropout=dropout
            )
        else:
            self.protein_encoder = LinearEncoder(
                input_dim=protein_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            self.ligand_encoder = LinearEncoder(
                input_dim=ligand_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        # Multi-task head
        self.task_head = MultiTaskHead(hidden_dim * 2, hidden_dim, dropout)
        
        # Store config for info
        self._config = {
            'protein_dim': protein_dim,
            'ligand_dim': ligand_dim,
            'hidden_dim': hidden_dim,
            'encoder_type': encoder_type,
            'num_cnn_layers': num_cnn_layers,
            'num_cross_attn_layers': num_cross_attn_layers,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'dropout': dropout,
            'positional_encoding_type': positional_encoding_type
        }
        
        logger.info(f"CrossAttentionAffinityModel initialized: {self.count_parameters():,} parameters")
    
    def encode_protein(
        self,
        protein_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode protein embedding matrix.
        
        Args:
            protein_matrix: [batch, protein_len, protein_dim]
            protein_mask: [batch, protein_len]
        Returns:
            [batch, protein_len, hidden_dim]
        """
        x = self.protein_encoder(protein_matrix, protein_mask)
        if self.use_positional_encoding:
            x = self.protein_pos_enc(x)
        return x
    
    def encode_ligand(
        self,
        ligand_matrix: torch.Tensor,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode ligand embedding matrix.
        
        Args:
            ligand_matrix: [batch, ligand_len, ligand_dim]
            ligand_mask: [batch, ligand_len]
        Returns:
            [batch, ligand_len, hidden_dim]
        """
        x = self.ligand_encoder(ligand_matrix, ligand_mask)
        if self.use_positional_encoding:
            x = self.ligand_pos_enc(x)
        return x

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Mean pool over sequence length while ignoring padded tokens.

        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] with 1 for valid tokens and 0 for padding
        Returns:
            [batch, hidden_dim]
        """
        if mask is None:
            return x.mean(dim=1)

        mask = mask.to(dtype=x.dtype).unsqueeze(-1)  # [batch, seq_len, 1]
        x_masked = x * mask
        lengths = mask.sum(dim=1).clamp_min(1.0)  # [batch, 1]
        return x_masked.sum(dim=1) / lengths
    
    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for protein-ligand affinity prediction.
        
        Args:
            protein_matrix: [batch, protein_len, protein_dim]
            ligand_matrix: [batch, ligand_len, ligand_dim]
            protein_mask: [batch, protein_len] (1=valid, 0=padding)
            ligand_mask: [batch, ligand_len] (1=valid, 0=padding)
            return_attention: If True, return attention weights
        
        Returns:
            Dict with:
                - 'classification': [batch, 1] logits
                - 'regression': [batch, 1] affinity values
                - 'protein_repr': [batch, hidden_dim] (optional)
                - 'ligand_repr': [batch, hidden_dim] (optional)
        """
        # Encode
        protein = self.encode_protein(protein_matrix, protein_mask)
        ligand = self.encode_ligand(ligand_matrix, ligand_mask)
        
        # Cross-attention blocks
        for block in self.cross_attn_blocks:
            protein, ligand = block(protein, ligand, protein_mask, ligand_mask)
        
        # Mask-aware pooling prevents padded tokens from biasing the representation.
        protein_pooled = self._masked_mean_pool(protein, protein_mask)
        ligand_pooled = self._masked_mean_pool(ligand, ligand_mask)
        
        # Concatenate
        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)
        
        # Multi-task prediction
        cls_logits, reg_output = self.task_head(combined)
        
        result = {
            'classification': cls_logits,
            'regression': reg_output,
        }
        
        if return_attention:
            result['protein_repr'] = protein_pooled
            result['ligand_repr'] = ligand_pooled
        
        return result
    
    def forward_classification(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification only (BaseClassifier interface).
        
        Returns:
            [batch, 1] classification logits
        """
        output = self.forward(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
        return output['classification']
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture information."""
        return {
            'model_type': f'CrossAttentionAffinityModel[{self.encoder_type}]',
            'config': self._config,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'encoder_params': {
                'protein_encoder': sum(p.numel() for p in self.protein_encoder.parameters()),
                'ligand_encoder': sum(p.numel() for p in self.ligand_encoder.parameters()),
            },
            'cross_attention_params': sum(
                sum(p.numel() for p in block.parameters())
                for block in self.cross_attn_blocks
            ),
            'task_head_params': sum(p.numel() for p in self.task_head.parameters())
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CrossAttentionAffinityModel':
        """Create model from configuration dictionary."""
        return cls(**config)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and regression losses.
    
    Loss = α * BCE(classification) + β * MSE(regression)
    
    With optional uncertainty weighting (Kendall et al., 2018).
    """
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        classification_pos_weight: Optional[float] = None,
        use_uncertainty_weighting: bool = False
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.classification_pos_weight = classification_pos_weight
        
        if use_uncertainty_weighting:
            # Learnable log variance parameters
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))

        if classification_pos_weight is not None and classification_pos_weight > 0:
            self.register_buffer(
                "pos_weight_tensor",
                torch.tensor([float(classification_pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight_tensor = None
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_output: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
        reg_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            cls_logits: [batch, 1] classification logits
            reg_output: [batch, 1] regression predictions
            cls_target: [batch, 1] binary labels
            reg_target: [batch, 1] affinity values
            reg_mask: [batch, 1] mask for samples with regression labels
        
        Returns:
            Dict with 'total', 'classification', 'regression' losses
        """
        # Classification loss
        if self.pos_weight_tensor is not None:
            pos_weight = self.pos_weight_tensor.to(device=cls_logits.device, dtype=cls_logits.dtype)
        else:
            pos_weight = None
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits,
            cls_target.float(),
            pos_weight=pos_weight
        )
        
        # Regression loss (only for samples with labels)
        if reg_mask is not None and reg_mask.sum() > 0:
            reg_output_masked = reg_output[reg_mask.bool()]
            reg_target_masked = reg_target[reg_mask.bool()]
            reg_loss = self.mse_loss(reg_output_masked, reg_target_masked)
        else:
            reg_loss = self.mse_loss(reg_output, reg_target)
        
        # Combine losses
        if self.use_uncertainty_weighting:
            # Uncertainty weighting
            precision_cls = torch.exp(-self.log_var_cls)
            precision_reg = torch.exp(-self.log_var_reg)
            
            total_loss = (
                precision_cls * cls_loss + self.log_var_cls +
                precision_reg * reg_loss + self.log_var_reg
            )
        else:
            total_loss = (
                self.classification_weight * cls_loss +
                self.regression_weight * reg_loss
            )
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'regression': reg_loss
        }


# Convenience function for creating model with common configurations
def create_cross_attention_model(
    protein_model: str = 'esm2_t36_3B_UR50D',
    ligand_model: str = 'SMI-TED',
    size: str = 'base',
    encoder_type: Literal['cnn', 'linear'] = 'cnn',
) -> CrossAttentionAffinityModel:
    """
    Factory function for creating CrossAttentionAffinityModel.
    
    Args:
        protein_model: Protein embedding model name
        ligand_model: Ligand embedding model name
        size: Model size ('small', 'base', 'large')
        encoder_type: Token encoder type ('cnn' or 'linear')
    
    Returns:
        Configured CrossAttentionAffinityModel
    """
    # Protein dimensions based on model
    protein_dims = {
        'esm2_t6_8M_UR50D': 320,
        'esm2_t12_35M_UR50D': 480,
        'esm2_t30_150M_UR50D': 640,
        'esm2_t33_650M_UR50D': 1280,
        'esm2_t36_3B_UR50D': 2560,
        'esm2_t48_15B_UR50D': 5120,
        'boltz2': 384,
        'esmc_300m': 960,
        'esmc_600m': 1152,
    }
    
    # Ligand dimensions
    ligand_dims = {
        'SMI-TED': 768,
        'SMILES-TED': 768,
    }
    
    # Model size configurations
    size_configs = {
        'small': {
            'hidden_dim': 128,
            'num_cnn_layers': 2,
            'num_cross_attn_layers': 1,
            'num_heads': 4,
            'ff_dim': 512,
            'dropout': 0.2
        },
        'base': {
            'hidden_dim': 256,
            'num_cnn_layers': 3,
            'num_cross_attn_layers': 2,
            'num_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1
        },
        'large': {
            'hidden_dim': 512,
            'num_cnn_layers': 4,
            'num_cross_attn_layers': 4,
            'num_heads': 16,
            'ff_dim': 2048,
            'dropout': 0.1
        }
    }
    
    protein_dim = protein_dims.get(protein_model, 2560)
    ligand_dim = ligand_dims.get(ligand_model, 768)
    config = size_configs.get(size, size_configs['base'])
    
    return CrossAttentionAffinityModel(
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        encoder_type=encoder_type,
        **config
    )


class CrossAttentionLiteAffinityModel(CrossAttentionAffinityModel):
    """
    Lightweight wrapper: linear token encoders + bidirectional cross-attention.
    """

    def __init__(
        self,
        protein_dim: int = 2560,
        ligand_dim: int = 768,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 1,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_protein_len: int = 2048,
        max_ligand_len: int = 512,
        use_positional_encoding: bool = True,
        positional_encoding_type: Literal['sinusoidal', 'rope'] = 'sinusoidal',
    ):
        super().__init__(
            protein_dim=protein_dim,
            ligand_dim=ligand_dim,
            hidden_dim=hidden_dim,
            encoder_type='linear',
            num_cnn_layers=1,  # Unused in linear mode; kept for compatibility.
            num_cross_attn_layers=num_cross_attn_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_protein_len=max_protein_len,
            max_ligand_len=max_ligand_len,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
        )


def create_cross_attention_lite_model(
    protein_model: str = 'esm2_t36_3B_UR50D',
    ligand_model: str = 'SMI-TED',
    size: str = 'base',
) -> CrossAttentionAffinityModel:
    """Factory helper for lite model presets."""
    return create_cross_attention_model(
        protein_model=protein_model,
        ligand_model=ligand_model,
        size=size,
        encoder_type='linear',
    )


if __name__ == "__main__":
    # Quick test
    print("Testing CrossAttentionAffinityModel...")
    
    model = create_cross_attention_model(size='small')
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Architecture info: {model.get_architecture_info()}")
    
    # Test forward pass
    batch_size = 4
    protein_len = 100
    ligand_len = 50
    
    protein_matrix = torch.randn(batch_size, protein_len, 2560)
    ligand_matrix = torch.randn(batch_size, ligand_len, 768)
    
    output = model(protein_matrix, ligand_matrix)
    
    print(f"\nOutput shapes:")
    print(f"  Classification: {output['classification'].shape}")
    print(f"  Regression: {output['regression'].shape}")
    
    # Test loss
    loss_fn = MultiTaskLoss()
    cls_target = torch.randint(0, 2, (batch_size, 1)).float()
    reg_target = torch.randn(batch_size, 1) * 2 + 6  # pIC50-like values
    
    losses = loss_fn(
        output['classification'],
        output['regression'],
        cls_target,
        reg_target
    )
    
    print(f"\nLosses:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Classification: {losses['classification'].item():.4f}")
    print(f"  Regression: {losses['regression'].item():.4f}")
    
    print("\n✅ All tests passed!")
