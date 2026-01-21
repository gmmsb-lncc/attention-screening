"""Flexible Cross-Attention Model with configurable encoders."""

from typing import Optional
import torch
import torch.nn as nn
from src.classifier.models.encoder_variants import create_encoder


class FlexibleCrossAttentionModel(nn.Module):
    """
    Cross-attention model with configurable encoder type.

    Supports three encoder types:
    - 'linear': Simple linear projection
    - 'cnn': Convolutional encoder
    - 'cnn_attention': CNN + self-attention hybrid
    """

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        hidden_dim: int = 256,
        encoder_type: str = 'cnn',
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize the model.

        Args:
            protein_dim: Protein embedding dimension
            ligand_dim: Ligand embedding dimension
            hidden_dim: Hidden dimension for all layers
            encoder_type: Type of encoder ('linear', 'cnn', 'cnn_attention')
            num_cross_attn_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder_type = encoder_type

        # Create encoders based on type
        self.protein_encoder = create_encoder(
            encoder_type, protein_dim, hidden_dim, dropout=dropout,
            num_layers=3, num_cnn_layers=2, num_attention_layers=1
        )
        self.ligand_encoder = create_encoder(
            encoder_type, ligand_dim, hidden_dim, dropout=dropout,
            num_layers=3, num_cnn_layers=2, num_attention_layers=1
        )

        # Cross-attention layers
        self.cross_attention_layers = self._build_cross_attention_layers(
            num_cross_attn_layers, hidden_dim, num_heads, ff_dim, dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def _build_cross_attention_layers(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float
    ) -> nn.ModuleList:
        """Build cross-attention layers."""
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(nn.ModuleDict({
                'protein_to_ligand': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'ligand_to_protein': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'protein_norm': nn.LayerNorm(hidden_dim),
                'ligand_norm': nn.LayerNorm(hidden_dim),
                'protein_ff': self._build_feedforward(hidden_dim, ff_dim, dropout),
                'ligand_ff': self._build_feedforward(hidden_dim, ff_dim, dropout),
                'protein_ff_norm': nn.LayerNorm(hidden_dim),
                'ligand_ff_norm': nn.LayerNorm(hidden_dim)
            }))
        return layers

    def _build_feedforward(self, hidden_dim: int, ff_dim: int, dropout: float) -> nn.Sequential:
        """Build feed-forward network."""
        return nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            protein_matrix: Protein embedding matrix [batch, seq_len, protein_dim]
            ligand_matrix: Ligand embedding matrix [batch, seq_len, ligand_dim]
            protein_mask: Protein attention mask [batch, seq_len]
            ligand_mask: Ligand attention mask [batch, seq_len]

        Returns:
            Logits [batch, 1]
        """
        # Encode
        protein_encoded = self.protein_encoder(protein_matrix, protein_mask)
        ligand_encoded = self.ligand_encoder(ligand_matrix, ligand_mask)

        # Create attention masks
        protein_key_mask = (protein_mask == 0) if protein_mask is not None else None
        ligand_key_mask = (ligand_mask == 0) if ligand_mask is not None else None

        # Cross-attention
        protein_encoded, ligand_encoded = self._apply_cross_attention(
            protein_encoded, ligand_encoded, protein_key_mask, ligand_key_mask
        )

        # Pool and classify
        protein_pooled = self._pool(protein_encoded, protein_mask)
        ligand_pooled = self._pool(ligand_encoded, ligand_mask)

        combined = torch.cat([protein_pooled, ligand_pooled], dim=-1)
        return self.classifier(combined)

    def _apply_cross_attention(
        self,
        protein_encoded: torch.Tensor,
        ligand_encoded: torch.Tensor,
        protein_key_mask: Optional[torch.Tensor],
        ligand_key_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention layers."""
        for layer in self.cross_attention_layers:
            # Save original states for parallel cross-attention
            protein_input = protein_encoded
            ligand_input = ligand_encoded

            # Protein attends to ligand
            protein_encoded = self._cross_attend(
                protein_input, ligand_input, ligand_key_mask,
                layer['protein_norm'], layer['protein_to_ligand'],
                layer['protein_ff_norm'], layer['protein_ff']
            )

            # Ligand attends to protein
            ligand_encoded = self._cross_attend(
                ligand_input, protein_input, protein_key_mask,
                layer['ligand_norm'], layer['ligand_to_protein'],
                layer['ligand_ff_norm'], layer['ligand_ff']
            )

        return protein_encoded, ligand_encoded

    def _cross_attend(
        self,
        query_input: torch.Tensor,
        key_value_input: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        norm1: nn.LayerNorm,
        attention: nn.MultiheadAttention,
        norm2: nn.LayerNorm,
        feedforward: nn.Sequential
    ) -> torch.Tensor:
        """Apply single cross-attention block with residual connections."""
        # Attention with pre-norm
        residual = query_input
        query = norm1(query_input)
        attn_out, _ = attention(query, key_value_input, key_value_input,
                               key_padding_mask=key_padding_mask)
        query_output = residual + attn_out

        # Feed-forward with pre-norm
        residual = query_output
        ff_input = norm2(query_output)
        ff_out = feedforward(ff_input)
        return residual + ff_out

    def _pool(self, encoded: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Pool encoded sequence using mask-aware mean pooling."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            pooled = (encoded * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = encoded.mean(1)
        return pooled
