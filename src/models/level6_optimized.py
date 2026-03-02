"""Level 6: Optimized Transformer with Hyperparameter Search.

This module implements Phase 1 of the Level 6 architecture:
- Full Transformer encoders for protein and ligand
- Multi-head cross-attention for interaction modeling
- Automated hyperparameter optimization via Optuna
- Advanced training techniques (warmup, label smoothing, etc.)

Scientific justification:
1. Full Transformer replaces simple linear projections → better sequence modeling
2. Multi-head attention captures diverse protein-ligand interaction patterns
3. Automated HPO finds optimal architecture for each dataset
4. Label smoothing prevents overconfident predictions
5. Warmup stabilizes training with large learning rates
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between protein and ligand representations."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Protein attends to ligand
        self.prot_to_lig = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=True
        )
        self.prot_norm1 = nn.LayerNorm(d_model)
        self.prot_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.prot_norm2 = nn.LayerNorm(d_model)
        
        # Ligand attends to protein
        self.lig_to_prot = nn.MultiheadAttention(
            d_model, nhead, dropout=attention_dropout, batch_first=True
        )
        self.lig_norm1 = nn.LayerNorm(d_model)
        self.lig_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.lig_norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        prot: torch.Tensor,
        lig: torch.Tensor,
        prot_mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention.
        
        Args:
            prot: [batch, prot_len, d_model]
            lig: [batch, lig_len, d_model]
            prot_mask: [batch, prot_len] boolean mask (True = valid)
            lig_mask: [batch, lig_len] boolean mask (True = valid)
        
        Returns:
            (prot_out, lig_out): Updated representations
        """
        # Protein attends to ligand
        prot_attn, _ = self.prot_to_lig(
            prot, lig, lig,
            key_padding_mask=~lig_mask if lig_mask is not None else None,
        )
        prot = self.prot_norm1(prot + prot_attn)
        prot = self.prot_norm2(prot + self.prot_ffn(prot))
        
        # Ligand attends to protein
        lig_attn, _ = self.lig_to_prot(
            lig, prot, prot,
            key_padding_mask=~prot_mask if prot_mask is not None else None,
        )
        lig = self.lig_norm1(lig + lig_attn)
        lig = self.lig_norm2(lig + self.lig_ffn(lig))
        
        return prot, lig


class Level6OptimizedModel(nn.Module):
    """Level 6 Optimized Transformer Architecture.
    
    Architecture:
    1. Input projection: protein_dim → d_model, ligand_dim → d_model
    2. Positional encoding
    3. Transformer encoders (separate for protein and ligand)
    4. Multi-layer bidirectional cross-attention
    5. Pooling (mean + max + CLS token)
    6. Classification head with dropout
    
    Hyperparameters are optimized via Optuna based on config.
    """
    
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        d_model: int = 384,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 2,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.protein_proj = nn.Linear(protein_dim, d_model)
        self.ligand_proj = nn.Linear(ligand_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Learnable CLS tokens
        self.protein_cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.ligand_cls = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.protein_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.ligand_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # Cross-attention blocks
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                nhead=cross_attention_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(cross_attention_layers)
        ])
        
        # Classifier head
        # Pooling: CLS + mean + max = 3 * d_model per modality
        fusion_dim = 6 * d_model
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            protein: [batch, prot_len, protein_dim]
            ligand: [batch, lig_len, ligand_dim]
            protein_mask: [batch, prot_len] boolean (True = valid)
            ligand_mask: [batch, lig_len] boolean (True = valid)
        
        Returns:
            logits: [batch, 1]
        """
        batch_size = protein.size(0)
        
        # Project to d_model
        prot = self.protein_proj(protein)  # [batch, prot_len, d_model]
        lig = self.ligand_proj(ligand)      # [batch, lig_len, d_model]
        
        # Add CLS tokens
        prot_cls = self.protein_cls.expand(batch_size, -1, -1)
        lig_cls = self.ligand_cls.expand(batch_size, -1, -1)
        
        prot = torch.cat([prot_cls, prot], dim=1)  # [batch, prot_len+1, d_model]
        lig = torch.cat([lig_cls, lig], dim=1)      # [batch, lig_len+1, d_model]
        
        # Update masks
        if protein_mask is not None:
            prot_mask_with_cls = torch.cat([
                torch.ones(batch_size, 1, device=protein_mask.device, dtype=torch.bool),
                protein_mask
            ], dim=1)
        else:
            prot_mask_with_cls = None
        
        if ligand_mask is not None:
            lig_mask_with_cls = torch.cat([
                torch.ones(batch_size, 1, device=ligand_mask.device, dtype=torch.bool),
                ligand_mask
            ], dim=1)
        else:
            lig_mask_with_cls = None
        
        # Positional encoding
        prot = self.pos_encoder(prot)
        lig = self.pos_encoder(lig)
        
        # Transformer encoding
        prot = self.protein_encoder(
            prot,
            src_key_padding_mask=~prot_mask_with_cls if prot_mask_with_cls is not None else None
        )
        lig = self.ligand_encoder(
            lig,
            src_key_padding_mask=~lig_mask_with_cls if lig_mask_with_cls is not None else None
        )
        
        # Cross-attention
        for cross_attn in self.cross_attention_layers:
            prot, lig = cross_attn(prot, lig, prot_mask_with_cls, lig_mask_with_cls)
        
        # Pooling: CLS + mean + max
        prot_cls_token = prot[:, 0]  # [batch, d_model]
        lig_cls_token = lig[:, 0]    # [batch, d_model]
        
        # Mean pooling (excluding CLS and padding)
        if prot_mask_with_cls is not None:
            prot_seq = prot[:, 1:] * prot_mask_with_cls[:, 1:].unsqueeze(-1)
            prot_mean = prot_seq.sum(dim=1) / prot_mask_with_cls[:, 1:].sum(dim=1, keepdim=True)
        else:
            prot_mean = prot[:, 1:].mean(dim=1)
        
        if lig_mask_with_cls is not None:
            lig_seq = lig[:, 1:] * lig_mask_with_cls[:, 1:].unsqueeze(-1)
            lig_mean = lig_seq.sum(dim=1) / lig_mask_with_cls[:, 1:].sum(dim=1, keepdim=True)
        else:
            lig_mean = lig[:, 1:].mean(dim=1)
        
        # Max pooling (excluding CLS and padding)
        if prot_mask_with_cls is not None:
            prot_seq_masked = prot[:, 1:].clone()
            prot_seq_masked[~prot_mask_with_cls[:, 1:]] = -1e9
            prot_max = prot_seq_masked.max(dim=1)[0]
        else:
            prot_max = prot[:, 1:].max(dim=1)[0]
        
        if lig_mask_with_cls is not None:
            lig_seq_masked = lig[:, 1:].clone()
            lig_seq_masked[~lig_mask_with_cls[:, 1:]] = -1e9
            lig_max = lig_seq_masked.max(dim=1)[0]
        else:
            lig_max = lig[:, 1:].max(dim=1)[0]
        
        # Concatenate all pooled representations
        fusion = torch.cat([
            prot_cls_token, prot_mean, prot_max,
            lig_cls_token, lig_mean, lig_max
        ], dim=-1)  # [batch, 6*d_model]
        
        # Classification
        logits = self.classifier(fusion)  # [batch, 1]
        
        return logits


def load_hparam_config(config_path: str = "configs/level6_hparam_search.json") -> Dict:
    """Load hyperparameter search configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_model_from_trial(
    trial,
    protein_dim: int,
    ligand_dim: int,
    config: Optional[Dict] = None,
) -> Level6OptimizedModel:
    """Create model with hyperparameters suggested by Optuna trial.
    
    Args:
        trial: Optuna trial object
        protein_dim: Protein embedding dimension
        ligand_dim: Ligand embedding dimension
        config: Hyperparameter search config (loaded from JSON)
    
    Returns:
        Level6OptimizedModel instance
    """
    if config is None:
        config = load_hparam_config()
    
    search_space = config['search_space']
    
    # Sample hyperparameters
    d_model = trial.suggest_categorical('d_model', search_space['d_model']['choices'])
    nhead = trial.suggest_categorical('nhead', search_space['nhead']['choices'])
    
    # Ensure nhead divides d_model
    while d_model % nhead != 0:
        nhead = trial.suggest_categorical('nhead', search_space['nhead']['choices'])
    
    num_encoder_layers = trial.suggest_int(
        'num_encoder_layers',
        search_space['num_encoder_layers']['low'],
        search_space['num_encoder_layers']['high']
    )
    
    dim_feedforward = trial.suggest_categorical(
        'dim_feedforward',
        search_space['dim_feedforward']['choices']
    )
    
    dropout = trial.suggest_float(
        'dropout',
        search_space['dropout']['low'],
        search_space['dropout']['high'],
        step=search_space['dropout']['step']
    )
    
    attention_dropout = trial.suggest_float(
        'attention_dropout',
        search_space['attention_dropout']['low'],
        search_space['attention_dropout']['high'],
        step=search_space['attention_dropout']['step']
    )
    
    cross_attention_heads = trial.suggest_categorical(
        'cross_attention_heads',
        search_space['cross_attention_heads']['choices']
    )
    
    cross_attention_layers = trial.suggest_int(
        'cross_attention_layers',
        search_space['cross_attention_layers']['low'],
        search_space['cross_attention_layers']['high']
    )
    
    classifier_dropout = trial.suggest_float(
        'classifier_dropout',
        search_space['classifier_dropout']['low'],
        search_space['classifier_dropout']['high'],
        step=search_space['classifier_dropout']['step']
    )
    
    model = Level6OptimizedModel(
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        attention_dropout=attention_dropout,
        cross_attention_heads=cross_attention_heads,
        cross_attention_layers=cross_attention_layers,
        classifier_dropout=classifier_dropout,
    )
    
    return model
