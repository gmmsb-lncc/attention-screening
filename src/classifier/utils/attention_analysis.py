"""
Attention Analysis Module for Protein-Ligand Interactions.

This module extracts and analyzes cross-attention weights from the
CNN + Cross-Attention model to understand which protein residues
are influenced by which ligand atoms.

The attention maps provide interpretable insights into:
1. Which protein residues interact most strongly with the ligand
2. Which ligand atoms/groups have the most influence on protein binding
3. The overall interaction pattern between the pair

Author: DockTKinase Team
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """
    Container for attention analysis results for a single protein-ligand pair.
    
    Attributes:
        protein_id: Identifier for the protein
        ligand_id: Identifier for the ligand
        protein_sequence: Full protein sequence (if available)
        ligand_smiles: SMILES representation of ligand (if available)
        protein_length: Length of protein sequence (after embedding)
        ligand_length: Length of ligand tokens (after embedding)
        
        protein_to_ligand_attention: [protein_len, ligand_len] - how each protein 
            position attends to ligand positions
        ligand_to_protein_attention: [ligand_len, protein_len] - how each ligand
            position attends to protein positions
        
        top_protein_residues: List of (residue_idx, attention_score) for most 
            attended protein positions
        top_ligand_atoms: List of (atom_idx, attention_score) for most 
            attended ligand positions
        
        interaction_hotspots: Pairs of (protein_idx, ligand_idx, score) with 
            highest mutual attention
    """
    protein_id: str
    ligand_id: str
    protein_sequence: Optional[str] = None
    ligand_smiles: Optional[str] = None
    protein_length: int = 0
    ligand_length: int = 0
    
    # Attention matrices (stored as lists for JSON serialization)
    protein_to_ligand_attention: Optional[List[List[float]]] = None
    ligand_to_protein_attention: Optional[List[List[float]]] = None
    
    # Aggregated attention scores
    protein_residue_importance: Optional[List[float]] = None  # Sum over ligand
    ligand_atom_importance: Optional[List[float]] = None      # Sum over protein
    
    # Top interactions
    top_protein_residues: Optional[List[Tuple[int, float]]] = None
    top_ligand_atoms: Optional[List[Tuple[int, float]]] = None
    interaction_hotspots: Optional[List[Tuple[int, int, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttentionResult':
        """Create from dictionary."""
        return cls(**data)


class AttentionExtractor(nn.Module):
    """
    Modified cross-attention that returns attention weights.
    
    This is a wrapper/replacement for the CrossAttention class that
    also returns the attention weight matrix for analysis.
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
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weight return.
        
        Args:
            query: [batch, query_len, hidden_dim]
            key: [batch, key_len, hidden_dim]
            value: [batch, key_len, hidden_dim]
            key_mask: [batch, key_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: [batch, query_len, hidden_dim]
            attention_weights: [batch, num_heads, query_len, key_len] if return_attention
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        output = self.out_proj(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


class ProteinLigandAttentionAnalyzer:
    """
    Analyzer for extracting and interpreting attention patterns from
    protein-ligand interaction models.
    
    This class provides methods to:
    1. Extract attention weights from a trained model
    2. Aggregate multi-head attention into interpretable scores
    3. Identify interaction hotspots
    4. Generate reports mapping attention to sequence positions
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        aggregation: str = 'mean'
    ):
        """
        Args:
            model: Trained CrossAttentionAffinityModel
            device: Computation device
            aggregation: How to aggregate multi-head attention ('mean', 'max', 'sum')
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.aggregation = aggregation
        self.model.to(self.device)
        self.model.eval()
        
        # Store hooks for attention extraction
        self._attention_hooks = []
        self._captured_attentions = {}
    
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        self._captured_attentions = {}
        
        def make_hook(name):
            def hook(module, input, output):
                # Store attention weights if they're returned
                if isinstance(output, tuple) and len(output) == 2:
                    self._captured_attentions[name] = output[1]
            return hook
        
        # Find cross-attention layers
        for name, module in self.model.named_modules():
            if 'cross_attn' in name.lower():
                handle = module.register_forward_hook(make_hook(name))
                self._attention_hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks = []
    
    @torch.no_grad()
    def extract_attention_weights(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model for a protein-ligand pair.
        
        Args:
            protein_matrix: [1, protein_len, protein_dim] or [protein_len, protein_dim]
            ligand_matrix: [1, ligand_len, ligand_dim] or [ligand_len, ligand_dim]
            protein_mask: Optional padding mask
            ligand_mask: Optional padding mask
            
        Returns:
            Dictionary with attention weight tensors
        """
        # Ensure batch dimension
        if protein_matrix.dim() == 2:
            protein_matrix = protein_matrix.unsqueeze(0)
        if ligand_matrix.dim() == 2:
            ligand_matrix = ligand_matrix.unsqueeze(0)
        
        protein_matrix = protein_matrix.to(self.device)
        ligand_matrix = ligand_matrix.to(self.device)
        
        if protein_mask is not None:
            protein_mask = protein_mask.to(self.device)
        if ligand_mask is not None:
            ligand_mask = ligand_mask.to(self.device)
        
        # We need to modify the forward pass to capture attention
        # This is a workaround since the model doesn't return attention by default
        attention_maps = self._extract_via_manual_forward(
            protein_matrix, ligand_matrix, protein_mask, ligand_mask
        )
        
        return attention_maps
    
    def _extract_via_manual_forward(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor],
        ligand_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Manually extract attention by partially replicating forward pass.
        
        This method accesses model internals to extract attention weights
        that aren't normally returned.
        """
        attention_maps = {}
        
        try:
            # Access CNN encoders
            protein_encoder = self.model.protein_encoder
            ligand_encoder = self.model.ligand_encoder
            
            # Encode through CNNs first (they do input projection)
            protein_encoded = protein_encoder(protein_matrix, protein_mask)
            ligand_encoded = ligand_encoder(ligand_matrix, ligand_mask)
            
            # Apply positional encoding AFTER CNN encoding (when dimensions match)
            if hasattr(self.model, 'protein_pos_enc'):
                protein_encoded = self.model.protein_pos_enc(protein_encoded)
                
            if hasattr(self.model, 'ligand_pos_enc'):
                ligand_encoded = self.model.ligand_pos_enc(ligand_encoded)
            
            # Access cross-attention blocks - try both naming conventions
            if hasattr(self.model, 'cross_attn_blocks'):
                cross_attn_blocks = self.model.cross_attn_blocks
            elif hasattr(self.model, 'cross_attention_blocks'):
                cross_attn_blocks = self.model.cross_attention_blocks
            else:
                raise AttributeError("Model has no cross-attention blocks")
            
            for layer_idx, block in enumerate(cross_attn_blocks):
                # Extract attention from protein attending to ligand
                p2l_attn = self._compute_attention_weights(
                    block.protein_cross_attn,
                    protein_encoded,  # query
                    ligand_encoded,   # key/value
                    ligand_mask
                )
                attention_maps[f'layer_{layer_idx}_protein_to_ligand'] = p2l_attn
                
                # Extract attention from ligand attending to protein
                l2p_attn = self._compute_attention_weights(
                    block.ligand_cross_attn,
                    ligand_encoded,   # query
                    protein_encoded,  # key/value
                    protein_mask
                )
                attention_maps[f'layer_{layer_idx}_ligand_to_protein'] = l2p_attn
                
                # Update representations for next layer
                # Try different naming conventions for layer norm
                protein_cross_out = block.protein_cross_attn(
                    protein_encoded, ligand_encoded, ligand_encoded, ligand_mask
                )
                if hasattr(block, 'protein_norm1'):
                    protein_encoded = block.protein_norm1(protein_encoded + protein_cross_out)
                elif hasattr(block, 'protein_ln1'):
                    protein_encoded = block.protein_ln1(protein_encoded + protein_cross_out)
                else:
                    protein_encoded = protein_encoded + protein_cross_out
                
                ligand_cross_out = block.ligand_cross_attn(
                    ligand_encoded, protein_encoded, protein_encoded, protein_mask
                )
                if hasattr(block, 'ligand_norm1'):
                    ligand_encoded = block.ligand_norm1(ligand_encoded + ligand_cross_out)
                elif hasattr(block, 'ligand_ln1'):
                    ligand_encoded = block.ligand_ln1(ligand_encoded + ligand_cross_out)
                else:
                    ligand_encoded = ligand_encoded + ligand_cross_out
                
        except AttributeError as e:
            logger.warning(f"Could not access model internals: {e}")
            logger.info("Using alternative attention extraction method")
            attention_maps = self._extract_via_hooks(
                protein_matrix, ligand_matrix, protein_mask, ligand_mask
            )
        
        return attention_maps
    
    def _compute_attention_weights(
        self,
        cross_attn_module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        key_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention weights from a cross-attention module.
        
        Args:
            cross_attn_module: CrossAttention module
            query: [batch, query_len, hidden_dim]
            key: [batch, key_len, hidden_dim]
            key_mask: [batch, key_len]
            
        Returns:
            attention_weights: [batch, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        hidden_dim = cross_attn_module.hidden_dim
        num_heads = cross_attn_module.num_heads
        head_dim = cross_attn_module.head_dim
        scale = cross_attn_module.scale
        
        Q = cross_attn_module.q_proj(query)
        K = cross_attn_module.k_proj(key)
        
        Q = Q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        return attn_weights
    
    def _extract_via_hooks(
        self,
        protein_matrix: torch.Tensor,
        ligand_matrix: torch.Tensor,
        protein_mask: Optional[torch.Tensor],
        ligand_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Alternative extraction using forward hooks."""
        self._register_attention_hooks()
        
        try:
            _ = self.model(protein_matrix, ligand_matrix, protein_mask, ligand_mask)
        finally:
            self._remove_hooks()
        
        return self._captured_attentions.copy()
    
    def aggregate_attention(
        self,
        attention_weights: torch.Tensor,
        method: str = None
    ) -> torch.Tensor:
        """
        Aggregate multi-head attention into a single attention map.
        
        Args:
            attention_weights: [batch, num_heads, query_len, key_len]
            method: Aggregation method ('mean', 'max', 'sum')
            
        Returns:
            aggregated: [batch, query_len, key_len]
        """
        method = method or self.aggregation
        
        if method == 'mean':
            return attention_weights.mean(dim=1)
        elif method == 'max':
            return attention_weights.max(dim=1)[0]
        elif method == 'sum':
            return attention_weights.sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def analyze_pair(
        self,
        protein_matrix: Union[torch.Tensor, np.ndarray, str, Path],
        ligand_matrix: Union[torch.Tensor, np.ndarray, str, Path],
        protein_id: str = 'unknown',
        ligand_id: str = 'unknown',
        protein_sequence: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        top_k: int = 10
    ) -> AttentionResult:
        """
        Analyze a single protein-ligand pair and return detailed attention results.
        
        Args:
            protein_matrix: Protein embedding matrix [seq_len, embed_dim] or path to .npy
            ligand_matrix: Ligand embedding matrix [seq_len, embed_dim] or path to .npy
            protein_id: Identifier for the protein
            ligand_id: Identifier for the ligand
            protein_sequence: Optional protein sequence string
            ligand_smiles: Optional SMILES string
            top_k: Number of top interactions to report
            
        Returns:
            AttentionResult with detailed analysis
        """
        # Load matrices if paths provided
        if isinstance(protein_matrix, (str, Path)):
            protein_matrix = np.load(protein_matrix)
        if isinstance(ligand_matrix, (str, Path)):
            ligand_matrix = np.load(ligand_matrix)
        
        # Convert to tensors
        if isinstance(protein_matrix, np.ndarray):
            protein_matrix = torch.from_numpy(protein_matrix).float()
        if isinstance(ligand_matrix, np.ndarray):
            ligand_matrix = torch.from_numpy(ligand_matrix).float()
        
        protein_len = protein_matrix.size(0) if protein_matrix.dim() == 2 else protein_matrix.size(1)
        ligand_len = ligand_matrix.size(0) if ligand_matrix.dim() == 2 else ligand_matrix.size(1)
        
        # Extract attention weights
        attention_maps = self.extract_attention_weights(protein_matrix, ligand_matrix)
        
        if not attention_maps:
            logger.warning("No attention maps extracted")
            return AttentionResult(
                protein_id=protein_id,
                ligand_id=ligand_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles
            )
        
        # Get last layer attention (typically most interpretable)
        # Filter for keys that match our naming pattern
        p2l_keys = [k for k in attention_maps.keys() if 'protein_to_ligand' in k]
        l2p_keys = [k for k in attention_maps.keys() if 'ligand_to_protein' in k]
        
        if not p2l_keys or not l2p_keys:
            logger.warning("Could not find attention maps with expected naming")
            return AttentionResult(
                protein_id=protein_id,
                ligand_id=ligand_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles
            )
        
        # Get the last layer (highest index)
        def get_layer_idx(key):
            parts = key.split('_')
            for i, p in enumerate(parts):
                if p == 'layer' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
            return 0
        
        last_p2l_key = max(p2l_keys, key=get_layer_idx)
        last_l2p_key = max(l2p_keys, key=get_layer_idx)
        
        p2l_attn = attention_maps.get(last_p2l_key)
        l2p_attn = attention_maps.get(last_l2p_key)
        
        if p2l_attn is None or l2p_attn is None:
            logger.warning("Could not extract attention maps")
            return AttentionResult(
                protein_id=protein_id,
                ligand_id=ligand_id,
                protein_sequence=protein_sequence,
                ligand_smiles=ligand_smiles
            )
        
        # Aggregate over heads
        p2l_aggregated = self.aggregate_attention(p2l_attn).squeeze(0)  # [protein_len, ligand_len]
        l2p_aggregated = self.aggregate_attention(l2p_attn).squeeze(0)  # [ligand_len, protein_len]
        
        # Convert to numpy
        p2l_np = p2l_aggregated.cpu().numpy()
        l2p_np = l2p_aggregated.cpu().numpy()
        
        # Calculate importance scores
        protein_importance = p2l_np.sum(axis=1)  # Sum attention over ligand positions
        ligand_importance = l2p_np.sum(axis=1)   # Sum attention over protein positions
        
        # Get top protein residues
        top_protein_indices = np.argsort(protein_importance)[-top_k:][::-1]
        top_protein_residues = [
            (int(idx), float(protein_importance[idx])) 
            for idx in top_protein_indices
        ]
        
        # Get top ligand atoms
        top_ligand_indices = np.argsort(ligand_importance)[-top_k:][::-1]
        top_ligand_atoms = [
            (int(idx), float(ligand_importance[idx])) 
            for idx in top_ligand_indices
        ]
        
        # Find interaction hotspots (highest attention pairs)
        # Combine both attention directions
        combined_attn = (p2l_np + l2p_np.T) / 2
        flat_indices = np.argsort(combined_attn.flatten())[-top_k:][::-1]
        interaction_hotspots = []
        for flat_idx in flat_indices:
            p_idx = flat_idx // ligand_len
            l_idx = flat_idx % ligand_len
            score = float(combined_attn[p_idx, l_idx])
            interaction_hotspots.append((int(p_idx), int(l_idx), score))
        
        return AttentionResult(
            protein_id=protein_id,
            ligand_id=ligand_id,
            protein_sequence=protein_sequence,
            ligand_smiles=ligand_smiles,
            protein_length=int(protein_len),
            ligand_length=int(ligand_len),
            protein_to_ligand_attention=p2l_np.tolist(),
            ligand_to_protein_attention=l2p_np.tolist(),
            protein_residue_importance=protein_importance.tolist(),
            ligand_atom_importance=ligand_importance.tolist(),
            top_protein_residues=top_protein_residues,
            top_ligand_atoms=top_ligand_atoms,
            interaction_hotspots=interaction_hotspots
        )
    
    def analyze_batch(
        self,
        data_df: pd.DataFrame,
        protein_matrix_dir: Union[str, Path],
        ligand_matrix_dir: Union[str, Path],
        protein_id_column: str = 'seq_id',
        ligand_id_column: str = 'chembl_id',
        sequence_column: Optional[str] = 'seq',
        smiles_column: Optional[str] = 'canonical_smiles',
        top_k: int = 10
    ) -> List[AttentionResult]:
        """
        Analyze multiple protein-ligand pairs from a dataframe.
        
        Args:
            data_df: DataFrame with protein and ligand identifiers
            protein_matrix_dir: Directory with protein embedding matrices
            ligand_matrix_dir: Directory with ligand embedding matrices
            protein_id_column: Column name for protein IDs
            ligand_id_column: Column name for ligand IDs
            sequence_column: Optional column with protein sequences
            smiles_column: Optional column with SMILES strings
            top_k: Number of top interactions to report per pair
            
        Returns:
            List of AttentionResult for each pair
        """
        protein_matrix_dir = Path(protein_matrix_dir)
        ligand_matrix_dir = Path(ligand_matrix_dir)
        
        results = []
        
        for idx, row in data_df.iterrows():
            protein_id = str(row[protein_id_column])
            ligand_id = str(row[ligand_id_column])
            
            protein_file = protein_matrix_dir / f'{protein_id}_matrix.npy'
            ligand_file = ligand_matrix_dir / f'{ligand_id}_matrix.npy'
            
            if not protein_file.exists() or not ligand_file.exists():
                logger.warning(f"Missing matrix files for {protein_id}/{ligand_id}")
                continue
            
            protein_seq = row.get(sequence_column) if sequence_column else None
            ligand_smiles = row.get(smiles_column) if smiles_column else None
            
            result = self.analyze_pair(
                protein_matrix=protein_file,
                ligand_matrix=ligand_file,
                protein_id=protein_id,
                ligand_id=ligand_id,
                protein_sequence=str(protein_seq) if protein_seq else None,
                ligand_smiles=str(ligand_smiles) if ligand_smiles else None,
                top_k=top_k
            )
            results.append(result)
        
        return results


def save_attention_analysis(
    results: List[AttentionResult],
    output_dir: Union[str, Path],
    save_matrices: bool = True
) -> None:
    """
    Save attention analysis results to files.
    
    Args:
        results: List of AttentionResult objects
        output_dir: Directory to save results
        save_matrices: Whether to save full attention matrices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary = []
    for r in results:
        entry = {
            'protein_id': r.protein_id,
            'ligand_id': r.ligand_id,
            'protein_length': r.protein_length,
            'ligand_length': r.ligand_length,
            'top_protein_residues': r.top_protein_residues,
            'top_ligand_atoms': r.top_ligand_atoms,
            'interaction_hotspots': r.interaction_hotspots
        }
        if r.protein_sequence:
            entry['protein_sequence'] = r.protein_sequence
        if r.ligand_smiles:
            entry['ligand_smiles'] = r.ligand_smiles
        summary.append(entry)
    
    with open(output_dir / 'attention_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results for each pair
    if save_matrices:
        matrices_dir = output_dir / 'attention_matrices'
        matrices_dir.mkdir(exist_ok=True)
        
        for r in results:
            pair_id = f"{r.protein_id}_{r.ligand_id}"
            
            if r.protein_to_ligand_attention:
                np.save(
                    matrices_dir / f'{pair_id}_protein_to_ligand.npy',
                    np.array(r.protein_to_ligand_attention)
                )
            
            if r.ligand_to_protein_attention:
                np.save(
                    matrices_dir / f'{pair_id}_ligand_to_protein.npy',
                    np.array(r.ligand_to_protein_attention)
                )
            
            # Save per-pair JSON
            with open(matrices_dir / f'{pair_id}_analysis.json', 'w') as f:
                json.dump(r.to_dict(), f, indent=2)


def generate_attention_report(
    results: List[AttentionResult],
    output_file: Union[str, Path]
) -> str:
    """
    Generate a human-readable report of attention analysis.
    
    Args:
        results: List of AttentionResult objects
        output_file: Path to save the report
        
    Returns:
        Report as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PROTEIN-LIGAND ATTENTION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total pairs analyzed: {len(results)}")
    lines.append("")
    
    for i, r in enumerate(results, 1):
        lines.append("-" * 80)
        lines.append(f"PAIR {i}: {r.protein_id} + {r.ligand_id}")
        lines.append("-" * 80)
        lines.append(f"  Protein length: {r.protein_length} residues")
        lines.append(f"  Ligand length:  {r.ligand_length} tokens")
        
        if r.ligand_smiles:
            smiles_display = r.ligand_smiles[:50] + "..." if len(r.ligand_smiles) > 50 else r.ligand_smiles
            lines.append(f"  SMILES: {smiles_display}")
        
        lines.append("")
        lines.append("  TOP PROTEIN RESIDUES (by attention to ligand):")
        if r.top_protein_residues:
            for rank, (idx, score) in enumerate(r.top_protein_residues[:5], 1):
                residue = ""
                if r.protein_sequence and idx < len(r.protein_sequence):
                    residue = f" ({r.protein_sequence[idx]})"
                lines.append(f"    {rank}. Position {idx}{residue}: attention = {score:.4f}")
        
        lines.append("")
        lines.append("  TOP LIGAND POSITIONS (by attention to protein):")
        if r.top_ligand_atoms:
            for rank, (idx, score) in enumerate(r.top_ligand_atoms[:5], 1):
                lines.append(f"    {rank}. Position {idx}: attention = {score:.4f}")
        
        lines.append("")
        lines.append("  INTERACTION HOTSPOTS (protein_pos, ligand_pos, score):")
        if r.interaction_hotspots:
            for rank, (p_idx, l_idx, score) in enumerate(r.interaction_hotspots[:5], 1):
                residue = ""
                if r.protein_sequence and p_idx < len(r.protein_sequence):
                    residue = f" ({r.protein_sequence[p_idx]})"
                lines.append(f"    {rank}. Protein[{p_idx}]{residue} <-> Ligand[{l_idx}]: {score:.4f}")
        
        lines.append("")
    
    report = "\n".join(lines)
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)
    
    return report


# Visualization functions (optional - requires matplotlib)
def plot_attention_heatmap(
    result: AttentionResult,
    output_file: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis'
) -> None:
    """
    Plot attention heatmap for a protein-ligand pair.
    
    Args:
        result: AttentionResult object
        output_file: Path to save the figure
        figsize: Figure size
        cmap: Colormap name
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return
    
    if result.protein_to_ligand_attention is None:
        logger.warning("No attention matrix available for plotting")
        return
    
    attn_matrix = np.array(result.protein_to_ligand_attention)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Protein -> Ligand attention
    ax1 = axes[0]
    im1 = ax1.imshow(attn_matrix, aspect='auto', cmap=cmap)
    ax1.set_xlabel('Ligand Position')
    ax1.set_ylabel('Protein Position')
    ax1.set_title(f'Protein → Ligand Attention\n{result.protein_id} + {result.ligand_id}')
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # Aggregated importance
    ax2 = axes[1]
    protein_imp = np.array(result.protein_residue_importance) if result.protein_residue_importance else attn_matrix.sum(axis=1)
    ax2.barh(range(len(protein_imp)), protein_imp, color='steelblue')
    ax2.set_xlabel('Total Attention')
    ax2.set_ylabel('Protein Position')
    ax2.set_title('Residue Importance')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Attention Analysis Module loaded successfully")
    print("\nUsage:")
    print("  from src.classifier.utils.attention_analysis import ProteinLigandAttentionAnalyzer")
    print("  ")
    print("  analyzer = ProteinLigandAttentionAnalyzer(model, device)")
    print("  result = analyzer.analyze_pair(protein_matrix, ligand_matrix, 'P12345', 'CHEMBL123')")
    print("  ")
    print("  # Access results:")
    print("  print(result.top_protein_residues)  # Most important residues")
    print("  print(result.interaction_hotspots)   # Key interaction sites")
