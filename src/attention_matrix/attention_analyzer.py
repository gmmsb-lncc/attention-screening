"""
Attention Analyzer for Cross-Attention weights visualization and interpretation.

Single Responsibility: Extract and analyze attention patterns.
Helps identify important protein-ligand interactions.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json


logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyzer for extracting and interpreting Cross-Attention weights.
    
    Extracts attention maps from trained models to identify:
    - Important protein residues for binding
    - Key ligand features contributing to activity
    - Protein-ligand interaction patterns
    
    Args:
        model: Trained CrossAttentionModel
        device: Device for inference
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Storage for attention weights
        self.attention_weights: List[Dict[str, np.ndarray]] = []
        self._hooks = []
        
        # Register hooks
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention_hook(name):
            def hook(module, input, output):
                # MultiheadAttention returns (output, attention_weights)
                if isinstance(output, tuple) and len(output) == 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.attention_weights.append({
                            'name': name,
                            'weights': attn_weights.detach().cpu().numpy()
                        })
            return hook
        
        # Find attention layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(get_attention_hook(name))
                self._hooks.append(hook)
                logger.debug(f"Registered hook for: {name}")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def extract_attention(
        self,
        protein_embedding: torch.Tensor,
        ligand_embedding: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights for a single protein-ligand pair.
        
        Args:
            protein_embedding: (1, seq_len, protein_dim)
            ligand_embedding: (1, n_tokens, ligand_dim)
            
        Returns:
            Dictionary with attention weights for each layer
        """
        self.attention_weights = []
        
        with torch.no_grad():
            protein_emb = protein_embedding.to(self.device)
            ligand_emb = ligand_embedding.to(self.device)
            
            # Forward pass (hooks capture attention)
            outputs = self.model(protein_emb, ligand_emb)
        
        # Organize results
        result = {}
        for i, attn_data in enumerate(self.attention_weights):
            result[f"layer_{i}_{attn_data['name']}"] = attn_data['weights']
        
        return result
    
    def get_residue_importance(
        self,
        attention_map: np.ndarray,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Compute importance score for each protein residue.
        
        Args:
            attention_map: (n_heads, seq_len, n_tokens) or (seq_len, n_tokens)
            aggregation: 'mean', 'max', or 'sum'
            
        Returns:
            (seq_len,) importance scores for each residue
        """
        # Handle multiple heads
        if attention_map.ndim == 3:
            # Aggregate over heads first
            attention_map = attention_map.mean(axis=0)
        
        # Aggregate over ligand tokens
        if aggregation == 'mean':
            scores = attention_map.mean(axis=-1)
        elif aggregation == 'max':
            scores = attention_map.max(axis=-1)
        else:  # sum
            scores = attention_map.sum(axis=-1)
        
        return scores
    
    def get_ligand_importance(
        self,
        attention_map: np.ndarray,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Compute importance score for each ligand token.
        
        Args:
            attention_map: (n_heads, seq_len, n_tokens) or (seq_len, n_tokens)
            aggregation: 'mean', 'max', or 'sum'
            
        Returns:
            (n_tokens,) importance scores for each ligand token
        """
        # Handle multiple heads
        if attention_map.ndim == 3:
            attention_map = attention_map.mean(axis=0)
        
        # Aggregate over protein residues
        if aggregation == 'mean':
            scores = attention_map.mean(axis=0)
        elif aggregation == 'max':
            scores = attention_map.max(axis=0)
        else:  # sum
            scores = attention_map.sum(axis=0)
        
        return scores
    
    def get_top_interactions(
        self,
        attention_map: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        Get top-k protein-ligand interactions by attention weight.
        
        Args:
            attention_map: (seq_len, n_tokens) or (n_heads, seq_len, n_tokens)
            top_k: Number of top interactions to return
            
        Returns:
            List of (residue_idx, ligand_idx, attention_weight)
        """
        if attention_map.ndim == 3:
            attention_map = attention_map.mean(axis=0)
        
        # Flatten and get top indices
        flat_idx = np.argsort(attention_map.flatten())[-top_k:][::-1]
        
        interactions = []
        for idx in flat_idx:
            res_idx = idx // attention_map.shape[1]
            lig_idx = idx % attention_map.shape[1]
            weight = attention_map[res_idx, lig_idx]
            interactions.append((int(res_idx), int(lig_idx), float(weight)))
        
        return interactions
    
    def analyze_batch(
        self,
        dataloader,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns across a batch of samples.
        
        Args:
            dataloader: DataLoader with samples
            max_samples: Limit number of samples to analyze
            
        Returns:
            Aggregated analysis results
        """
        all_residue_scores = []
        all_ligand_scores = []
        all_interactions = []
        
        n_samples = 0
        
        for batch in dataloader:
            if max_samples and n_samples >= max_samples:
                break
            
            protein_emb = batch['protein_embedding']
            ligand_emb = batch['ligand_embedding']
            
            batch_size = protein_emb.size(0)
            
            for i in range(batch_size):
                if max_samples and n_samples >= max_samples:
                    break
                
                # Extract attention
                attn = self.extract_attention(
                    protein_emb[i:i+1],
                    ligand_emb[i:i+1]
                )
                
                # Get first attention layer
                layer_name = list(attn.keys())[0] if attn else None
                if layer_name:
                    attn_map = attn[layer_name][0]  # Remove batch dim
                    
                    # Compute scores
                    res_scores = self.get_residue_importance(attn_map)
                    lig_scores = self.get_ligand_importance(attn_map)
                    interactions = self.get_top_interactions(attn_map)
                    
                    all_residue_scores.append(res_scores)
                    all_ligand_scores.append(lig_scores)
                    all_interactions.extend(interactions)
                
                n_samples += 1
        
        # Aggregate results
        return {
            'n_samples': n_samples,
            'mean_residue_importance': {
                i: float(np.mean([s[i] for s in all_residue_scores if len(s) > i]))
                for i in range(max(len(s) for s in all_residue_scores) if all_residue_scores else 0)
            },
            'top_interactions': sorted(
                all_interactions,
                key=lambda x: x[2],
                reverse=True
            )[:50]
        }
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: str
    ):
        """Save analysis results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Analysis saved to: {output_path}")


def create_attention_heatmap_data(
    attention_map: np.ndarray,
    residue_labels: Optional[List[str]] = None,
    ligand_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepare attention map data for visualization.
    
    Args:
        attention_map: (seq_len, n_tokens) attention weights
        residue_labels: Optional labels for protein residues
        ligand_labels: Optional labels for ligand tokens
        
    Returns:
        Dictionary with heatmap data for visualization
    """
    seq_len, n_tokens = attention_map.shape
    
    if residue_labels is None:
        residue_labels = [f"R{i}" for i in range(seq_len)]
    if ligand_labels is None:
        ligand_labels = [f"L{i}" for i in range(n_tokens)]
    
    return {
        'attention_matrix': attention_map.tolist(),
        'residue_labels': residue_labels,
        'ligand_labels': ligand_labels,
        'shape': [seq_len, n_tokens]
    }
