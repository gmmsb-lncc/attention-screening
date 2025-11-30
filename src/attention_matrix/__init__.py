"""
Attention Matrix Module
=======================

Module for training and evaluating Cross-Attention models using 
protein and ligand matrix embeddings.

This module implements:
- CrossAttentionModel: CNN + Multi-head Cross-Attention architecture
- AttentionMatrixPipeline: Complete training and evaluation pipeline
- Data management for matrix embeddings
- Leakage-aware data splitting with protein clustering
- Attention weight analysis and interpretation

Design Principles:
- SOLID: Single Responsibility, Open/Closed, Interface Segregation
- KISS: Keep It Simple, Stupid
- Clean Code: Readable, maintainable, testable

Example:
    >>> from attention_matrix import AttentionMatrixPipeline, AttentionMatrixConfig
    >>> config = AttentionMatrixConfig(hidden_dim=256, num_heads=8)
    >>> pipeline = AttentionMatrixPipeline(config, output_dir="results/attention")
    >>> results = pipeline.run(
    ...     data_path="data/compounds.tsv",
    ...     protein_dir="embeddings/proteins",
    ...     ligand_dir="embeddings/ligands"
    ... )
"""

from .config import AttentionMatrixConfig
from .model import CrossAttentionModel, ImprovedCrossAttentionModel
from .dataset import ProteinLigandDataset, create_dataloaders
from .pipeline import AttentionMatrixPipeline
from .trainer import AttentionTrainer
from .evaluator import AttentionEvaluator
from .splitter import LeakageAwareSplitter, SimpleSplitter
from .attention_analyzer import AttentionAnalyzer, create_attention_heatmap_data
from .data_loader import EmbeddingDataLoader, load_data_for_attention_matrix
from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_all_metrics,
    print_metrics_summary,
    format_metrics_for_json
)

__all__ = [
    # Config
    'AttentionMatrixConfig',
    # Models
    'CrossAttentionModel',
    'ImprovedCrossAttentionModel',
    # Data Loading
    'EmbeddingDataLoader',
    'load_data_for_attention_matrix',
    'ProteinLigandDataset',
    'create_dataloaders',
    # Training & Evaluation
    'AttentionTrainer',
    'AttentionEvaluator',
    # Pipeline
    'AttentionMatrixPipeline',
    # Data Splitting
    'LeakageAwareSplitter',
    'SimpleSplitter',
    # Metrics
    'compute_classification_metrics',
    'compute_regression_metrics',
    'compute_all_metrics',
    'print_metrics_summary',
    'format_metrics_for_json',
    # Analysis
    'AttentionAnalyzer',
    'create_attention_heatmap_data',
]

__version__ = '1.0.0'
