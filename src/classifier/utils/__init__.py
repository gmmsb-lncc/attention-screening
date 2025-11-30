"""Utilitários, métricas e validação de dados."""

# Imports dos utilitários modularizados  
from .import_utils import (
    get_classifier_root,
    setup_classifier_imports,
    safe_import,
    safe_import_optional
)

# Matrix embedding utilities
from .matrix_dataloader import (
    MatrixEmbeddingDataset,
    VectorEmbeddingDataset,
    collate_matrix_batch,
    create_matrix_dataloader
)

from .matrix_trainer import (
    TrainingConfig,
    MetricTracker,
    EarlyStopping,
    CrossAttentionTrainer,
    train_cross_attention_model
)

from .matrix_pipeline import (
    MatrixPipelineConfig,
    MatrixAffinityPipeline,
    run_matrix_pipeline
)

from .matrix_metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    MultiTaskMetrics,
    MatrixMetricsCalculator
)

from .attention_analysis import (
    AttentionResult,
    AttentionExtractor,
    ProteinLigandAttentionAnalyzer,
    save_attention_analysis,
    generate_attention_report,
    plot_attention_heatmap
)

__all__ = [
    # Import utils
    "get_classifier_root", 
    "setup_classifier_imports",
    "safe_import",
    "safe_import_optional",
    # Matrix dataloader
    "MatrixEmbeddingDataset",
    "VectorEmbeddingDataset",
    "collate_matrix_batch",
    "create_matrix_dataloader",
    # Matrix trainer
    "TrainingConfig",
    "MetricTracker",
    "EarlyStopping",
    "CrossAttentionTrainer",
    "train_cross_attention_model",
    # Matrix pipeline
    "MatrixPipelineConfig",
    "MatrixAffinityPipeline",
    "run_matrix_pipeline",
    # Matrix metrics
    "ClassificationMetrics",
    "RegressionMetrics",
    "MultiTaskMetrics",
    "MatrixMetricsCalculator",
    # Attention analysis
    "AttentionResult",
    "AttentionExtractor",
    "ProteinLigandAttentionAnalyzer",
    "save_attention_analysis",
    "generate_attention_report",
    "plot_attention_heatmap"
]
