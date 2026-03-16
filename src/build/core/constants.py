"""
Constants used throughout the build module.
"""

import os

__all__ = [
    'BuildConstants',
    'DEFAULT_BASE_DIR', 'DEFAULT_LIGAND_DIR', 'DEFAULT_PROTEIN_DIR',
    'DEFAULT_LIGAND_OUTPUT_DIR', 'DEFAULT_PROTEIN_OUTPUT_DIR', 'DEFAULT_MATRIX_OUTPUT_DIR',
    'DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR', 'DEFAULT_LIGAND_MATRIX_OUTPUT_DIR',
    'DEFAULT_CONCATENATED_OUTPUT_DIR', 'DEFAULT_EMBEDDING_TYPE',
    'DEFAULT_LIGAND_DIM', 'DEFAULT_PROTEIN_DIM', 'DEFAULT_BATCH_SIZE',
    'DEFAULT_ESM_MODEL', 'DEFAULT_FM4M_MODEL',
    'ESM_MODELS', 'FM4M_MODELS',
    'SPARK_CONFIG', 'MEMORY_CONFIG',
    'LOG_FORMAT', 'LOG_LEVEL',
    'CHECKPOINT_FILE', 'PROCESSED_FILES_LOG',
    'MIN_EMBEDDING_SIZE', 'MAX_EMBEDDING_SIZE', 'MIN_MATRIX_ROWS',
    'STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM', 'STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD',
    'STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE', 'STRATIFICATION_DEFAULT_STRATIFY_BY',
    'STRATIFICATION_SUPPORTED_ALGORITHMS'
]

class BuildConstants:
    """Container for build system constants."""
    
    # Default directories
    DEFAULT_BASE_DIR = '.'
    DEFAULT_LIGAND_DIR = 'ligand'
    DEFAULT_PROTEIN_DIR = 'protein'
    DEFAULT_LIGAND_OUTPUT_DIR = 'ligand_embeddings'
    DEFAULT_PROTEIN_OUTPUT_DIR = 'protein_embeddings'
    DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR = 'protein_matrix_embeddings'  # NEW: per-token matrices
    DEFAULT_LIGAND_MATRIX_OUTPUT_DIR = 'ligand_molformer_matrices'    # NEW: per-token matrices for MoLFormer
    DEFAULT_MATRIX_OUTPUT_DIR = 'matrix_embedding'
    DEFAULT_CONCATENATED_OUTPUT_DIR = 'concatenated_embeddings'
    
    # File extensions
    NPY_EXTENSION = '.npy'
    TSV_EXTENSION = '.tsv'
    CSV_EXTENSION = '.csv'
    TXT_EXTENSION = '.txt'
    LOG_EXTENSION = '.log'
    
    # Model configurations
    DEFAULT_EMBEDDING_TYPE = 'cls'
    DEFAULT_LIGAND_DIM = 768
    DEFAULT_PROTEIN_DIM = 2560  # ESM-2 t36 3B (modelo mais recente e robusto)
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_ESM_MODEL = 'esm2_t36_3B_UR50D'  # ESM-2 com 3 bilhões de parâmetros
    DEFAULT_FM4M_MODEL = 'SMILES-TED'

# Legacy constants for backward compatibility
DEFAULT_BASE_DIR = BuildConstants.DEFAULT_BASE_DIR
DEFAULT_LIGAND_DIR = BuildConstants.DEFAULT_LIGAND_DIR
DEFAULT_PROTEIN_DIR = BuildConstants.DEFAULT_PROTEIN_DIR
DEFAULT_LIGAND_OUTPUT_DIR = BuildConstants.DEFAULT_LIGAND_OUTPUT_DIR
DEFAULT_PROTEIN_OUTPUT_DIR = BuildConstants.DEFAULT_PROTEIN_OUTPUT_DIR
DEFAULT_MATRIX_OUTPUT_DIR = BuildConstants.DEFAULT_MATRIX_OUTPUT_DIR
DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR = BuildConstants.DEFAULT_PROTEIN_MATRIX_OUTPUT_DIR  # NEW
DEFAULT_LIGAND_MATRIX_OUTPUT_DIR = BuildConstants.DEFAULT_LIGAND_MATRIX_OUTPUT_DIR    # NEW
DEFAULT_CONCATENATED_OUTPUT_DIR = BuildConstants.DEFAULT_CONCATENATED_OUTPUT_DIR
DEFAULT_EMBEDDING_TYPE = BuildConstants.DEFAULT_EMBEDDING_TYPE
DEFAULT_LIGAND_DIM = BuildConstants.DEFAULT_LIGAND_DIM
DEFAULT_PROTEIN_DIM = BuildConstants.DEFAULT_PROTEIN_DIM
DEFAULT_BATCH_SIZE = BuildConstants.DEFAULT_BATCH_SIZE
DEFAULT_ESM_MODEL = BuildConstants.DEFAULT_ESM_MODEL
DEFAULT_FM4M_MODEL = BuildConstants.DEFAULT_FM4M_MODEL

# Modelos ESM disponíveis
# max_len: Tamanho máximo de sequência suportado (conservador para evitar OOM)
# dim: Dimensão do embedding de SAÍDA (todos os embeddings terão este tamanho fixo)
ESM_MODELS = {
    # ESM-2 (Meta AI / Fair-ESM)
    'esm2_t48_15B_UR50D': {'dim': 5120, 'layers': 48, 'max_len': 5120},  # 15B: até 5120 tokens (rotary embeddings + CPU offloading)
    'esm2_t36_3B_UR50D': {'dim': 2560, 'layers': 36, 'max_len': 4096},   # 3B: até 4096 tokens
    'esm2_t33_650M_UR50D': {'dim': 1280, 'layers': 33, 'max_len': 1024}, # 650M: até 1024 tokens
    'esm2_t30_150M_UR50D': {'dim': 640, 'layers': 30, 'max_len': 1024},  # 150M: até 1024 tokens
    'esm2_t12_35M_UR50D': {'dim': 480, 'layers': 12, 'max_len': 1024},   # 35M: até 1024 tokens
    'esm2_t6_8M_UR50D': {'dim': 320, 'layers': 6, 'max_len': 1024},      # 8M: até 1024 tokens
    
    # ESM-C (EvolutionaryScale Cambrian) - Fast representation learning
    'esmc-300m-2024-12': {'dim': 960, 'layers': 30, 'max_len': 2048},    # 300M: 960-dim, mean pooling (local)
    'esmc-600m-2024-12': {'dim': 1152, 'layers': 36, 'max_len': 2048},   # 600M: 1152-dim, mean pooling (local)
    'esmc-6b-2024-12': {'dim': 3072, 'layers': 56, 'max_len': 2048},     # 6B: 3072-dim, via Forge API (requer ESM_API_KEY)
    
    # OpenFold3 (AlphaFold3 reproduction) - Structure-aware embeddings
    'openfold3': {'dim': 384, 'layers': 48, 'max_len': 2048},            # OpenFold3: 384-dim single representation
    
    # Boltz-2 (Biomolecular foundation model) - Structure + affinity prediction
    'boltz2': {'dim': 384, 'layers': 64, 'max_len': 2048},               # Boltz-2: 384-dim single representation (mean pooling)
}


def get_esm_model_info(model_name: str) -> dict:
    """
    Get ESM model configuration.
    
    Args:
        model_name: Name of the ESM model
        
    Returns:
        Dictionary with 'dim', 'layers', 'max_len'
        
    Example:
        >>> get_esm_model_info('esm2_t6_8M_UR50D')
        {'dim': 320, 'layers': 6, 'max_len': 1024}
    """
    if model_name not in ESM_MODELS:
        raise ValueError(f"Unknown ESM model: {model_name}. Available: {list(ESM_MODELS.keys())}")
    return ESM_MODELS[model_name].copy()


def get_esm_max_length(model_name: str) -> int:
    """
    Get maximum sequence length for ESM model.
    
    Always use this function to get max_len - it ensures consistency
    and uses the full capacity of the model.
    
    Args:
        model_name: Name of the ESM model
        
    Returns:
        Maximum sequence length supported by the model
        
    Example:
        >>> get_esm_max_length('esm2_t36_3B_UR50D')
        4096
    """
    return get_esm_model_info(model_name)['max_len']


def get_esm_embedding_dim(model_name: str) -> int:
    """
    Get embedding dimension for ESM model.
    
    Args:
        model_name: Name of the ESM model
        
    Returns:
        Embedding dimension
        
    Example:
        >>> get_esm_embedding_dim('esm2_t36_3B_UR50D')
        2560
    """
    return get_esm_model_info(model_name)['dim']

# Modelos FM4M disponíveis
FM4M_MODELS = {
    'SMI-TED': {'dim': 768, 'type': 'transformer'},
    'SELFIES-TED': {'dim': 768, 'type': 'transformer'},
    'SMI-SSED': {'dim': 768, 'type': 'encoder'},
    'MHG': {'dim': 768, 'type': 'graph'},
    'MOL-MOE': {'dim': 768, 'type': 'mixture'},
    'MOLFORMER': {'dim': 768, 'type': 'transformer'},
    'CHEMBERTA': {'dim': 384, 'type': 'transformer'},
}

# ChemBERTa model configuration (same as GraphBAN)
CHEMBERTA_MODEL_NAME = 'DeepChem/ChemBERTa-77M-MTR'
CHEMBERTA_DIM = 384
CHEMBERTA_MAX_LEN = 290

# Configurações de Spark
SPARK_CONFIG = {
    'app_name': 'DockTKinase-Build',
    'memory_fraction': 0.8,
    'offheap_fraction': 0.2,
    'gc_type': 'G1GC'
}

# Configurações de memória
MEMORY_CONFIG = {
    'low_memory_threshold': 4,  # GB
    'high_memory_threshold': 16,  # GB
    'batch_size_factor': 2
}

# Logging
LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
LOG_LEVEL = 'INFO'

# Checkpoints
CHECKPOINT_FILE = 'build_checkpoint.txt'
PROCESSED_FILES_LOG = 'processed_files.log'

# Validação
MIN_EMBEDDING_SIZE = 100
MAX_EMBEDDING_SIZE = 10000
MIN_MATRIX_ROWS = 1
MAX_MATRIX_ROWS = 1000000

# Estratificação
STRATIFICATION_DEFAULT_CLUSTERING_ALGORITHM = 'dbscan'
STRATIFICATION_DEFAULT_SIMILARITY_THRESHOLD = 0.8
STRATIFICATION_DEFAULT_CLUSTER_MIN_SIZE = 5
STRATIFICATION_DEFAULT_STRATIFY_BY = 'both'  # 'ligand', 'protein', 'both', 'combined'
STRATIFICATION_SUPPORTED_ALGORITHMS = ['dbscan', 'hierarchical', 'kmeans', 'random']
