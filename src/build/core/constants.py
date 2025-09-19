"""
Constants used throughout the build module.
"""

import os

__all__ = [
    'BuildConstants',
    'DEFAULT_BASE_DIR', 'DEFAULT_LIGAND_DIR', 'DEFAULT_PROTEIN_DIR',
    'DEFAULT_LIGAND_OUTPUT_DIR', 'DEFAULT_PROTEIN_OUTPUT_DIR', 'DEFAULT_MATRIX_OUTPUT_DIR',
    'DEFAULT_CONCATENATED_OUTPUT_DIR', 'DEFAULT_EMBEDDING_TYPE',
    'DEFAULT_LIGAND_DIM', 'DEFAULT_PROTEIN_DIM', 'DEFAULT_BATCH_SIZE',
    'DEFAULT_ESM_MODEL', 'DEFAULT_FM4M_MODEL',
    'ESM_MODELS', 'FM4M_MODELS',
    'SPARK_CONFIG', 'MEMORY_CONFIG',
    'LOG_FORMAT', 'LOG_LEVEL',
    'CHECKPOINT_FILE', 'PROCESSED_FILES_LOG',
    'MIN_EMBEDDING_SIZE', 'MAX_EMBEDDING_SIZE', 'MIN_MATRIX_ROWS'
]

class BuildConstants:
    """Container for build system constants."""
    
    # Default directories
    DEFAULT_BASE_DIR = '.'
    DEFAULT_LIGAND_DIR = 'ligand'
    DEFAULT_PROTEIN_DIR = 'protein'
    DEFAULT_LIGAND_OUTPUT_DIR = 'ligand_embeddings'
    DEFAULT_PROTEIN_OUTPUT_DIR = 'protein_embeddings'
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
    DEFAULT_PROTEIN_DIM = 2560
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_ESM_MODEL = 'esm2_t33_650M_UR50D'
    DEFAULT_FM4M_MODEL = 'SMILES-TED'

# Legacy constants for backward compatibility
DEFAULT_BASE_DIR = BuildConstants.DEFAULT_BASE_DIR
DEFAULT_LIGAND_DIR = BuildConstants.DEFAULT_LIGAND_DIR
DEFAULT_PROTEIN_DIR = BuildConstants.DEFAULT_PROTEIN_DIR
DEFAULT_LIGAND_OUTPUT_DIR = BuildConstants.DEFAULT_LIGAND_OUTPUT_DIR
DEFAULT_PROTEIN_OUTPUT_DIR = BuildConstants.DEFAULT_PROTEIN_OUTPUT_DIR
DEFAULT_MATRIX_OUTPUT_DIR = BuildConstants.DEFAULT_MATRIX_OUTPUT_DIR
DEFAULT_CONCATENATED_OUTPUT_DIR = BuildConstants.DEFAULT_CONCATENATED_OUTPUT_DIR
DEFAULT_EMBEDDING_TYPE = BuildConstants.DEFAULT_EMBEDDING_TYPE
DEFAULT_LIGAND_DIM = BuildConstants.DEFAULT_LIGAND_DIM
DEFAULT_PROTEIN_DIM = BuildConstants.DEFAULT_PROTEIN_DIM
DEFAULT_BATCH_SIZE = BuildConstants.DEFAULT_BATCH_SIZE
DEFAULT_ESM_MODEL = BuildConstants.DEFAULT_ESM_MODEL
DEFAULT_FM4M_MODEL = BuildConstants.DEFAULT_FM4M_MODEL

# Modelos ESM disponíveis
ESM_MODELS = {
    'esm2_t48_15B_UR50D': {'dim': 5120, 'layers': 48},
    'esm2_t36_3B_UR50D': {'dim': 2560, 'layers': 36},
    'esm2_t33_650M_UR50D': {'dim': 1280, 'layers': 33},
    'esm2_t30_150M_UR50D': {'dim': 640, 'layers': 30},
    'esm2_t12_35M_UR50D': {'dim': 480, 'layers': 12},
    'esm2_t6_8M_UR50D': {'dim': 320, 'layers': 6}
}

# Modelos FM4M disponíveis
FM4M_MODELS = {
    'SMI-TED': {'dim': 768, 'type': 'transformer'},
    'SELFIES-TED': {'dim': 768, 'type': 'transformer'},
    'SMI-SSED': {'dim': 768, 'type': 'encoder'},
    'MHG': {'dim': 768, 'type': 'graph'},
    'MOL-MOE': {'dim': 768, 'type': 'mixture'}
}

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
