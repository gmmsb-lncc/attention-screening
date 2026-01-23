"""
Model Registry for Embeddings

Centralized registry for all supported embedding models.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an embedding model."""
    name: str
    type: str  # 'esm' or 'fm4m'
    embedding_dim: int
    description: str
    default_layer: Optional[int] = None
    requires_gpu: bool = False


class ModelRegistry:
    """
    Registry of all supported embedding models.
    
    Provides:
    - Model discovery
    - Model information lookup
    - Model validation
    - Factory methods
    """
    
    # ESM Models
    ESM_MODELS = {
        'esm2_t48_15B_UR50D': ModelInfo(
            name='esm2_t48_15B_UR50D',
            type='esm',
            embedding_dim=5120,
            description='ESM-2 15B parameters (best quality, requires significant GPU)',
            default_layer=48,
            requires_gpu=True
        ),
        'esm2_t36_3B_UR50D': ModelInfo(
            name='esm2_t36_3B_UR50D',
            type='esm',
            embedding_dim=2560,
            description='ESM-2 3B parameters (high quality)',
            default_layer=36,
            requires_gpu=True
        ),
        'esm2_t33_650M_UR50D': ModelInfo(
            name='esm2_t33_650M_UR50D',
            type='esm',
            embedding_dim=1280,
            description='ESM-2 650M parameters (good balance, default)',
            default_layer=33,
            requires_gpu=False
        ),
        'esm2_t30_150M_UR50D': ModelInfo(
            name='esm2_t30_150M_UR50D',
            type='esm',
            embedding_dim=640,
            description='ESM-2 150M parameters (fast)',
            default_layer=30,
            requires_gpu=False
        ),
        'esm2_t12_35M_UR50D': ModelInfo(
            name='esm2_t12_35M_UR50D',
            type='esm',
            embedding_dim=480,
            description='ESM-2 35M parameters (fastest)',
            default_layer=12,
            requires_gpu=False
        ),
        'esm2_t6_8M_UR50D': ModelInfo(
            name='esm2_t6_8M_UR50D',
            type='esm',
            embedding_dim=320,
            description='ESM-2 8M parameters (ultra-fast)',
            default_layer=6,
            requires_gpu=False
        ),
        'esm1b_t33_650M_UR50S': ModelInfo(
            name='esm1b_t33_650M_UR50S',
            type='esm',
            embedding_dim=1280,
            description='ESM-1b 650M parameters (legacy)',
            default_layer=33,
            requires_gpu=False
        ),
        'esmc-300m-2024-12': ModelInfo(
            name='esmc-300m-2024-12',
            type='esm',
            embedding_dim=960,
            description='ESM-C 300M parameters (ESM-3 Cambrian)',
            requires_gpu=False
        ),
        'esmc-600m-2024-12': ModelInfo(
            name='esmc-600m-2024-12',
            type='esm',
            embedding_dim=1152,
            description='ESM-C 600M parameters (ESM-3 Cambrian)',
            requires_gpu=True
        ),
        'openfold3': ModelInfo(
            name='openfold3',
            type='esm',
            embedding_dim=384,
            description='OpenFold3 - structure-aware embeddings',
            requires_gpu=True
        ),
        'boltz2': ModelInfo(
            name='boltz2',
            type='esm',
            embedding_dim=384,
            description='Boltz-2 - structure + affinity prediction (single representation)',
            requires_gpu=True
        )
    }
    
    # FM4M models (SMI-TED)
    # Note: Only Light model is available on HuggingFace
    FM4M_MODELS = {
        'smi_ted_light': ModelInfo(
            name='smi_ted_light',
            type='fm4m',
            embedding_dim=768,
            description='SMI-TED Light model (40 layers) for molecular embeddings'
        ),
        'molformer': ModelInfo(
            name='molformer',
            type='fm4m',
            embedding_dim=768,
            description='MoLFormer-XL model (12 layers) for molecular embeddings - supports per-token representations'
        )
    }
    
    # Combined registry
    ALL_MODELS = {**ESM_MODELS, **FM4M_MODELS}
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object or None if not found
        """
        return cls.ALL_MODELS.get(model_name)
    
    @classmethod
    def is_valid_model(cls, model_name: str, model_type: Optional[str] = None) -> bool:
        """
        Check if a model name is valid.
        
        Args:
            model_name: Name of the model
            model_type: Optional type filter ('esm' or 'fm4m')
            
        Returns:
            True if model is valid
        """
        if model_name not in cls.ALL_MODELS:
            return False
        
        if model_type:
            return cls.ALL_MODELS[model_name].type == model_type
        
        return True
    
    @classmethod
    def get_models_by_type(cls, model_type: str) -> Dict[str, ModelInfo]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: 'esm' or 'fm4m'
            
        Returns:
            Dictionary of model names to ModelInfo
        """
        if model_type == 'esm':
            return cls.ESM_MODELS.copy()
        elif model_type == 'fm4m':
            return cls.FM4M_MODELS.copy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_default_model(cls, model_type: str) -> str:
        """
        Get default model name for a type.
        
        Args:
            model_type: 'esm' or 'fm4m'
            
        Returns:
            Default model name
        """
        if model_type == 'esm':
            return 'esm2_t33_650M_UR50D'
        elif model_type == 'fm4m':
            return 'smi_ted_light'  # Only Light model available on HuggingFace
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_embedding_dim(cls, model_name: str) -> int:
        """
        Get embedding dimension for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Embedding dimension
        """
        info = cls.get_model_info(model_name)
        if info is None:
            raise ValueError(f"Unknown model: {model_name}")
        return info.embedding_dim
    
    @classmethod
    def get_repr_layer(cls, model_name: str) -> Optional[int]:
        """
        Get default representation layer for ESM models.
        
        Args:
            model_name: Name of the ESM model
            
        Returns:
            Default layer number or None
        """
        info = cls.get_model_info(model_name)
        if info is None or info.type != 'esm':
            return None
        return info.default_layer
    
    @classmethod
    def list_models(cls, model_type: Optional[str] = None, gpu_only: bool = False) -> List[str]:
        """
        List available models.
        
        Args:
            model_type: Optional type filter ('esm' or 'fm4m')
            gpu_only: If True, only return GPU-required models
            
        Returns:
            List of model names
        """
        models = cls.ALL_MODELS
        
        if model_type:
            models = {k: v for k, v in models.items() if v.type == model_type}
        
        if gpu_only:
            models = {k: v for k, v in models.items() if v.requires_gpu}
        
        return sorted(models.keys())
    
    @classmethod
    def print_models(cls, model_type: Optional[str] = None):
        """
        Print formatted list of available models.
        
        Args:
            model_type: Optional type filter ('esm' or 'fm4m')
        """
        models = cls.get_models_by_type(model_type) if model_type else cls.ALL_MODELS
        
        print(f"\n{'='*80}")
        print(f"Available {'ESM' if model_type == 'esm' else 'FM4M' if model_type == 'fm4m' else 'Embedding'} Models")
        print(f"{'='*80}\n")
        
        for name, info in sorted(models.items()):
            print(f"📦 {name}")
            print(f"   Type: {info.type.upper()}")
            print(f"   Embedding Dimension: {info.embedding_dim}")
            print(f"   Description: {info.description}")
            if info.default_layer:
                print(f"   Default Layer: {info.default_layer}")
            if info.requires_gpu:
                print(f"   ⚠️  Requires GPU")
            print()
