"""
Model Manager for Embedding Generation

Handles loading and managing ESM and FM4M models.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class ModelManager:
    """
    Manages loading and caching of embedding models (ESM, FM4M).
    
    Features:
    - Lazy loading of models
    - Device management (CPU/GPU)
    - Model caching
    - Memory optimization
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize ModelManager.
        
        Args:
            use_gpu: Whether to use GPU if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if self.verbose:
            print(f"   🔧 ModelManager initialized on device: {self.device}")
        
        # Model caches
        self._esm_models = {}
        self._fm4m_models = {}
        self._model_info = {}
    
    def load_esm_model(
        self,
        model_name: str,
        repr_layer: int = 33
    ) -> Tuple[Any, Any]:
        """
        Load ESM protein language model.
        
        Args:
            model_name: Name of ESM model (e.g., 'esm2_t33_650M_UR50D')
            repr_layer: Layer to extract representations from
            
        Returns:
            Tuple of (model, alphabet)
        """
        # Check cache
        cache_key = f"{model_name}_{repr_layer}"
        if cache_key in self._esm_models:
            if self.verbose:
                print(f"   ♻️  Using cached ESM model: {model_name}")
            return self._esm_models[cache_key]
        
        if self.verbose:
            print(f"   📥 Loading ESM model: {model_name}")
        
        try:
            import esm
            
            # Load model
            if hasattr(esm.pretrained, model_name):
                model, alphabet = getattr(esm.pretrained, model_name)()
            else:
                raise ValueError(f"Unknown ESM model: {model_name}")
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            
            # Cache
            self._esm_models[cache_key] = (model, alphabet)
            self._model_info[cache_key] = {
                'type': 'esm',
                'name': model_name,
                'repr_layer': repr_layer,
                'device': str(self.device),
                'embedding_dim': model.embed_dim
            }
            
            if self.verbose:
                print(f"   ✅ ESM model loaded successfully")
                print(f"      Embedding dimension: {model.embed_dim}")
            
            return model, alphabet
            
        except ImportError:
            raise ImportError(
                "ESM library not installed. Install with: pip install fair-esm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM model: {e}")
    
    def load_fm4m_model(
        self,
        model_name: str = 'default',
        model_path: Optional[Path] = None
    ) -> Any:
        """
        Load FM4M small molecule model.
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to model files (if not using default)
            
        Returns:
            Loaded FM4M model
        """
        # Check cache
        cache_key = f"fm4m_{model_name}"
        if cache_key in self._fm4m_models:
            if self.verbose:
                print(f"   ♻️  Using cached FM4M model: {model_name}")
            return self._fm4m_models[cache_key]
        
        if self.verbose:
            print(f"   📥 Loading FM4M model: {model_name}")
        
        try:
            # Import FM4M modules
            import sys
            fm4m_path = Path(__file__).parent.parent.parent.parent.parent / 'FM4M'
            if str(fm4m_path) not in sys.path:
                sys.path.insert(0, str(fm4m_path))
            
            # Only Light model is available on HuggingFace
            checkpoint_file = 'smi-ted-Light_40.pt'
            version = 'Light'
            from models.smi_ted.smi_ted_light.load import load_smi_ted
            
            # Determine model path
            if model_path is None:
                model_path = fm4m_path / 'model_files'
            
            # Check if checkpoint file exists
            ckpt_path = Path(model_path) / checkpoint_file
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"FM4M checkpoint not found: {checkpoint_file}\n"
                    f"Expected at: {ckpt_path}\n"
                    f"Download from: https://huggingface.co/ibm/materials.smi-ted"
                )
            
            # Load model - load_smi_ted returns the model instance
            model = load_smi_ted(
                folder=str(model_path),
                ckpt_filename=checkpoint_file
            )
            
            # Cache
            self._fm4m_models[cache_key] = model
            self._model_info[cache_key] = {
                'type': 'fm4m',
                'name': model_name,
                'path': str(model_path),
                'embedding_dim': 768,  # FM4M embedding dimension
                'version': version,
                'checkpoint': checkpoint_file
            }
            
            if self.verbose:
                print(f"   ✅ FM4M model loaded successfully (SMI-TED {version})")
                print(f"      Checkpoint: {checkpoint_file}")
                print(f"      Embedding dimension: 768")
            
            return model
            
        except ImportError as e:
            raise ImportError(
                f"FM4M modules not found. Check FM4M installation: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FM4M model: {e}")
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            Dictionary with model information
        """
        return self._model_info.get(model_key, {})
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_type: Type of models to clear ('esm', 'fm4m', or None for all)
        """
        if model_type == 'esm' or model_type is None:
            self._esm_models.clear()
            if self.verbose:
                print("   🗑️  Cleared ESM model cache")
        
        if model_type == 'fm4m' or model_type is None:
            self._fm4m_models.clear()
            if self.verbose:
                print("   🗑️  Cleared FM4M model cache")
        
        # Clear matching model info
        if model_type:
            keys_to_remove = [
                k for k, v in self._model_info.items() 
                if v.get('type') == model_type
            ]
            for key in keys_to_remove:
                del self._model_info[key]
        else:
            self._model_info.clear()
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'device': str(self.device),
            'device_type': self.device.type,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        esm_count = len(self._esm_models)
        fm4m_count = len(self._fm4m_models)
        return (
            f"ModelManager(device={self.device}, "
            f"esm_models={esm_count}, fm4m_models={fm4m_count})"
        )
