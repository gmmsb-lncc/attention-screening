"""
Model Manager for Embedding Generation

Handles loading and managing ESM and FM4M models with advanced memory optimization.
"""

import torch
from typing import Optional, Dict, Any, Tuple, Literal
from pathlib import Path
import warnings


class ModelManager:
    """
    Manages loading and caching of embedding models (ESM, FM4M) with memory optimization.
    
    Features:
    - Lazy loading of models
    - Device management (CPU/GPU)
    - Model caching
    - **NEW**: CPU offloading for large models (ESM-2 3B, 15B)
    - **NEW**: Mixed precision support (FP16/BF16)
    - **NEW**: 8-bit quantization support
    - **NEW**: Automatic device_map for multi-GPU
    
    Memory Optimization Strategies:
    - Large models (>2B params): Automatic CPU offloading
    - Medium models (650M-2B): Optional mixed precision
    - Small models (<650M): Standard loading
    """
    
    # Model size thresholds (in millions of parameters)
    _MODEL_SIZES = {
        'esm2_t48_15B_UR50D': 15000,
        'esm2_t36_3B_UR50D': 3000,
        'esm2_t33_650M_UR50D': 650,
        'esm2_t30_150M_UR50D': 150,
        'esm2_t12_35M_UR50D': 35,
        'esm2_t6_8M_UR50D': 8,
    }
    
    def __init__(
        self,
        use_gpu: bool = False,
        device: Optional[str] = None,
        enable_offload: bool = True,
        use_mixed_precision: bool = False,
        use_8bit: bool = False,
        max_memory_gpu: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize ModelManager with advanced memory management.
        
        Args:
            use_gpu: Whether to use GPU if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            enable_offload: Enable CPU offloading for large models (default: True)
            use_mixed_precision: Use FP16/BF16 for faster inference (default: False)
            use_8bit: Use 8-bit quantization (requires bitsandbytes, default: False)
            max_memory_gpu: Max GPU memory per device (e.g., "10GB", "0.8" for 80%)
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.enable_offload = enable_offload
        self.use_mixed_precision = use_mixed_precision
        self.use_8bit = use_8bit
        self.max_memory_gpu = max_memory_gpu
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Check for accelerate library (for offloading)
        self.has_accelerate = False
        if enable_offload:
            try:
                import accelerate
                self.has_accelerate = True
            except ImportError:
                if self.verbose:
                    warnings.warn(
                        "accelerate library not found. CPU offloading disabled.\n"
                        "Install with: pip install accelerate"
                    )
                self.enable_offload = False
        
        # Check for bitsandbytes (for 8-bit)
        if use_8bit:
            try:
                import bitsandbytes
            except ImportError:
                if self.verbose:
                    warnings.warn(
                        "bitsandbytes library not found. 8-bit quantization disabled.\n"
                        "Install with: pip install bitsandbytes"
                    )
                self.use_8bit = False
        
        if self.verbose:
            print(f"   🔧 ModelManager initialized")
            print(f"      Device: {self.device}")
            if self.enable_offload:
                print(f"      CPU Offloading: ✅ Enabled (for models >2B params)")
            if self.use_mixed_precision:
                print(f"      Mixed Precision: ✅ Enabled (FP16/BF16)")
            if self.use_8bit:
                print(f"      8-bit Quantization: ✅ Enabled")
            if self.max_memory_gpu:
                print(f"      Max GPU Memory: {self.max_memory_gpu}")
        
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
        Load ESM protein language model with automatic memory optimization.
        
        Automatically applies memory optimization strategies based on model size:
        - Small models (<650M): Standard loading
        - Medium models (650M-2B): Optional mixed precision
        - Large models (>2B): CPU offloading + mixed precision
        
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
            
            # Get model size
            model_size_mb = self._MODEL_SIZES.get(model_name, 0)
            is_large_model = model_size_mb >= 2000  # >= 2B params
            is_medium_model = 650 <= model_size_mb < 2000
            
            if self.verbose and is_large_model:
                print(f"      ⚠️  Large model detected ({model_size_mb}M params)")
                print(f"      Applying memory optimizations...")
            
            # Load model
            if hasattr(esm.pretrained, model_name):
                model, alphabet = getattr(esm.pretrained, model_name)()
            else:
                raise ValueError(f"Unknown ESM model: {model_name}")
            
            # Apply memory optimizations for large models
            if is_large_model and self.enable_offload and self.has_accelerate:
                model = self._apply_cpu_offload(model, model_name)
            elif is_medium_model and self.use_mixed_precision:
                model = self._apply_mixed_precision(model)
            elif not is_large_model:
                # Standard loading for small models
                model = model.to(self.device)
            
            # Apply 8-bit quantization if requested (only for CUDA)
            if self.use_8bit and self.device.type == 'cuda':
                model = self._apply_8bit_quantization(model)
            
            model.eval()
            
            # Cache
            self._esm_models[cache_key] = (model, alphabet)
            self._model_info[cache_key] = {
                'type': 'esm',
                'name': model_name,
                'repr_layer': repr_layer,
                'device': str(self.device),
                'embedding_dim': model.embed_dim,
                'model_size_mb': model_size_mb,
                'optimizations': self._get_applied_optimizations(is_large_model, is_medium_model)
            }
            
            if self.verbose:
                print(f"   ✅ ESM model loaded successfully")
                print(f"      Embedding dimension: {model.embed_dim}")
                print(f"      Model size: {model_size_mb}M parameters")
                opts = self._model_info[cache_key]['optimizations']
                if opts:
                    print(f"      Optimizations: {', '.join(opts)}")
            
            return model, alphabet
            
        except ImportError:
            raise ImportError(
                "ESM library not installed. Install with: pip install fair-esm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM model: {e}")
    
    def _apply_cpu_offload(self, model: Any, model_name: str) -> Any:
        """
        Apply CPU offloading for large models using accelerate.
        
        Args:
            model: PyTorch model
            model_name: Model identifier for logging
            
        Returns:
            Model with CPU offloading enabled
        """
        try:
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory
            
            if self.verbose:
                print(f"      🔄 Applying CPU offloading...")
            
            # Calculate max memory for each device
            max_memory = {}
            if torch.cuda.is_available():
                # Get available GPU memory
                for i in range(torch.cuda.device_count()):
                    if self.max_memory_gpu:
                        max_memory[i] = self.max_memory_gpu
                    else:
                        # Use 80% of available GPU memory by default
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        max_memory[i] = int(total_memory * 0.8)
            
            # Add CPU as fallback
            max_memory["cpu"] = "100GiB"  # Generous CPU memory limit
            
            # Infer device map automatically
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["ESM1bLayerNorm", "Attention", "TransformerLayer"],
            )
            
            if self.verbose:
                gpu_layers = sum(1 for v in device_map.values() if isinstance(v, int))
                cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
                print(f"      ✅ Device map created: {gpu_layers} GPU layers, {cpu_layers} CPU layers")
            
            # Dispatch model to devices
            model = dispatch_model(model, device_map=device_map)
            
            return model
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"CPU offloading failed, falling back to standard loading: {e}")
            return model.to(self.device)
    
    def _apply_mixed_precision(self, model: Any) -> Any:
        """
        Apply mixed precision (FP16/BF16) to model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model in mixed precision
        """
        if self.verbose:
            print(f"      🎯 Applying mixed precision...")
        
        try:
            # Use BF16 if available (better for transformers), else FP16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # Ampere or newer supports BF16
                dtype = torch.bfloat16
                precision_name = "BF16"
            else:
                dtype = torch.float16
                precision_name = "FP16"
            
            model = model.to(self.device).to(dtype)
            
            if self.verbose:
                print(f"      ✅ Using {precision_name} precision")
            
            return model
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Mixed precision failed, using FP32: {e}")
            return model.to(self.device)
    
    def _apply_8bit_quantization(self, model: Any) -> Any:
        """
        Apply 8-bit quantization to model (requires bitsandbytes).
        
        Args:
            model: PyTorch model
            
        Returns:
            Quantized model
        """
        if self.verbose:
            print(f"      🔢 Applying 8-bit quantization...")
        
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Linear8bitLt
            
            # Replace Linear layers with 8-bit versions
            def replace_linear_with_8bit(module):
                for name, child in module.named_children():
                    if isinstance(child, torch.nn.Linear):
                        # Replace with 8-bit linear
                        setattr(
                            module,
                            name,
                            Linear8bitLt(
                                child.in_features,
                                child.out_features,
                                child.bias is not None,
                            )
                        )
                    else:
                        replace_linear_with_8bit(child)
            
            replace_linear_with_8bit(model)
            model = model.to(self.device)
            
            if self.verbose:
                print(f"      ✅ 8-bit quantization applied")
            
            return model
            
        except Exception as e:
            if self.verbose:
                warnings.warn(f"8-bit quantization failed, using FP32: {e}")
            return model.to(self.device)
    
    def _get_applied_optimizations(self, is_large: bool, is_medium: bool) -> list:
        """Get list of applied optimizations."""
        opts = []
        if is_large and self.enable_offload and self.has_accelerate:
            opts.append("CPU Offloading")
        if (is_large or is_medium) and self.use_mixed_precision:
            opts.append("Mixed Precision (FP16/BF16)")
        if self.use_8bit and self.device.type == 'cuda':
            opts.append("8-bit Quantization")
        return opts
    
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
