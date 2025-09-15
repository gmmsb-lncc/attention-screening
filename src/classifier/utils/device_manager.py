"""
Sistema Consolidado de Gerenciamento de Devices para DockTKinase.

Combina funcionalidades dos sistemas simples e complexo em uma interface unificada.
Oferece detecção automática, validação robusta e fallback inteligente.
"""

import torch
import subprocess
import platform
import logging
import psutil
import time
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Informações detalhadas sobre um device."""
    
    device: torch.device
    name: str
    type: str  # "cuda", "cpu", "mps" 
    index: Optional[int] = None
    
    # Capacidades
    total_memory: Optional[float] = None  # GB
    available_memory: Optional[float] = None  # GB
    compute_capability: Optional[Tuple[int, int]] = None
    
    # Performance
    benchmark_score: Optional[float] = None
    is_available: bool = True
    is_recommended: bool = False
    
    # Issues
    warnings: List[str] = None
    limitations: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.limitations is None:
            self.limitations = []
    
    def get_memory_gb(self) -> str:
        """Retorna memória formatada."""
        if self.total_memory:
            return f"{self.total_memory:.1f}GB"
        return "N/A"
    
    def get_capability_str(self) -> str:
        """Retorna compute capability formatada."""
        if self.compute_capability:
            return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return "N/A"
    
    def get_summary(self) -> str:
        """Retorna resumo formatado do device."""
        summary = f"{self.name} ({self.type})"
        if self.total_memory:
            summary += f" | {self.get_memory_gb()}"
        if self.is_recommended:
            summary += " - ✅ RECOMENDADO"
        elif self.is_available:
            summary += " - 💡 disponível"
        else:
            summary += " - ❌ indisponível"
        return summary


class DeviceManager:
    """
    Gerenciador consolidado de devices com múltiplos modos de operação.
    
    Modos:
    - simple: Detecção rápida, mínima validação
    - smart: Detecção com validação e ranking
    - complex: Detecção completa, benchmark e otimização
    """
    
    def __init__(self, 
                 mode: str = "smart",
                 min_gpu_memory_gb: float = 1.0,
                 enable_benchmarking: bool = False,
                 prefer_gpu: bool = True):
        """
        Inicializa o gerenciador de devices.
        
        Args:
            mode: "simple", "smart", "complex"
            min_gpu_memory_gb: Memória GPU mínima
            enable_benchmarking: Ativar benchmark de performance
            prefer_gpu: Preferir GPU quando disponível
        """
        self.mode = mode
        self.min_gpu_memory_gb = min_gpu_memory_gb
        self.enable_benchmarking = enable_benchmarking
        self.prefer_gpu = prefer_gpu
        
        # Cache
        self._device_cache: Dict[str, DeviceInfo] = {}
        self._validation_cache: Dict[str, bool] = {}
        self._selected_device: Optional[DeviceInfo] = None
        
        logger.info(f"🔧 DeviceManager inicializado (modo: {mode})")
        if mode != "simple":
            logger.info(f"   • Memória GPU mínima: {min_gpu_memory_gb:.1f}GB")
            logger.info(f"   • Benchmark ativo: {enable_benchmarking}")
    
    def get_device(self, requirement: str = "auto") -> torch.device:
        """
        Obtém o melhor device baseado no modo e requisitos.
        
        Args:
            requirement: "auto", "gpu_only", "cpu_only", "fastest"
            
        Returns:
            torch.device otimizado
        """
        if self.mode == "simple":
            return self._get_device_simple(requirement)
        elif self.mode == "smart":
            return self._get_device_smart(requirement)
        else:  # complex
            return self._get_device_complex(requirement)
    
    def get_device_info(self) -> Optional[DeviceInfo]:
        """Retorna informações do device selecionado."""
        return self._selected_device
    
    def get_available_devices(self) -> List[DeviceInfo]:
        """Retorna lista de todos os devices disponíveis."""
        if self.mode == "simple":
            return self._detect_devices_simple()
        else:
            return self._detect_devices_detailed()
    
    def validate_device_status(self) -> bool:
        """Valida se o device atual está funcionando."""
        if not self._selected_device:
            return False
        
        try:
            test_tensor = torch.randn(10, 10).to(self._selected_device.device)
            result = test_tensor @ test_tensor
            del test_tensor, result
            return True
        except Exception as e:
            logger.warning(f"Device validation failed: {e}")
            return False
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor para o device selecionado."""
        if self._selected_device:
            return tensor.to(self._selected_device.device)
        return tensor.to(self._get_fallback_device())
    
    # ==================== MODO SIMPLE ====================
    
    def _get_device_simple(self, requirement: str) -> torch.device:
        """Modo simples: detecção rápida sem validação extensiva."""
        if requirement == "cpu_only":
            device = torch.device("cpu")
        elif requirement == "gpu_only":
            device = self._get_best_gpu_simple()
            if device is None:
                raise RuntimeError("GPU não disponível")
        else:  # auto, fastest
            device = self._get_best_gpu_simple()
            if device is None:
                device = torch.device("cpu")
        
        # Criar DeviceInfo simples
        self._selected_device = DeviceInfo(
            device=device,
            name=self._get_device_name_simple(device),
            type=device.type,
            is_available=True
        )
        
        logger.info(f"Device selecionado: {device}")
        return device
    
    def _get_best_gpu_simple(self) -> Optional[torch.device]:
        """Seleciona melhor GPU disponível (modo simples)."""
        # CUDA
        if torch.cuda.is_available():
            return torch.device('cuda')
        
        # Apple MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        
        return None
    
    def _get_device_name_simple(self, device: torch.device) -> str:
        """Obtém nome do device (modo simples)."""
        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device.index or 0)
                return props.name
            except:
                return "CUDA GPU"
        elif device.type == "mps":
            return "Apple Metal Performance Shaders"
        else:
            return f"CPU ({platform.processor() or 'Unknown'})"
    
    def _detect_devices_simple(self) -> List[DeviceInfo]:
        """Detecta devices no modo simples."""
        devices = []
        
        # CPU
        devices.append(DeviceInfo(
            device=torch.device("cpu"),
            name=f"CPU ({platform.processor() or 'Unknown'})",
            type="cpu",
            is_available=True
        ))
        
        # GPU
        gpu_device = self._get_best_gpu_simple()
        if gpu_device:
            devices.append(DeviceInfo(
                device=gpu_device,
                name=self._get_device_name_simple(gpu_device),
                type=gpu_device.type,
                is_available=True,
                is_recommended=True
            ))
        
        return devices
    
    # ==================== MODO SMART/COMPLEX ====================
    
    def _get_device_smart(self, requirement: str) -> torch.device:
        """Modo smart: detecção com validação e ranking."""
        devices = self._detect_devices_detailed()
        
        if not devices:
            raise RuntimeError("Nenhum device válido encontrado")
        
        selected = self._select_device_by_requirement(devices, requirement)
        self._selected_device = selected
        
        logger.info(f"✅ Device selecionado: {selected.get_summary()}")
        self._log_device_warnings(selected)
        
        return selected.device
    
    def _get_device_complex(self, requirement: str) -> torch.device:
        """Modo complex: detecção completa com benchmark."""
        devices = self._detect_devices_detailed()
        
        # Benchmark se habilitado
        if self.enable_benchmarking:
            devices = self._benchmark_devices(devices)
        
        if not devices:
            # Fallback para CPU
            logger.warning("🔄 Fallback para CPU...")
            cpu_device = torch.device("cpu")
            self._selected_device = DeviceInfo(
                device=cpu_device,
                name=f"CPU ({platform.processor() or 'Unknown'})",
                type="cpu",
                is_available=True
            )
            return cpu_device
        
        selected = self._select_device_by_requirement(devices, requirement)
        self._selected_device = selected
        
        logger.info(f"✅ Device selecionado: {selected.get_summary()}")
        if selected.benchmark_score:
            logger.info(f"   📊 Score de benchmark: {selected.benchmark_score:.2f}")
        self._log_device_warnings(selected)
        
        return selected.device
    
    def _detect_devices_detailed(self) -> List[DeviceInfo]:
        """Detecta devices com informações detalhadas."""
        logger.info("🔍 Detectando devices disponíveis...")
        
        devices = []
        
        # 1. CPU
        cpu_info = self._get_cpu_info()
        if self._validate_device(cpu_info):
            devices.append(cpu_info)
        
        # 2. CUDA GPUs
        cuda_devices = self._get_cuda_devices()
        for device in cuda_devices:
            if self._validate_device(device):
                devices.append(device)
        
        # 3. Apple MPS
        mps_device = self._get_mps_device()
        if mps_device and self._validate_device(mps_device):
            devices.append(mps_device)
        
        # 4. Ranking
        devices = self._rank_devices(devices)
        
        # 5. Log resumo
        self._log_devices_summary(devices)
        
        return devices
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Obtém informações detalhadas da CPU."""
        device = torch.device("cpu")
        
        try:
            cpu_count = torch.get_num_threads()
            memory_gb = psutil.virtual_memory().total / 1024**3
            available_memory = psutil.virtual_memory().available / 1024**3
            
            info = DeviceInfo(
                device=device,
                name=f"CPU ({platform.machine()})",
                type="cpu",
                total_memory=memory_gb,
                available_memory=available_memory
            )
            
            # Validações
            if cpu_count < 4:
                info.warnings.append("Poucos threads disponíveis")
            if memory_gb < 8:
                info.warnings.append("Pouca memória RAM")
                
            return info
            
        except Exception as e:
            logger.warning(f"Erro ao obter info CPU: {e}")
            return DeviceInfo(
                device=device,
                name="CPU (Unknown)",
                type="cpu",
                is_available=True
            )
    
    def _get_cuda_devices(self) -> List[DeviceInfo]:
        """Obtém informações de devices CUDA."""
        devices = []
        
        if not torch.cuda.is_available():
            logger.info("🔍 CUDA não disponível")
            return devices
        
        try:
            device_count = torch.cuda.device_count()
            logger.info(f"🔍 {device_count} CUDA device(s) detectado(s)")
            
            for i in range(device_count):
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(i)
                
                total_memory = props.total_memory / 1024**3
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    available = total_memory - allocated
                except:
                    available = total_memory  # Fallback
                
                info = DeviceInfo(
                    device=device,
                    name=props.name,
                    type="cuda",
                    index=i,
                    total_memory=total_memory,
                    available_memory=available,
                    compute_capability=(props.major, props.minor)
                )
                
                # Validações CUDA
                if total_memory < self.min_gpu_memory_gb:
                    info.warnings.append(f"Pouca memória: {total_memory:.1f}GB")
                if props.major < 3:
                    info.limitations.append("Compute capability antiga")
                
                devices.append(info)
                
        except Exception as e:
            logger.warning(f"Erro ao detectar CUDA: {e}")
        
        return devices
    
    def _get_mps_device(self) -> Optional[DeviceInfo]:
        """Obtém informações do Metal Performance Shaders."""
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return None
        
        try:
            device = torch.device("mps")
            
            info = DeviceInfo(
                device=device,
                name="Apple Metal Performance Shaders",
                type="mps"
            )
            
            info.warnings.append("MPS é experimental - pode haver incompatibilidades")
            
            return info
            
        except Exception as e:
            logger.warning(f"Erro ao detectar MPS: {e}")
            return None
    
    def _validate_device(self, device_info: DeviceInfo) -> bool:
        """Valida se device funciona adequadamente."""
        cache_key = f"{device_info.type}_{device_info.index or 0}"
        
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        try:
            # Teste básico
            test_tensor = torch.randn(100, 100).to(device_info.device)
            result = test_tensor @ test_tensor
            del test_tensor, result
            
            # Cache resultado
            self._validation_cache[cache_key] = True
            device_info.is_available = True
            return True
            
        except Exception as e:
            device_info.warnings.append(f"Falha no teste: {str(e)[:50]}")
            device_info.is_available = False
            self._validation_cache[cache_key] = False
            return False
    
    def _benchmark_devices(self, devices: List[DeviceInfo]) -> List[DeviceInfo]:
        """Executa benchmark de performance nos devices."""
        logger.info("📊 Executando benchmarks...")
        
        for device_info in devices:
            if not device_info.is_available:
                continue
                
            try:
                start_time = time.time()
                
                # Benchmark: multiplicação de matrizes
                device = device_info.device
                a = torch.randn(1000, 1000).to(device)
                b = torch.randn(1000, 1000).to(device)
                
                # Múltiplas operações para média
                for _ in range(10):
                    c = torch.mm(a, b)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                device_info.benchmark_score = 1.0 / elapsed  # Higher is better
                
                logger.info(f"   {device_info.name}: {elapsed:.3f}s")
                
                del a, b, c
                
            except Exception as e:
                device_info.warnings.append(f"Benchmark falhou: {str(e)[:50]}")
                device_info.benchmark_score = 0.0
        
        return devices
    
    def _rank_devices(self, devices: List[DeviceInfo]) -> List[DeviceInfo]:
        """Rankeia devices por prioridade."""
        def device_score(device: DeviceInfo) -> tuple:
            # Prioridade: (is_available, prefer_gpu, benchmark_score, memory)
            gpu_bonus = 1.0 if device.type in ["cuda", "mps"] and self.prefer_gpu else 0.5
            benchmark = device.benchmark_score or 0.0
            memory = device.total_memory or 0.0
            available = 1.0 if device.is_available else 0.0
            
            return (available, gpu_bonus, benchmark, memory)
        
        sorted_devices = sorted(devices, key=device_score, reverse=True)
        
        # Marcar primeiro como recomendado se disponível
        if sorted_devices and sorted_devices[0].is_available:
            sorted_devices[0].is_recommended = True
        
        return sorted_devices
    
    def _select_device_by_requirement(self, devices: List[DeviceInfo], requirement: str) -> DeviceInfo:
        """Seleciona device baseado no requisito."""
        available_devices = [d for d in devices if d.is_available]
        
        if not available_devices:
            raise RuntimeError("Nenhum device válido disponível")
        
        if requirement == "cpu_only":
            cpu_devices = [d for d in available_devices if d.type == "cpu"]
            if not cpu_devices:
                raise RuntimeError("CPU não disponível")
            return cpu_devices[0]
        
        elif requirement == "gpu_only":
            gpu_devices = [d for d in available_devices if d.type in ["cuda", "mps"]]
            if not gpu_devices:
                raise RuntimeError("GPU não disponível")
            return gpu_devices[0]
        
        elif requirement == "fastest":
            if self.enable_benchmarking:
                benchmark_devices = [d for d in available_devices if d.benchmark_score]
                if benchmark_devices:
                    return max(benchmark_devices, key=lambda x: x.benchmark_score)
            return available_devices[0]
        
        else:  # auto
            recommended = [d for d in available_devices if d.is_recommended]
            return recommended[0] if recommended else available_devices[0]
    
    def _log_devices_summary(self, devices: List[DeviceInfo]):
        """Log resumo dos devices detectados."""
        available_count = sum(1 for d in devices if d.is_available)
        logger.info(f"🔍 Resumo de devices ({available_count} encontrados):")
        
        for i, device in enumerate(devices, 1):
            if device.is_available:
                logger.info(f"   {i}. {device.get_summary()}")
                for warning in device.warnings[:1]:  # Máximo 1 warning por device
                    logger.warning(f"      ⚠️  {warning}")
    
    def _log_device_warnings(self, device: DeviceInfo):
        """Log warnings do device selecionado."""
        for warning in device.warnings:
            logger.warning(f"⚠️  {warning}")
        for limitation in device.limitations:
            logger.info(f"💡 {limitation}")
    
    def _get_fallback_device(self) -> torch.device:
        """Retorna device de fallback (CPU)."""
        return torch.device("cpu")


# ==================== ALIASES PARA COMPATIBILIDADE ====================

class SimpleDeviceManager(DeviceManager):
    """Alias para modo simples."""
    def __init__(self, **kwargs):
        super().__init__(mode="simple", **kwargs)

class SmartDeviceManager(DeviceManager):
    """Alias para modo smart."""
    def __init__(self, **kwargs):
        super().__init__(mode="smart", **kwargs)

class ComplexDeviceManager(DeviceManager):
    """Alias para modo complex."""
    def __init__(self, **kwargs):
        super().__init__(mode="complex", enable_benchmarking=True, **kwargs)


# ==================== FUNÇÃO DE CONVENIÊNCIA ====================

def get_best_device(requirement: str = "auto", 
                   mode: str = "smart",
                   **kwargs) -> torch.device:
    """
    Função de conveniência para obter o melhor device disponível.
    
    Args:
        requirement: "auto", "gpu_only", "cpu_only", "fastest"
        mode: "simple", "smart", "complex"
        **kwargs: Argumentos para DeviceManager
        
    Returns:
        torch.device otimizado
        
    Example:
        device = get_best_device("auto", "smart")
        model = model.to(device)
    """
    manager = DeviceManager(mode=mode, **kwargs)
    return manager.get_device(requirement)
