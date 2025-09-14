"""
Sistema robusto de validação e seleção de device (GPU/CPU).
Resolve problemas de falhas silenciosas e uso ineficiente de recursos.
"""

import torch
import subprocess
import platform
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

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
        return "Unknown"
    
    def get_capability_str(self) -> str:
        """Retorna compute capability formatada."""
        if self.compute_capability:
            return f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        return "Unknown"


class DeviceValidator:
    """
    Validador robusto de devices com fallback inteligente.
    
    Resolve problemas:
    - Falhas silenciosas quando GPU não funciona adequadamente
    - Uso de GPU inadequada (memória insuficiente)
    - Lack de verificação de drivers/CUDA
    - Seleção não otimizada de device
    """
    
    def __init__(self, min_gpu_memory_gb: float = 1.0, 
                 enable_benchmarking: bool = False,
                 prefer_gpu: bool = True):
        self.min_gpu_memory_gb = min_gpu_memory_gb
        self.enable_benchmarking = enable_benchmarking
        self.prefer_gpu = prefer_gpu
        
        # Cache para evitar recálculos
        self._device_cache: Dict[str, DeviceInfo] = {}
        self._validation_cache: Dict[str, bool] = {}
        
        logger.info(f"🔍 DeviceValidator inicializado")
        logger.info(f"   • Memória GPU mínima: {min_gpu_memory_gb:.1f}GB")
        logger.info(f"   • Benchmark ativo: {enable_benchmarking}")
    
    def detect_available_devices(self) -> List[DeviceInfo]:
        """
        Detecta todos os devices disponíveis com informações detalhadas.
        
        Returns:
            Lista de DeviceInfo ordenada por recomendação
        """
        logger.info("🔍 Detectando devices disponíveis...")
        
        devices = []
        
        # 1. CPU (sempre disponível)
        cpu_info = self._get_cpu_info()
        devices.append(cpu_info)
        
        # 2. CUDA GPUs
        cuda_devices = self._get_cuda_devices()
        devices.extend(cuda_devices)
        
        # 3. Apple Metal (MPS) 
        mps_device = self._get_mps_device()
        if mps_device:
            devices.append(mps_device)
        
        # 4. Validar e rankear devices
        validated_devices = []
        for device_info in devices:
            if self._validate_device(device_info):
                if self.enable_benchmarking:
                    device_info.benchmark_score = self._benchmark_device(device_info)
                validated_devices.append(device_info)
        
        # 5. Ordenar por recomendação
        validated_devices = self._rank_devices(validated_devices)
        
        # 6. Log resumo
        self._log_device_summary(validated_devices)
        
        return validated_devices
    
    def select_best_device(self, requirement: str = "auto") -> DeviceInfo:
        """
        Seleciona o melhor device baseado nos requisitos.
        
        Args:
            requirement: "auto", "gpu_only", "cpu_only", "fastest"
            
        Returns:
            DeviceInfo do melhor device
        """
        devices = self.detect_available_devices()
        
        if not devices:
            raise RuntimeError("❌ Nenhum device válido encontrado")
        
        if requirement == "cpu_only":
            cpu_devices = [d for d in devices if d.type == "cpu"]
            if not cpu_devices:
                raise RuntimeError("❌ CPU não disponível")
            selected = cpu_devices[0]
            
        elif requirement == "gpu_only":
            gpu_devices = [d for d in devices if d.type in ["cuda", "mps"]]
            if not gpu_devices:
                raise RuntimeError("❌ GPU não disponível ou adequada")
            selected = gpu_devices[0]
            
        elif requirement == "fastest":
            # Device com maior benchmark score
            benchmark_devices = [d for d in devices if d.benchmark_score is not None]
            if benchmark_devices:
                selected = max(benchmark_devices, key=lambda x: x.benchmark_score or 0)
            else:
                selected = devices[0]
                
        else:  # "auto"
            # Primeiro device recomendado
            recommended = [d for d in devices if d.is_recommended]
            selected = recommended[0] if recommended else devices[0]
        
        logger.info(f"✅ Device selecionado: {selected.name} ({selected.type})")
        if selected.warnings:
            for warning in selected.warnings:
                logger.warning(f"⚠️  {warning}")
        
        return selected
    
    def validate_device_compatibility(self, device: torch.device) -> Dict[str, Any]:
        """
        Valida compatibilidade de um device específico.
        
        Args:
            device: Device para validar
            
        Returns:
            Dict com resultado da validação
        """
        result = {
            "is_compatible": False,
            "device": device,
            "issues": [],
            "recommendations": []
        }
        
        try:
            if device.type == "cuda":
                result.update(self._validate_cuda_device(device))
            elif device.type == "cpu":
                result.update(self._validate_cpu_device(device))
            elif device.type == "mps":
                result.update(self._validate_mps_device(device))
            else:
                result["issues"].append(f"Tipo de device não suportado: {device.type}")
                
        except Exception as e:
            result["issues"].append(f"Erro na validação: {e}")
            
        result["is_compatible"] = len(result["issues"]) == 0
        return result
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Obtém informações da CPU."""
        import psutil
        
        device = torch.device("cpu")
        
        # Informações básicas
        cpu_count = torch.get_num_threads()
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        info = DeviceInfo(
            device=device,
            name=f"CPU ({platform.processor() or 'Unknown'})",
            type="cpu",
            total_memory=memory_gb,
            available_memory=psutil.virtual_memory().available / 1024**3
        )
        
        # CPU é sempre compatível, mas pode ter limitações
        if cpu_count < 4:
            info.warnings.append("Poucos threads disponíveis para CPU")
        if memory_gb < 8:
            info.warnings.append("Pouca memória RAM disponível")
        
        return info
    
    def _get_cuda_devices(self) -> List[DeviceInfo]:
        """Obtém informações de todos os devices CUDA."""
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
                
                # Memória
                total_memory = props.total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3  
                available = total_memory - allocated
                
                info = DeviceInfo(
                    device=device,
                    name=props.name,
                    type="cuda",
                    index=i,
                    total_memory=total_memory,
                    available_memory=available,
                    compute_capability=(props.major, props.minor)
                )
                
                # Validações específicas CUDA
                self._validate_cuda_capabilities(info, props)
                
                devices.append(info)
                
        except Exception as e:
            logger.warning(f"⚠️  Erro ao detectar CUDA devices: {e}")
        
        return devices
    
    def _get_mps_device(self) -> Optional[DeviceInfo]:
        """Obtém informações do Metal Performance Shaders (Apple Silicon)."""
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return None
        
        try:
            device = torch.device("mps")
            
            info = DeviceInfo(
                device=device,
                name="Apple Metal Performance Shaders",
                type="mps"
            )
            
            # MPS é relativamente novo, adicionar avisos
            info.warnings.append("MPS é experimental - pode haver incompatibilidades")
            
            return info
            
        except Exception as e:
            logger.warning(f"⚠️  Erro ao detectar MPS: {e}")
            return None
    
    def _validate_device(self, device_info: DeviceInfo) -> bool:
        """Valida se device é adequado."""
        cache_key = f"{device_info.type}_{device_info.index}"
        
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        is_valid = True
        
        try:
            # Teste básico: criar tensor pequeno
            test_tensor = torch.randn(10, 10).to(device_info.device)
            result = test_tensor @ test_tensor
            del test_tensor, result
            
            # Validações específicas por tipo
            if device_info.type == "cuda":
                is_valid = self._validate_cuda_specific(device_info)
            elif device_info.type == "mps":
                is_valid = self._validate_mps_specific(device_info)
                
        except Exception as e:
            is_valid = False
            device_info.warnings.append(f"Falha no teste básico: {e}")
            logger.warning(f"⚠️  Device {device_info.name} falhou teste básico: {e}")
        
        # Cache resultado
        self._validation_cache[cache_key] = is_valid
        device_info.is_available = is_valid
        
        return is_valid
    
    def _validate_cuda_capabilities(self, info: DeviceInfo, props):
        """Valida capacidades específicas CUDA."""
        # Memória mínima
        if info.total_memory and info.total_memory < self.min_gpu_memory_gb:
            info.warnings.append(f"Pouca memória GPU: {info.total_memory:.1f}GB < {self.min_gpu_memory_gb:.1f}GB")
        
        # Compute capability mínima (3.5 para funcionalidades modernas)
        if info.compute_capability and info.compute_capability[0] < 3:
            info.limitations.append("Compute capability antiga - algumas funcionalidades podem não funcionar")
        
        # Verificar driver
        try:
            driver_version = torch.version.cuda
            if driver_version is None:
                info.warnings.append("Versão do driver CUDA não detectada")
        except:
            info.warnings.append("Problema na detecção do driver CUDA")
    
    def _validate_cuda_specific(self, device_info: DeviceInfo) -> bool:
        """Validações específicas para CUDA."""
        try:
            # Teste de alocação de memória
            if device_info.available_memory and device_info.available_memory < 0.5:
                device_info.warnings.append("Pouca memória GPU disponível")
                return False
            
            # Teste operações básicas
            device = device_info.device
            a = torch.randn(1000, 1000).to(device)
            b = torch.randn(1000, 1000).to(device) 
            c = torch.mm(a, b)  # Matrix multiplication
            del a, b, c
            
            return True
            
        except Exception as e:
            device_info.warnings.append(f"Falha em teste CUDA: {e}")
            return False
    
    def _validate_mps_specific(self, device_info: DeviceInfo) -> bool:
        """Validações específicas para MPS."""
        try:
            # MPS tem limitações conhecidas
            device_info.limitations.append("Algumas operações podem não ser suportadas")
            return True
        except Exception as e:
            device_info.warnings.append(f"Falha em teste MPS: {e}")
            return False
    
    def _validate_cuda_device(self, device: torch.device) -> Dict[str, Any]:
        """Validação detalhada de device CUDA."""
        result = {"issues": [], "recommendations": []}
        
        try:
            # Verificar se CUDA está disponível
            if not torch.cuda.is_available():
                result["issues"].append("CUDA não está disponível no sistema")
                return result
            
            # Verificar device específico
            if device.index >= torch.cuda.device_count():
                result["issues"].append(f"Device CUDA {device.index} não existe")
                return result
            
            # Teste de memória
            props = torch.cuda.get_device_properties(device.index)
            total_memory_gb = props.total_memory / 1024**3
            
            if total_memory_gb < self.min_gpu_memory_gb:
                result["issues"].append(f"Memória insuficiente: {total_memory_gb:.1f}GB < {self.min_gpu_memory_gb:.1f}GB")
            
            # Compute capability
            if props.major < 3:
                result["issues"].append(f"Compute capability muito antiga: {props.major}.{props.minor}")
            
        except Exception as e:
            result["issues"].append(f"Erro na validação CUDA: {e}")
        
        return result
    
    def _validate_cpu_device(self, device: torch.device) -> Dict[str, Any]:
        """Validação de device CPU."""
        return {"issues": [], "recommendations": ["CPU é sempre compatível"]}
    
    def _validate_mps_device(self, device: torch.device) -> Dict[str, Any]:
        """Validação de device MPS."""
        result = {"issues": [], "recommendations": []}
        
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            result["issues"].append("MPS não está disponível")
        
        return result
    
    def _benchmark_device(self, device_info: DeviceInfo) -> float:
        """Faz benchmark simples do device."""
        if not device_info.is_available:
            return 0.0
        
        try:
            device = device_info.device
            
            # Benchmark: multiplicação de matrizes
            size = 1000
            iterations = 5
            
            times = []
            for _ in range(iterations):
                a = torch.randn(size, size).to(device)
                b = torch.randn(size, size).to(device)
                
                start_time = time.time()
                c = torch.mm(a, b)
                if device.type == "cuda":
                    torch.cuda.synchronize()  # Aguardar conclusão
                end_time = time.time()
                
                times.append(end_time - start_time)
                del a, b, c
            
            # Score = 1 / tempo médio (maior = melhor)
            avg_time = sum(times) / len(times)
            score = 1.0 / avg_time if avg_time > 0 else 0.0
            
            logger.debug(f"⚡ Benchmark {device_info.name}: {avg_time:.3f}s → score {score:.1f}")
            return score
            
        except Exception as e:
            logger.warning(f"⚠️  Erro no benchmark de {device_info.name}: {e}")
            return 0.0
    
    def _rank_devices(self, devices: List[DeviceInfo]) -> List[DeviceInfo]:
        """Ordena devices por recomendação."""
        def device_score(device: DeviceInfo) -> Tuple[int, float, float]:
            # Prioridade: (is_gpu, benchmark_score, memory)
            is_gpu = 1 if device.type in ["cuda", "mps"] else 0
            benchmark = device.benchmark_score or 0.0
            memory = device.total_memory or 0.0
            return (is_gpu, benchmark, memory)
        
        # Ordenar por score decrescente  
        devices.sort(key=device_score, reverse=True)
        
        # Marcar device recomendado
        if devices and self.prefer_gpu:
            # Primeiro GPU válido ou primeiro device
            for device in devices:
                if device.type in ["cuda", "mps"] and device.is_available:
                    device.is_recommended = True
                    break
            else:
                # Nenhum GPU, usar primeiro device
                devices[0].is_recommended = True
        elif devices:
            devices[0].is_recommended = True
        
        return devices
    
    def _log_device_summary(self, devices: List[DeviceInfo]):
        """Log resumo dos devices detectados."""
        logger.info(f"🔍 Resumo de devices ({len(devices)} encontrados):")
        
        for i, device in enumerate(devices):
            status = "✅ RECOMENDADO" if device.is_recommended else "💡 disponível"
            memory_str = f" | {device.get_memory_gb()}" if device.total_memory else ""
            
            logger.info(f"   {i+1}. {device.name} ({device.type}){memory_str} - {status}")
            
            for warning in device.warnings[:2]:  # Mostrar apenas 2 primeiros
                logger.warning(f"      ⚠️  {warning}")


class SmartDeviceManager:
    """
    Gerenciador inteligente de devices com fallback automático.
    
    Uso simplificado do DeviceValidator com fallback inteligente.
    """
    
    def __init__(self, **validator_kwargs):
        self.validator = DeviceValidator(**validator_kwargs)
        self._selected_device: Optional[DeviceInfo] = None
    
    def get_device(self, requirement: str = "auto") -> torch.device:
        """
        Obtém o melhor device disponível com fallback inteligente.
        
        Args:
            requirement: "auto", "gpu_only", "cpu_only", "fastest"
            
        Returns:
            torch.device otimizado
        """
        try:
            self._selected_device = self.validator.select_best_device(requirement)
            return self._selected_device.device
            
        except RuntimeError as e:
            logger.error(f"❌ {e}")
            
            # Fallback: tentar CPU
            if requirement != "cpu_only":
                logger.warning("🔄 Fallback para CPU...")
                try:
                    self._selected_device = self.validator.select_best_device("cpu_only")
                    return self._selected_device.device
                except:
                    pass
            
            # Último recurso
            logger.error("💀 Usando CPU como último recurso")
            return torch.device("cpu")
    
    def get_device_info(self) -> Optional[DeviceInfo]:
        """Retorna informações do device selecionado."""
        return self._selected_device
    
    def validate_current_device(self) -> bool:
        """Valida se o device atual ainda está funcionando."""
        if not self._selected_device:
            return False
        
        try:
            # Teste rápido
            test_tensor = torch.randn(10, 10).to(self._selected_device.device)
            result = test_tensor @ test_tensor
            del test_tensor, result
            return True
        except:
            return False
