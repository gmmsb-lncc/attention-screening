"""
Utilitários para gestão de memória e recursos do sistema.
"""

import gc
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
import time
from functools import wraps

from src.build.core.exceptions import BuildMemoryError

logger = logging.getLogger(__name__)

def get_memory_usage() -> Dict[str, float]:
    """
    Obtém informações de uso de memória do sistema.
    
    Returns:
        Dicionário com informações de memória
    """
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent,
        'free_gb': memory.free / (1024**3)
    }

def get_cpu_info() -> Dict[str, Any]:
    """
    Obtém informações de CPU.
    
    Returns:
        Dicionário com informações de CPU
    """
    return {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }

def get_disk_usage(path: str = '/') -> Dict[str, float]:
    """
    Obtém informações de uso de disco.
    
    Args:
        path: Caminho para verificar uso
        
    Returns:
        Dicionário com informações de disco
    """
    usage = psutil.disk_usage(path)
    return {
        'total_gb': usage.total / (1024**3),
        'used_gb': usage.used / (1024**3),
        'free_gb': usage.free / (1024**3),
        'percent': (usage.used / usage.total) * 100
    }

def check_memory_available(required_gb: float, 
                          safety_margin: float = 0.8) -> bool:
    """
    Verifica se há memória suficiente disponível.
    
    Args:
        required_gb: Memória requerida em GB
        safety_margin: Margem de segurança (0.0 a 1.0)
        
    Returns:
        True se há memória suficiente
    """
    memory_info = get_memory_usage()
    available_with_margin = memory_info['available_gb'] * safety_margin
    return available_with_margin >= required_gb

def check_disk_space(path: str, required_gb: float) -> bool:
    """
    Verifica se há espaço em disco suficiente.
    
    Args:
        path: Caminho para verificar
        required_gb: Espaço requerido em GB
        
    Returns:
        True se há espaço suficiente
    """
    disk_info = get_disk_usage(path)
    return disk_info['free_gb'] >= required_gb

def force_garbage_collection() -> int:
    """
    Força coleta de lixo e retorna objetos coletados.
    
    Returns:
        Número de objetos coletados
    """
    collected = 0
    for generation in range(3):
        collected += gc.collect(generation)
    
    return collected

def optimize_batch_size(base_batch_size: int,
                       memory_threshold: float = 0.7) -> int:
    """
    Otimiza batch size baseado na memória disponível.
    
    Args:
        base_batch_size: Batch size base
        memory_threshold: Threshold de memória para reduzir batch
        
    Returns:
        Batch size otimizado
    """
    memory_info = get_memory_usage()
    memory_usage_percent = memory_info['percent'] / 100
    
    if memory_usage_percent > memory_threshold:
        # Reduzir batch size proporcionalmente
        reduction_factor = (memory_usage_percent - memory_threshold) / (1 - memory_threshold)
        reduction_factor = min(reduction_factor, 0.75)  # Máximo 75% de redução
        
        optimized_batch_size = int(base_batch_size * (1 - reduction_factor))
        optimized_batch_size = max(optimized_batch_size, 1)  # Mínimo 1
        
        logger.warning(f"Memória alta ({memory_usage_percent:.1%}). "
                      f"Reduzindo batch size de {base_batch_size} para {optimized_batch_size}")
        
        return optimized_batch_size
    
    return base_batch_size

def memory_monitor(threshold_percent: float = 90.0):
    """
    Decorator para monitorar uso de memória de uma função.
    
    Args:
        threshold_percent: Threshold para emitir warning
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Memória antes da execução
            memory_before = get_memory_usage()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Memória após execução
                memory_after = get_memory_usage()
                execution_time = time.time() - start_time
                
                # Log de informações
                memory_delta = memory_after['used_gb'] - memory_before['used_gb']
                logger.info(f"{func.__name__} - Tempo: {execution_time:.2f}s, "
                           f"Memória: {memory_delta:+.2f}GB "
                           f"({memory_after['percent']:.1f}% total)")
                
                # Warning se memória alta
                if memory_after['percent'] > threshold_percent:
                    logger.warning(f"Uso de memória alto após {func.__name__}: "
                                 f"{memory_after['percent']:.1f}%")
                
                return result
                
            except Exception as e:
                # Memória em caso de erro
                memory_error = get_memory_usage()
                logger.error(f"Erro em {func.__name__} - Memória: {memory_error['percent']:.1f}%")
                raise
        
        return wrapper
    return decorator

def estimate_memory_usage(data_size: int, 
                         dtype_size: int = 8,
                         overhead_factor: float = 1.5) -> float:
    """
    Estima uso de memória para dados.
    
    Args:
        data_size: Tamanho dos dados (número de elementos)
        dtype_size: Tamanho do tipo de dados em bytes
        overhead_factor: Fator de overhead (Python + NumPy)
        
    Returns:
        Uso estimado em GB
    """
    base_memory = data_size * dtype_size
    total_memory = base_memory * overhead_factor
    return total_memory / (1024**3)

def wait_for_memory(required_gb: float, 
                   max_wait_time: float = 300,
                   check_interval: float = 5) -> bool:
    """
    Aguarda até que memória suficiente esteja disponível.
    
    Args:
        required_gb: Memória necessária em GB
        max_wait_time: Tempo máximo de espera em segundos
        check_interval: Intervalo entre verificações em segundos
        
    Returns:
        True se memória ficou disponível, False se timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if check_memory_available(required_gb):
            return True
        
        logger.info(f"Aguardando memória suficiente ({required_gb:.1f}GB)...")
        time.sleep(check_interval)
    
    logger.warning(f"Timeout aguardando memória após {max_wait_time}s")
    return False

class MemoryContext:
    """Context manager para monitoramento de memória."""
    
    def __init__(self, name: str = "Operation", 
                 cleanup: bool = True,
                 threshold_gb: Optional[float] = None):
        self.name = name
        self.cleanup = cleanup
        self.threshold_gb = threshold_gb
        self.start_memory = None
        self.start_time = None
    
    def __enter__(self):
        self.start_memory = get_memory_usage()
        self.start_time = time.time()
        
        if self.threshold_gb:
            if not check_memory_available(self.threshold_gb):
                raise BuildMemoryError(
                    f"Memória insuficiente para {self.name}. "
                    f"Requerido: {self.threshold_gb:.1f}GB"
                )
        
        logger.info(f"Iniciando {self.name} - "
                   f"Memória inicial: {self.start_memory['percent']:.1f}%")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = get_memory_usage()
        execution_time = time.time() - self.start_time
        
        memory_delta = end_memory['used_gb'] - self.start_memory['used_gb']
        
        logger.info(f"Finalizando {self.name} - "
                   f"Tempo: {execution_time:.2f}s, "
                   f"Memória: {memory_delta:+.2f}GB "
                   f"({end_memory['percent']:.1f}% total)")
        
        if self.cleanup:
            collected = force_garbage_collection()
            if collected > 0:
                logger.debug(f"Coletados {collected} objetos na limpeza")

def get_system_info() -> Dict[str, Any]:
    """
    Obtém informações completas do sistema.
    
    Returns:
        Dicionário com informações do sistema
    """
    return {
        'memory': get_memory_usage(),
        'cpu': get_cpu_info(),
        'disk': get_disk_usage(),
        'timestamp': time.time()
    }
