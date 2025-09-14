"""
Gerenciamento de memória para o DockTKinase Classifier.
Monitora e otimiza o uso de memória durante treinamento e inferência.
"""

import logging
import gc
import os
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Tentar importar torch se disponível
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch não disponível - funcionalidades GPU limitadas")


@dataclass
class MemorySnapshot:
    """Snapshot do uso de memória em um momento específico."""
    timestamp: datetime
    system_total_gb: float
    system_available_gb: float
    system_used_gb: float
    system_percent: float
    process_rss_gb: float
    process_vms_gb: float
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0


class MemoryTracker:
    """Rastreador contínuo de uso de memória."""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.snapshots: List[MemorySnapshot] = []
        self.tracking = False
        self._thread: Optional[threading.Thread] = None
        self.max_snapshots = 1000  # Limitar histórico
    
    def start_tracking(self):
        """Inicia rastreamento contínuo."""
        if self.tracking:
            return
        
        self.tracking = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        logger.info(f"📊 Rastreamento de memória iniciado (intervalo: {self.interval}s)")
    
    def stop_tracking(self):
        """Para rastreamento."""
        self.tracking = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("📊 Rastreamento de memória parado")
    
    def _track_loop(self):
        """Loop principal de rastreamento."""
        while self.tracking:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                
                # Limitar histórico
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)
                
                threading.Event().wait(self.interval)
            except Exception as e:
                logger.error(f"Erro no rastreamento: {e}")
    
    def _capture_snapshot(self) -> MemorySnapshot:
        """Captura snapshot atual."""
        # Memória do sistema
        system_mem = psutil.virtual_memory()
        
        # Memória do processo
        process = psutil.Process()
        process_mem = process.memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            system_total_gb=system_mem.total / 1024**3,
            system_available_gb=system_mem.available / 1024**3,
            system_used_gb=system_mem.used / 1024**3,
            system_percent=system_mem.percent,
            process_rss_gb=process_mem.rss / 1024**3,
            process_vms_gb=process_mem.vms / 1024**3
        )
        
        # Adicionar info GPU se disponível
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                snapshot.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
                snapshot.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                snapshot.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                pass
        
        return snapshot
    
    def get_current_usage(self) -> MemorySnapshot:
        """Retorna usage atual."""
        return self._capture_snapshot()
    
    def get_peak_usage(self, window_minutes: int = 10) -> Optional[MemorySnapshot]:
        """Retorna pico de uso em uma janela de tempo."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff]
        
        if not recent_snapshots:
            return None
        
        return max(recent_snapshots, key=lambda s: s.process_rss_gb)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        if not self.snapshots:
            return {}
        
        rss_values = [s.process_rss_gb for s in self.snapshots]
        gpu_values = [s.gpu_allocated_gb for s in self.snapshots if s.gpu_allocated_gb > 0]
        
        stats = {
            'snapshots_count': len(self.snapshots),
            'tracking_duration_minutes': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds() / 60,
            'process_memory': {
                'current_gb': rss_values[-1] if rss_values else 0,
                'peak_gb': max(rss_values) if rss_values else 0,
                'average_gb': sum(rss_values) / len(rss_values) if rss_values else 0
            }
        }
        
        if gpu_values:
            stats['gpu_memory'] = {
                'current_gb': gpu_values[-1],
                'peak_gb': max(gpu_values),
                'average_gb': sum(gpu_values) / len(gpu_values)
            }
        
        return stats


class MemoryManager:
    """
    Gerenciador principal de memória para o DockTKinase.
    Fornece monitoramento, otimização e controle de uso de memória.
    """
    
    def __init__(self,
                 memory_limit_gb: Optional[float] = None,
                 gpu_memory_fraction: float = 0.9,
                 auto_cleanup: bool = True,
                 track_usage: bool = True):
        """
        Inicializa o gerenciador de memória.
        
        Args:
            memory_limit_gb: Limite de memória em GB (None = automático)
            gpu_memory_fraction: Fração da GPU a usar
            auto_cleanup: Limpeza automática
            track_usage: Rastrear uso de memória
        """
        self.memory_limit_gb = memory_limit_gb
        self.gpu_memory_fraction = gpu_memory_fraction
        self.auto_cleanup = auto_cleanup
        
        # Determinar limite se não especificado
        if self.memory_limit_gb is None:
            system_memory = psutil.virtual_memory().total / 1024**3
            self.memory_limit_gb = system_memory * 0.8  # 80% da memória do sistema
        
        self.tracker = MemoryTracker() if track_usage else None
        self.cleanup_callbacks: List[Callable] = []
        
        # Configurar GPU se disponível
        self._setup_gpu_memory()
        
        logger.info("💾 MemoryManager inicializado")
        logger.info(f"   Limite de memória: {self.memory_limit_gb:.1f}GB")
        logger.info(f"   GPU memory fraction: {self.gpu_memory_fraction:.1%}")
        logger.info(f"   Auto cleanup: {self.auto_cleanup}")
        logger.info(f"   Tracking: {track_usage}")
    
    def _setup_gpu_memory(self):
        """Configura gestão de memória GPU."""
        if not TORCH_AVAILABLE:
            return
        
        try:
            if torch.cuda.is_available():
                # Configurar fração de memória
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                logger.info(f"🖥️  GPU memory configurada: {self.gpu_memory_fraction:.1%}")
            
            # Configurar para Apple MPS se disponível
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("🍎 Apple MPS detectado")
        
        except Exception as e:
            logger.warning(f"Erro ao configurar GPU: {e}")
    
    def start_monitoring(self):
        """Inicia monitoramento de memória."""
        if self.tracker:
            self.tracker.start_tracking()
    
    def stop_monitoring(self):
        """Para monitoramento de memória."""
        if self.tracker:
            self.tracker.stop_tracking()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Retorna informações detalhadas de uso de memória.
        
        Returns:
            Dicionário com informações de memória
        """
        # Memória do sistema
        system_mem = psutil.virtual_memory()
        
        # Memória do processo
        process = psutil.Process()
        process_mem = process.memory_info()
        
        usage = {
            'system': {
                'total_gb': system_mem.total / 1024**3,
                'available_gb': system_mem.available / 1024**3,
                'used_gb': system_mem.used / 1024**3,
                'percent': system_mem.percent
            },
            'process': {
                'rss_gb': process_mem.rss / 1024**3,
                'vms_gb': process_mem.vms / 1024**3,
                'percent_of_system': (process_mem.rss / system_mem.total) * 100
            },
            'limits': {
                'memory_limit_gb': self.memory_limit_gb,
                'approaching_limit': process_mem.rss / 1024**3 > self.memory_limit_gb * 0.8
            }
        }
        
        # Adicionar informações GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_info = {
                    'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                    'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                    'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
                }
                
                # Total da GPU
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    gpu_info['total_gb'] = props.total_memory / 1024**3
                    gpu_info['percent_used'] = (gpu_info['reserved_gb'] / gpu_info['total_gb']) * 100
                
                usage['gpu'] = gpu_info
            except Exception as e:
                logger.warning(f"Erro ao obter info GPU: {e}")
        
        # Adicionar estatísticas do tracker
        if self.tracker:
            usage['tracking'] = self.tracker.get_stats()
        
        return usage
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """
        Verifica pressão de memória atual.
        
        Returns:
            Dicionário com status de pressão de memória
        """
        usage = self.get_memory_usage()
        
        # Verificar pressão do sistema
        system_pressure = usage['system']['percent'] > 85
        
        # Verificar pressão do processo
        process_gb = usage['process']['rss_gb']
        process_pressure = process_gb > self.memory_limit_gb * 0.9
        
        # Verificar pressão GPU
        gpu_pressure = False
        if 'gpu' in usage:
            gpu_pressure = usage['gpu'].get('percent_used', 0) > 85
        
        status = {
            'system_pressure': system_pressure,
            'process_pressure': process_pressure,
            'gpu_pressure': gpu_pressure,
            'overall_pressure': system_pressure or process_pressure or gpu_pressure,
            'recommendations': []
        }
        
        # Gerar recomendações
        if system_pressure:
            status['recommendations'].append("Sistema com pouca memória - considere fechar outros programas")
        
        if process_pressure:
            status['recommendations'].append("Processo usando muita memória - executar limpeza")
        
        if gpu_pressure:
            status['recommendations'].append("GPU com pouca memória - reduzir batch size")
        
        return status
    
    def cleanup_memory(self, force_gpu: bool = False):
        """
        Executa limpeza de memória.
        
        Args:
            force_gpu: Forçar limpeza de cache GPU
        """
        logger.info("🧹 Executando limpeza de memória...")
        
        # Limpeza de cache Python
        gc.collect()
        
        # Executar callbacks registrados
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Erro em callback de limpeza: {e}")
        
        # Limpeza GPU
        if TORCH_AVAILABLE and (torch.cuda.is_available() or force_gpu):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("🖥️  Cache GPU limpo")
            except Exception as e:
                logger.warning(f"Erro na limpeza GPU: {e}")
        
        # Log do resultado
        usage_after = self.get_memory_usage()
        logger.info(f"✅ Limpeza concluída - Uso atual: {usage_after['process']['rss_gb']:.1f}GB")
    
    def set_memory_limit(self, limit_gb: float):
        """Define novo limite de memória."""
        self.memory_limit_gb = limit_gb
        logger.info(f"💾 Limite de memória atualizado: {limit_gb:.1f}GB")
    
    def register_cleanup_callback(self, callback: Callable):
        """Registra callback para limpeza automática."""
        self.cleanup_callbacks.append(callback)
        logger.info("🔧 Callback de limpeza registrado")
    
    @contextmanager
    def memory_context(self, cleanup_on_exit: bool = True):
        """
        Context manager para gestão automática de memória.
        
        Args:
            cleanup_on_exit: Executar limpeza ao sair do contexto
        """
        # Capturar estado inicial
        initial_usage = self.get_memory_usage()
        
        logger.info(f"🔄 Entrando em contexto de memória (inicial: {initial_usage['process']['rss_gb']:.1f}GB)")
        
        try:
            yield self
        finally:
            if cleanup_on_exit:
                self.cleanup_memory()
            
            # Log do estado final
            final_usage = self.get_memory_usage()
            logger.info(f"🔄 Saindo de contexto de memória (final: {final_usage['process']['rss_gb']:.1f}GB)")
    
    def optimize_for_large_data(self):
        """Otimiza configurações para datasets grandes."""
        logger.info("🔧 Otimizando para datasets grandes...")
        
        # Reduzir cache GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.7)  # Mais conservador
                torch.cuda.empty_cache()
                logger.info("🖥️  GPU otimizada para datasets grandes")
            except Exception:
                pass
        
        # Configurar garbage collection mais agressivo
        gc.set_threshold(100, 10, 10)  # Mais agressivo que padrão (700, 10, 10)
        
        logger.info("✅ Otimização para datasets grandes concluída")
    
    def get_recommendations(self) -> List[str]:
        """Retorna recomendações para otimização de memória."""
        usage = self.get_memory_usage()
        recommendations = []
        
        # Analisar uso atual
        process_gb = usage['process']['rss_gb']
        system_percent = usage['system']['percent']
        
        if process_gb > 4.0:
            recommendations.append("Considere usar lazy loading para datasets grandes")
        
        if system_percent > 80:
            recommendations.append("Sistema com pouca memória - feche outras aplicações")
        
        if 'gpu' in usage and usage['gpu'].get('percent_used', 0) > 80:
            recommendations.append("GPU com pouca memória - reduza batch size")
        
        if not recommendations:
            recommendations.append("Uso de memória está otimizado")
        
        return recommendations
    
    def __enter__(self):
        """Entrada do context manager."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Saída do context manager."""
        self.stop_monitoring()
        if self.auto_cleanup:
            self.cleanup_memory()


# Funções utilitárias
def get_system_memory_gb() -> float:
    """Retorna memória total do sistema em GB."""
    return psutil.virtual_memory().total / 1024**3


def get_available_memory_gb() -> float:
    """Retorna memória disponível em GB."""
    return psutil.virtual_memory().available / 1024**3


def get_gpu_memory_gb() -> float:
    """Retorna memória total da GPU em GB."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0.0
    
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / 1024**3
    except Exception:
        return 0.0


def format_memory_usage(usage_dict: Dict[str, Any]) -> str:
    """Formata informações de uso de memória para display."""
    lines = []
    
    # Sistema
    sys_info = usage_dict['system']
    lines.append(f"💻 Sistema: {sys_info['used_gb']:.1f}GB / {sys_info['total_gb']:.1f}GB ({sys_info['percent']:.1f}%)")
    
    # Processo
    proc_info = usage_dict['process']
    lines.append(f"🔧 Processo: {proc_info['rss_gb']:.1f}GB ({proc_info['percent_of_system']:.1f}% do sistema)")
    
    # GPU se disponível
    if 'gpu' in usage_dict:
        gpu_info = usage_dict['gpu']
        if 'total_gb' in gpu_info:
            lines.append(f"🖥️  GPU: {gpu_info['reserved_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB ({gpu_info['percent_used']:.1f}%)")
        else:
            lines.append(f"🖥️  GPU: {gpu_info['allocated_gb']:.1f}GB alocados")
    
    return "\n".join(lines)


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar gerenciador
    memory_mgr = MemoryManager()
    
    # Mostrar uso atual
    usage = memory_mgr.get_memory_usage()
    print("📊 Uso atual de memória:")
    print(format_memory_usage(usage))
    
    # Testar context manager
    with memory_mgr.memory_context():
        # Simular uso de memória
        data = [i for i in range(100000)]
        print(f"✅ Dados criados: {len(data)} elementos")
    
    print("✅ Exemplo concluído")
