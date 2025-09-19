"""
Sistema de logging unificado para o módulo build.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json

from build.core.constants import LOG_FORMAT, LOG_LEVEL

class BuildLogger:
    """Logger personalizado para o sistema build."""
    
    def __init__(self, 
                 name: str,
                 log_level: str = LOG_LEVEL,
                 log_format: str = LOG_FORMAT,
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Inicializa o logger.
        
        Args:
            name: Nome do logger
            log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Formato das mensagens de log
            log_file: Arquivo de log (opcional)
            max_file_size: Tamanho máximo do arquivo de log
            backup_count: Número de arquivos de backup
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Evitar duplicação de handlers
        if self.logger.handlers:
            return
        
        # Configurar nível
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        # File handler (se especificado)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    def log_system_info(self) -> None:
        """Log informações do sistema."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            self.info("=== INFORMAÇÕES DO SISTEMA ===")
            self.info(f"CPU: {cpu_count} cores")
            self.info(f"Memória: {memory.total / (1024**3):.1f}GB total, "
                     f"{memory.available / (1024**3):.1f}GB disponível")
            self.info(f"Python: {sys.version}")
        except ImportError:
            self.warning("psutil não disponível - informações do sistema limitadas")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuração do sistema."""
        self.info("=== CONFIGURAÇÃO ATIVA ===")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"  {sub_key}: {sub_value}")
            else:
                self.info(f"{key}: {value}")

class ProgressLogger:
    """Logger para progresso de operações longas."""
    
    def __init__(self, logger: Union[BuildLogger, logging.Logger], 
                 total: int, 
                 description: str = "Progresso",
                 log_interval: int = 100):
        """
        Inicializa logger de progresso.
        
        Args:
            logger: Logger a usar
            total: Total de itens a processar
            description: Descrição da operação
            log_interval: Intervalo para log de progresso
        """
        self.logger = logger if hasattr(logger, 'info') else logger.logger
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.processed = 0
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
    
    def update(self, increment: int = 1) -> None:
        """
        Atualiza progresso.
        
        Args:
            increment: Incremento do progresso
        """
        self.processed += increment
        
        # Log em intervalos regulares
        if self.processed % self.log_interval == 0 or self.processed == self.total:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log progresso atual."""
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        if self.processed > 0:
            rate = self.processed / elapsed.total_seconds()
            remaining = (self.total - self.processed) / rate if rate > 0 else 0
            eta = current_time.replace(microsecond=0) + \
                  datetime.timedelta(seconds=remaining)
        else:
            rate = 0
            eta = "N/A"
        
        percentage = (self.processed / self.total) * 100
        
        self.logger.info(
            f"{self.description}: {self.processed}/{self.total} "
            f"({percentage:.1f}%) - "
            f"{rate:.1f} items/s - "
            f"ETA: {eta}"
        )
        
        self.last_log_time = current_time
    
    def finish(self) -> None:
        """Finaliza e log estatísticas finais."""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        avg_rate = self.processed / total_time.total_seconds()
        
        self.logger.info(
            f"{self.description} CONCLUÍDO: {self.processed} items em "
            f"{total_time} (média: {avg_rate:.1f} items/s)"
        )

class MetricsLogger:
    """Logger para métricas e estatísticas."""
    
    def __init__(self, logger: Union[BuildLogger, logging.Logger]):
        """
        Inicializa logger de métricas.
        
        Args:
            logger: Logger base
        """
        self.logger = logger if hasattr(logger, 'info') else logger.logger
        self.metrics = {}
    
    def log_metric(self, name: str, value: Union[int, float, str], 
                   unit: Optional[str] = None) -> None:
        """
        Log métrica individual.
        
        Args:
            name: Nome da métrica
            value: Valor da métrica
            unit: Unidade (opcional)
        """
        self.metrics[name] = {'value': value, 'unit': unit, 'timestamp': datetime.now()}
        
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"MÉTRICA {name}: {value}{unit_str}")
    
    def log_metrics_summary(self) -> None:
        """Log resumo de todas as métricas."""
        if not self.metrics:
            return
        
        self.logger.info("=== RESUMO DE MÉTRICAS ===")
        for name, data in self.metrics.items():
            value = data['value']
            unit = data['unit']
            timestamp = data['timestamp'].strftime('%H:%M:%S')
            
            unit_str = f" {unit}" if unit else ""
            self.logger.info(f"  {name}: {value}{unit_str} ({timestamp})")
    
    def save_metrics(self, file_path: str) -> None:
        """
        Salva métricas em arquivo JSON.
        
        Args:
            file_path: Caminho do arquivo
        """
        # Converter datetime para string
        serializable_metrics = {}
        for name, data in self.metrics.items():
            serializable_metrics[name] = {
                'value': data['value'],
                'unit': data['unit'],
                'timestamp': data['timestamp'].isoformat()
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Métricas salvas em: {file_path}")

def setup_logging(name: str, 
                 log_level: str = LOG_LEVEL,
                 log_file: Optional[str] = None) -> BuildLogger:
    """
    Setup rápido de logging.
    
    Args:
        name: Nome do logger
        log_level: Nível de log
        log_file: Arquivo de log (opcional)
        
    Returns:
        Logger configurado
    """
    return BuildLogger(name, log_level=log_level, log_file=log_file)

def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger padrão do Python.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger padrão
    """
    return logging.getLogger(name)
