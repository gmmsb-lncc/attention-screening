#!/usr/bin/env python3
"""
Sistema de Logging - Regressão DockTKinase
===========================================

Sistema de logging estruturado para o pipeline de regressão.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class RegressionLogger:
    """
    Logger personalizado para pipeline de regressão.
    
    Features:
    - Logs para arquivo e console
    - Formatação colorida no console
    - Níveis de verbosidade
    - Timestamps automáticos
    """
    
    # Códigos ANSI para cores
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(
        self, 
        name: str = 'regression',
        log_file: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_colors: bool = True
    ):
        """
        Inicializa logger.
        
        Args:
            name: Nome do logger
            log_file: Path para arquivo de log (opcional)
            console_level: Nível de log para console
            file_level: Nível de log para arquivo
            use_colors: Usar cores no console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Limpar handlers existentes
        self.use_colors = use_colors
        
        # Formatter para console (mais simples)
        console_format = '%(levelname)s - %(message)s'
        if use_colors and sys.stdout.isatty():
            console_formatter = self.ColoredFormatter(console_format)
        else:
            console_formatter = logging.Formatter(console_format)
        
        # Formatter para arquivo (mais detalhado)
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (se especificado)
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    class ColoredFormatter(logging.Formatter):
        """Formatter que adiciona cores aos níveis de log."""
        
        def format(self, record):
            # Salvar levelname original
            orig_levelname = record.levelname
            
            # Adicionar cor
            if record.levelname in RegressionLogger.COLORS:
                color = RegressionLogger.COLORS[record.levelname]
                reset = RegressionLogger.COLORS['RESET']
                record.levelname = f'{color}{record.levelname}{reset}'
            
            # Formatar
            result = super().format(record)
            
            # Restaurar levelname
            record.levelname = orig_levelname
            
            return result
    
    def debug(self, msg: str):
        """Log de debug."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log de informação."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log de warning."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log de erro."""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log crítico."""
        self.logger.critical(msg)
    
    def section(self, title: str, symbol: str = '='):
        """
        Imprime uma seção destacada.
        
        Args:
            title: Título da seção
            symbol: Caractere para linha
        """
        line = symbol * 60
        self.info('')
        self.info(line)
        self.info(f'  {title}')
        self.info(line)
    
    def metrics(self, metrics_dict: dict, prefix: str = ''):
        """
        Imprime métricas formatadas.
        
        Args:
            metrics_dict: Dicionário com métricas
            prefix: Prefixo para cada linha
        """
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                self.info(f'{prefix}{key}: {value:.4f}')
            else:
                self.info(f'{prefix}{key}: {value}')
    
    def step(self, step_num: int, total_steps: int, description: str):
        """
        Log de progresso de etapa.
        
        Args:
            step_num: Número da etapa atual
            total_steps: Total de etapas
            description: Descrição da etapa
        """
        self.info(f'[{step_num}/{total_steps}] {description}')
    
    def success(self, msg: str):
        """Log de sucesso (usando INFO com emoji)."""
        self.info(f'✅ {msg}')
    
    def failure(self, msg: str):
        """Log de falha (usando ERROR com emoji)."""
        self.error(f'❌ {msg}')
    
    def model_training(self, model_name: str, status: str = 'start'):
        """
        Log específico para treinamento de modelo.
        
        Args:
            model_name: Nome do modelo
            status: 'start', 'end', 'error'
        """
        if status == 'start':
            self.info(f'🔄 Treinando {model_name}...')
        elif status == 'end':
            self.success(f'Modelo {model_name} treinado')
        elif status == 'error':
            self.failure(f'Erro ao treinar {model_name}')


def create_logger(
    log_dir: Optional[Path] = None,
    verbose: bool = True,
    name: str = 'regression'
) -> RegressionLogger:
    """
    Factory para criar logger configurado.
    
    Args:
        log_dir: Diretório para logs (None = apenas console)
        verbose: Se True, nível DEBUG. Se False, INFO
        name: Nome do logger
    
    Returns:
        RegressionLogger configurado
    """
    console_level = logging.DEBUG if verbose else logging.INFO
    
    log_file = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'
    
    return RegressionLogger(
        name=name,
        log_file=log_file,
        console_level=console_level,
        file_level=logging.DEBUG,
        use_colors=True
    )
