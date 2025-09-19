"""
Classe base abstrata para todos os builders do sistema.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os

from .config import BuildConfig
from .exceptions import BuildException, ConfigurationError

class BaseBuilder(ABC):
    """Classe base abstrata para todos os construtores."""
    
    def __init__(self, config: Optional[BuildConfig] = None, **kwargs):
        """
        Inicializa o builder.
        
        Args:
            config: Configuração do sistema
            **kwargs: Argumentos adicionais de configuração
        """
        # Configuração
        if config is None:
            config = BuildConfig(**kwargs)
        elif kwargs:
            config.update(kwargs)
        
        self.config = config
        
        # Configurar logging
        self._setup_logging()
        
        # Validar configuração específica
        self._validate_config()
        
        # Estado interno
        self._initialized = False
        self._built = False
        
        self.logger.info(f"Inicializando {self.__class__.__name__}")
    
    def _setup_logging(self) -> None:
        """Configura sistema de logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config.get('log_format'))
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        self.logger.setLevel(level)
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Valida configuração específica do builder.
        Deve ser implementado por cada subclasse.
        """
        pass
    
    @abstractmethod
    def build(self) -> Any:
        """
        Executa a construção principal.
        Deve ser implementado por cada subclasse.
        
        Returns:
            Resultado da construção
        """
        pass
    
    def initialize(self) -> None:
        """Inicializa recursos necessários."""
        if self._initialized:
            return
        
        try:
            self._do_initialize()
            self._initialized = True
            self.logger.info("Inicialização concluída")
        except Exception as e:
            raise BuildException(f"Initialization error: {e}")
    
    def _do_initialize(self) -> None:
        """Implementação específica da inicialização."""
        # Criar diretórios necessários
        self.config.ensure_directories()
    
    def cleanup(self) -> None:
        """Limpa recursos utilizados."""
        try:
            self._do_cleanup()
            self.logger.info("Limpeza concluída")
        except Exception as e:
            self.logger.warning(f"Erro na limpeza: {e}")
    
    def _do_cleanup(self) -> None:
        """Implementação específica da limpeza."""
        pass
    
    def run(self) -> Any:
        """
        Executa o pipeline completo: inicializa, constrói e limpa.
        
        Returns:
            Resultado da construção
        """
        try:
            self.initialize()
            result = self.build()
            self._built = True
            return result
        finally:
            self.cleanup()
    
    def is_initialized(self) -> bool:
        """Verifica se foi inicializado."""
        return self._initialized
    
    def is_built(self) -> bool:
        """Verifica se foi construído."""
        return self._built
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Define valor de configuração."""
        self.config.set(key, value)
    
    def __enter__(self):
        """Context manager: entrada."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: saída."""
        self.cleanup()
    
    def __repr__(self) -> str:
        status = []
        if self._initialized:
            status.append("initialized")
        if self._built:
            status.append("built")
        
        status_str = ", ".join(status) if status else "not ready"
        return f"{self.__class__.__name__}({status_str})"
