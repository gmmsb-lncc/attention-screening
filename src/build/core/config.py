"""
Sistema de configuração para o módulo build.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from src.build.core.constants import *
from src.build.core.exceptions import ConfigurationError

class BuildConfig:
    """Classe para gerenciar configurações do sistema build."""
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_data: Dictionary with configuration data or file path
            **kwargs: Additional configuration options
        """
        self._config = self._load_default_config()
        
        # If config_data is a string, treat as file path
        if isinstance(config_data, (str, Path)):
            config_file = Path(config_data)
            if config_file.exists():
                self._load_from_file(config_file)
        # If config_data is a dict, merge it
        elif isinstance(config_data, dict):
            self._config.update(config_data)
        
        # Override with kwargs
        self._config.update(kwargs)
        
        # Converter device para use_gpu (para compatibilidade com IntegratedConfig)
        self._sync_device_to_use_gpu()
        
        # Sincronizar dimensões após carregar todas as configurações
        self._sync_model_dimensions()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Carrega configuração padrão."""
        return {
            # Diretórios
            'base_dir': DEFAULT_BASE_DIR,
            'ligand_dir': DEFAULT_LIGAND_DIR,
            'protein_dir': DEFAULT_PROTEIN_DIR,
            'ligand_output_dir': DEFAULT_LIGAND_OUTPUT_DIR,
            'protein_output_dir': DEFAULT_PROTEIN_OUTPUT_DIR,
            'matrix_output_dir': DEFAULT_MATRIX_OUTPUT_DIR,
            'concatenated_output_dir': DEFAULT_CONCATENATED_OUTPUT_DIR,
            
            # Diretórios de embeddings pré-existentes (reutilização)
            'protein_embeddings_dir': None,  # Se None, gera novos
            'ligand_embeddings_dir': None,   # Se None, gera novos (compartilhável entre experimentos)
            
            # Embeddings
            'embedding_type': DEFAULT_EMBEDDING_TYPE,
            'ligand_dim': DEFAULT_LIGAND_DIM,
            'protein_dim': DEFAULT_PROTEIN_DIM,
            'batch_size': DEFAULT_BATCH_SIZE,
            
            # Modelos
            'esm_model': 'esm2_t6_8M_UR50D',  # Modelo menor para testes rápidos
            'fm4m_model': 'SMI-TED',
            
            # Sistema
            'use_gpu': True,
            'use_parallel': True,
            
            # Spark
            'spark_config': SPARK_CONFIG.copy(),
            
            # Memória
            'memory_config': MEMORY_CONFIG.copy(),
            
            # Logging
            'log_level': LOG_LEVEL,
            'log_format': LOG_FORMAT,
            
            # Checkpoints
            'checkpoint_enabled': True,
            'checkpoint_file': CHECKPOINT_FILE,
            'processed_files_log': PROCESSED_FILES_LOG,
            
            # Validação
            'validation_enabled': True,
            'min_embedding_size': MIN_EMBEDDING_SIZE,
            'max_embedding_size': MAX_EMBEDDING_SIZE,
            
            # Estratificação
            'stratification_enabled': False,
            'stratification_params': {
                'clustering_algorithm': 'adaptive',
                'similarity_threshold': None,  # None = auto-threshold
                'cluster_min_size': 3,
                'stratify_by': 'both',
                'protein_weight': 0.6,
                'ligand_weight': 0.4,
                'adaptive_method': 'target'  # silhouette, elbow, target, percentile, leakage_aware
            }
        }
    
    def _load_from_file(self, config_file: str) -> None:
        """Carrega configuração de arquivo JSON."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            raise ConfigurationError(f"Erro ao carregar arquivo de configuração {config_file}: {e}")
    
    def _sync_device_to_use_gpu(self) -> None:
        """
        Converte parâmetro 'device' para 'use_gpu'.
        
        Esta conversão é necessária para compatibilidade com IntegratedConfig
        que usa 'device' (auto, cuda, mps, cpu) enquanto BuildConfig usa 'use_gpu' (bool).
        
        Mapping:
          - 'auto': Detecta automaticamente (use_gpu=True, deixa ProteinEmbedding decidir)
          - 'cuda': Força GPU CUDA (use_gpu=True)
          - 'mps': Força GPU MPS (use_gpu=True)
          - 'cpu': Força CPU (use_gpu=False)
        """
        device = self._config.get('device', 'auto')
        
        if device == 'cpu':
            self._config['use_gpu'] = False
        elif device in ('auto', 'cuda', 'mps'):
            self._config['use_gpu'] = True
        # Se não especificado, mantém use_gpu padrão (True)
    
    def _sync_model_dimensions(self) -> None:
        """
        Sincroniza dimensões com os modelos escolhidos.
        
        Se esm_model ou fm4m_model estão configurados, atualiza automaticamente
        protein_dim e ligand_dim para corresponder às dimensões corretas do modelo.
        
        Se esm_dim for explicitamente fornecido, ele tem precedência sobre a dimensão padrão do modelo.
        """
        # Sincronizar dimensão de proteína com modelo ESM
        esm_model = self._config.get('esm_model')
        esm_dim_override = self._config.get('esm_dim')  # Dimensão customizada via CLI
        
        if esm_dim_override is not None:
            # Usar dimensão customizada fornecida via --esm-dim
            self._config['protein_dim'] = esm_dim_override
        elif esm_model and esm_model in ESM_MODELS:
            # Usar dimensão padrão do modelo
            model_dim = ESM_MODELS[esm_model]['dim']
            current_dim = self._config.get('protein_dim')
            
            # Atualizar se diferente
            if current_dim != model_dim:
                self._config['protein_dim'] = model_dim
                # Log quando houver logger disponível
                # self.logger.info(f"Dimensão de proteína atualizada: {model_dim} (modelo: {esm_model})")
        
        # Sincronizar dimensão de ligante com modelo FM4M
        fm4m_model = self._config.get('fm4m_model')
        if fm4m_model and fm4m_model in FM4M_MODELS:
            model_dim = FM4M_MODELS[fm4m_model]['dim']
            current_dim = self._config.get('ligand_dim')
            
            # Atualizar se diferente
            if current_dim != model_dim:
                self._config['ligand_dim'] = model_dim
                # Log quando houver logger disponível
                # self.logger.info(f"Dimensão de ligante atualizada: {model_dim} (modelo: {fm4m_model})")
    
    def _validate_config(self) -> None:
        """Valida configuração."""
        # IMPORTANTE: Sincronizar dimensões ANTES de validar
        self._sync_model_dimensions()
        
        # Validar diretórios
        required_dirs = ['base_dir']
        for dir_key in required_dirs:
            if not self._config.get(dir_key):
                raise ConfigurationError(f"Diretório obrigatório não configurado: {dir_key}")
        
        # Validar modelos
        esm_model = self._config.get('esm_model')
        if esm_model and esm_model not in ESM_MODELS:
            raise ConfigurationError(f"Modelo ESM inválido: {esm_model}")
        
        fm4m_model = self._config.get('fm4m_model')
        if fm4m_model and fm4m_model not in FM4M_MODELS:
            raise ConfigurationError(f"Modelo FM4M inválido: {fm4m_model}")
        
        # Validar dimensões
        if self._config.get('ligand_dim', 0) <= 0:
            raise ConfigurationError("Dimensão de ligante deve ser positiva")
        
        if self._config.get('protein_dim', 0) <= 0:
            raise ConfigurationError("Dimensão de proteína deve ser positiva")
        
        # Validar batch size
        if self._config.get('batch_size', 0) <= 0:
            raise ConfigurationError("Batch size deve ser positivo")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Define valor de configuração."""
        self._config[key] = value
        # Re-sincronizar dimensões se modelo foi alterado
        if key in ('esm_model', 'fm4m_model'):
            self._sync_model_dimensions()
        self._validate_config()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Atualiza múltiplas configurações."""
        self._config.update(config_dict)
        # Re-sincronizar dimensões após update
        self._sync_model_dimensions()
        self._validate_config()
    
    def get_model_dimensions(self) -> Dict[str, int]:
        """
        Retorna as dimensões corretas baseadas nos modelos configurados.
        
        Returns:
            Dict com 'protein_dim', 'ligand_dim', e 'total_dim'
        """
        self._sync_model_dimensions()
        return {
            'protein_dim': self._config['protein_dim'],
            'ligand_dim': self._config['ligand_dim'],
            'total_dim': self._config['protein_dim'] + self._config['ligand_dim']
        }
    
    @classmethod
    def from_json(cls, config_file: str) -> 'BuildConfig':
        """Create BuildConfig from JSON file."""
        return cls(config_file)
    
    def to_dict(self) -> Dict[str, Any]:
        """Retorna configuração como dicionário."""
        return self._config.copy()
    
    def save(self, config_file: str) -> None:
        """Salva configuração em arquivo JSON."""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise ConfigurationError(f"Erro ao salvar configuração: {e}")
    
    def get_absolute_path(self, path_key: str) -> str:
        """Obtém caminho absoluto baseado na configuração."""
        relative_path = self.get(path_key)
        if not relative_path:
            raise ConfigurationError(f"Caminho não configurado: {path_key}")
        
        base_dir = self.get('base_dir')
        return os.path.abspath(os.path.join(base_dir, relative_path))
    
    def ensure_directories(self) -> None:
        """Garante que todos os diretórios necessários existam."""
        dir_keys = [
            'ligand_output_dir', 'protein_output_dir', 
            'matrix_output_dir', 'concatenated_output_dir'
        ]
        
        for dir_key in dir_keys:
            dir_path = self.get_absolute_path(dir_key)
            os.makedirs(dir_path, exist_ok=True)
    
    # Properties for direct access (backward compatibility)
    @property
    def ligand_dim(self) -> int:
        return self.get('ligand_dim', DEFAULT_LIGAND_DIM)
    
    @property
    def protein_dim(self) -> int:
        return self.get('protein_dim', DEFAULT_PROTEIN_DIM)
    
    @property
    def embedding_type(self) -> str:
        return self.get('embedding_type', DEFAULT_EMBEDDING_TYPE)
    
    @property
    def batch_size(self) -> int:
        return self.get('batch_size', DEFAULT_BATCH_SIZE)
    
    @property
    def use_gpu(self) -> bool:
        return self.get('use_gpu', True)
    
    @property
    def use_parallel(self) -> bool:
        return self.get('use_parallel', True)

    def __repr__(self) -> str:
        return f"BuildConfig({len(self._config)} configurações)"
