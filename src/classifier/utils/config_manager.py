"""
Sistema de Configuração Centralizada - Ponto 3.

Gerencia todas as configurações do sistema de forma unificada, com:
- Validação robusta de tipos e valores
- Templates e profiles predefinidos  
- Auto-configuração baseada em recursos
- Serialização em múltiplos formatos
- Versionamento e migração
"""

import json
try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from dataclasses import dataclass, asdict, fields, field
from abc import ABC, abstractmethod
import logging
from copy import deepcopy
from datetime import datetime
import warnings

# Imports locais
from ..config.mlp_config import MLPConfig
from ..core.trainer import TrainingConfig
from .device_manager import SmartDeviceManager

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configurações de dados e pré-processamento."""
    
    # Carregamento
    batch_size: Optional[int] = None  # None = automático
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # Validação
    max_nan_ratio: float = 0.1
    min_feature_variance: float = 1e-6
    outlier_method: str = "iqr"  # "iqr", "zscore", "none"
    outlier_threshold: float = 3.0
    
    # Divisão de dados
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify: bool = True
    shuffle: bool = True
    
    # Pré-processamento
    normalize_features: bool = True
    scale_method: str = "standard"  # "standard", "minmax", "robust", "none"
    handle_missing: str = "drop"  # "drop", "mean", "median", "mode"
    
    # Memory management
    lazy_loading: bool = True
    memory_efficient: bool = True
    max_samples_in_memory: int = 100000
    
    def __post_init__(self):
        # Validar ratios somam 1
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios devem somar 1.0, got {total_ratio}")


@dataclass  
class DeviceConfig:
    """Configurações de device e hardware."""
    
    # Seleção de device
    requirement: str = "auto"  # "auto", "gpu_only", "cpu_only", "fastest"
    enable_benchmarking: bool = False
    min_gpu_memory_gb: float = 1.0
    prefer_gpu: bool = True
    
    # Memory management
    empty_cache_frequency: int = 100  # A cada N batches
    max_gpu_memory_fraction: float = 0.9
    allow_growth: bool = True
    
    # Performance
    deterministic: bool = False  # Pode reduzir performance
    allow_tf32: bool = True  # NVIDIA Ampere+
    benchmark_cudnn: bool = True
    
    # Multi-GPU (futuro)
    use_data_parallel: bool = False
    use_distributed: bool = False


@dataclass
class LoggingConfig:
    """Configurações de logging e monitoramento."""
    
    # Levels
    level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Output
    log_to_file: bool = True
    log_to_console: bool = True
    log_file: str = "classifier.log"
    
    # Formatação
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Métricas e plots
    log_metrics: bool = True
    plot_training: bool = True
    save_plots: bool = True
    plot_format: str = "png"
    
    # Frequência
    log_frequency: int = 10  # A cada N epochs
    plot_frequency: int = 50
    
    # Avançado
    capture_warnings: bool = True
    log_gpu_memory: bool = True
    log_system_metrics: bool = False


@dataclass
class UnifiedConfig:
    """Configuração unificada do sistema completo."""
    
    # Componentes principais
    model: MLPConfig = field(default_factory=MLPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig) 
    data: DataConfig = field(default_factory=DataConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    name: str = "default"
    description: str = ""
    version: str = "1.0.0"
    created_at: Optional[datetime] = None
    
    # Tags para organização
    tags: List[str] = field(default_factory=list)
    profile: str = "development"  # "development", "production", "research"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ConfigValidator:
    """Validador robusto de configurações."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: UnifiedConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Valida configuração completa.
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validar cada componente
        self._validate_model_config(config.model)
        self._validate_training_config(config.training)
        self._validate_data_config(config.data)
        self._validate_device_config(config.device)
        self._validate_logging_config(config.logging)
        
        # Validações cruzadas
        self._validate_cross_dependencies(config)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors.copy(), self.warnings.copy()
    
    def _validate_model_config(self, config: MLPConfig):
        """Valida configuração do modelo."""
        if not config.hidden_layers or len(config.hidden_layers) == 0:
            self.errors.append("hidden_layers não pode estar vazio")
        
        if any(dim <= 0 for dim in config.hidden_layers):
            self.errors.append("Todas as dimensões devem ser positivas")
        
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            self.errors.append("dropout_rate deve estar em [0, 1)")
        
        if config.weight_decay < 0:
            self.errors.append("weight_decay deve ser não-negativo")
    
    def _validate_training_config(self, config: TrainingConfig):
        """Valida configuração de treinamento."""
        if config.max_epochs <= 0:
            self.errors.append("max_epochs deve ser positivo")
        
        if config.patience is not None and config.patience <= 0:
            self.errors.append("patience deve ser positivo")
        
        if config.max_epochs > 1000:
            self.warnings.append("max_epochs muito alto - considere early stopping")
    
    def _validate_data_config(self, config: DataConfig):
        """Valida configuração de dados."""
        if config.max_nan_ratio < 0 or config.max_nan_ratio > 1:
            self.errors.append("max_nan_ratio deve estar em [0, 1]")
        
        if config.outlier_threshold <= 0:
            self.errors.append("outlier_threshold deve ser positivo")
        
        if config.outlier_method not in ["iqr", "zscore", "none"]:
            self.errors.append("outlier_method inválido")
        
        if config.scale_method not in ["standard", "minmax", "robust", "none"]:
            self.errors.append("scale_method inválido")
        
        if config.handle_missing not in ["drop", "mean", "median", "mode"]:
            self.errors.append("handle_missing inválido")
    
    def _validate_device_config(self, config: DeviceConfig):
        """Valida configuração de device."""
        valid_requirements = ["auto", "gpu_only", "cpu_only", "fastest"]
        if config.requirement not in valid_requirements:
            self.errors.append(f"requirement deve ser um de {valid_requirements}")
        
        if config.min_gpu_memory_gb < 0:
            self.errors.append("min_gpu_memory_gb deve ser não-negativo")
        
        if config.max_gpu_memory_fraction <= 0 or config.max_gpu_memory_fraction > 1:
            self.errors.append("max_gpu_memory_fraction deve estar em (0, 1]")
    
    def _validate_logging_config(self, config: LoggingConfig):
        """Valida configuração de logging."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.level not in valid_levels:
            self.errors.append(f"level deve ser um de {valid_levels}")
        
        if config.console_level not in valid_levels:
            self.errors.append(f"console_level deve ser um de {valid_levels}")
        
        if config.file_level not in valid_levels:
            self.errors.append(f"file_level deve ser um de {valid_levels}")
        
        if config.log_frequency <= 0:
            self.errors.append("log_frequency deve ser positivo")
        
        if config.plot_frequency <= 0:
            self.errors.append("plot_frequency deve ser positivo")
    
    def _validate_cross_dependencies(self, config: UnifiedConfig):
        """Valida dependências entre diferentes configurações."""
        
        # Device vs Training  
        if config.device.requirement == "cpu_only" and config.training.amp_enabled:
            self.warnings.append("AMP pode não funcionar com CPU")
        
        # Memory vs Performance
        if (config.data.lazy_loading and 
            config.device.requirement == "gpu_only" and
            config.data.batch_size is not None and 
            config.data.batch_size > 1000):
            self.warnings.append("Lazy loading com GPU e batch grande pode reduzir performance")


class ConfigTemplateManager:
    """Gerencia templates e profiles de configuração."""
    
    def __init__(self):
        self.templates: Dict[str, UnifiedConfig] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Carrega templates predefinidos."""
        
        # Template de desenvolvimento
        dev_config = UnifiedConfig(
            name="development",
            description="Configuração para desenvolvimento e debug",
            profile="development",
            tags=["dev", "debug"],
            model=MLPConfig(
                hidden_layers=[128, 64, 32],
                dropout_rate=0.3,
                use_batch_norm=True
            ),
            training=TrainingConfig(
                max_epochs=50,
                patience=10
            ),
            data=DataConfig(
                batch_size=32,
                lazy_loading=False,  # Dados pequenos, carregar tudo
                memory_efficient=False
            ),
            device=DeviceConfig(
                requirement="auto",
                enable_benchmarking=False
            ),
            logging=LoggingConfig(
                level="DEBUG",
                log_to_console=True,
                log_to_file=True,
                plot_training=True
            )
        )
        
        # Template de produção
        prod_config = UnifiedConfig(
            name="production", 
            description="Configuração otimizada para produção",
            profile="production",
            tags=["prod", "optimized"],
            model=MLPConfig(
                hidden_layers=[256, 128, 64],
                dropout_rate=0.2,
                use_batch_norm=True,
                weight_decay=1e-5
            ),
            training=TrainingConfig(
                max_epochs=200,
                patience=20,
                amp_enabled=True
            ),
            data=DataConfig(
                batch_size=64,
                lazy_loading=True,
                memory_efficient=True,
                num_workers=8,
                persistent_workers=True
            ),
            device=DeviceConfig(
                requirement="auto",
                enable_benchmarking=True,
                min_gpu_memory_gb=2.0,
                benchmark_cudnn=True
            ),
            logging=LoggingConfig(
                level="INFO",
                log_to_console=True,
                log_to_file=True,
                log_gpu_memory=True
            )
        )
        
        # Template de pesquisa
        research_config = UnifiedConfig(
            name="research",
            description="Configuração para experimentos de pesquisa",
            profile="research", 
            tags=["research", "experiments"],
            model=MLPConfig(
                hidden_layers=[512, 256, 128, 64],
                dropout_rate=0.4,
                use_batch_norm=True,
                weight_decay=1e-4
            ),
            training=TrainingConfig(
                max_epochs=500,
                patience=50,
                amp_enabled=True,
                gradient_clip_value=1.0
            ),
            data=DataConfig(
                batch_size=128,
                lazy_loading=True,
                memory_efficient=True,
                outlier_method="iqr",
                scale_method="robust"
            ),
            device=DeviceConfig(
                requirement="fastest",
                enable_benchmarking=True,
                min_gpu_memory_gb=4.0
            ),
            logging=LoggingConfig(
                level="INFO",
                log_metrics=True,
                plot_training=True,
                save_plots=True,
                log_gpu_memory=True,
                log_system_metrics=True
            )
        )
        
        self.templates["development"] = dev_config
        self.templates["production"] = prod_config
        self.templates["research"] = research_config
    
    def get_template(self, name: str) -> Optional[UnifiedConfig]:
        """Obtém template por nome."""
        return deepcopy(self.templates.get(name))
    
    def list_templates(self) -> List[str]:
        """Lista templates disponíveis."""
        return list(self.templates.keys())
    
    def register_template(self, name: str, config: UnifiedConfig):
        """Registra novo template."""
        self.templates[name] = deepcopy(config)
        logger.info(f"Template '{name}' registrado")


class AutoConfigurator:
    """Sistema de auto-configuração baseado em dados e recursos."""
    
    def __init__(self, device_manager: Optional[SmartDeviceManager] = None):
        self.device_manager = device_manager or SmartDeviceManager()
    
    def auto_configure(self, 
                      base_config: UnifiedConfig,
                      n_samples: Optional[int] = None,
                      n_features: Optional[int] = None,
                      n_classes: Optional[int] = None,
                      available_memory_gb: Optional[float] = None) -> UnifiedConfig:
        """
        Auto-configura baseado em dados e recursos.
        
        Args:
            base_config: Configuração base
            n_samples: Número de amostras no dataset
            n_features: Número de features
            n_classes: Número de classes
            available_memory_gb: Memória disponível
            
        Returns:
            Configuração otimizada
        """
        config = deepcopy(base_config)
        
        logger.info("🔧 Iniciando auto-configuração...")
        
        # Auto-configurar device
        config = self._auto_configure_device(config)
        
        # Auto-configurar baseado nos dados
        if n_samples and n_features:
            config = self._auto_configure_data(config, n_samples, n_features, n_classes)
            config = self._auto_configure_model(config, n_samples, n_features, n_classes)
            config = self._auto_configure_training(config, n_samples)
        
        # Auto-configurar baseado na memória
        if available_memory_gb:
            config = self._auto_configure_memory(config, available_memory_gb)
        
        logger.info("✅ Auto-configuração concluída")
        return config
    
    def _auto_configure_device(self, config: UnifiedConfig) -> UnifiedConfig:
        """Auto-configura device baseado nos recursos disponíveis."""
        try:
            device = self.device_manager.get_device(config.device.requirement)
            device_info = self.device_manager.get_device_info()
            
            if device_info:
                # Ajustar configurações baseado no device
                if device_info.type == "cuda":
                    config.device.benchmark_cudnn = True
                    config.device.allow_tf32 = True
                    config.training.mixed_precision = True
                    
                elif device_info.type == "mps":
                    config.device.benchmark_cudnn = False  
                    config.device.allow_tf32 = False
                    config.training.mixed_precision = False  # MPS pode ter problemas
                    
                elif device_info.type == "cpu":
                    config.device.benchmark_cudnn = False
                    config.device.allow_tf32 = False
                    config.training.mixed_precision = False
                    config.data.num_workers = min(8, config.data.num_workers)
                
                logger.info(f"   🔧 Device auto-configurado para: {device_info.name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Erro na auto-configuração de device: {e}")
        
        return config
    
    def _auto_configure_data(self, config: UnifiedConfig, 
                           n_samples: int, n_features: int, 
                           n_classes: Optional[int]) -> UnifiedConfig:
        """Auto-configura dados baseado no tamanho do dataset."""
        
        # Batch size baseado no número de amostras
        if config.data.batch_size is None:
            if n_samples < 1000:
                batch_size = min(32, n_samples // 10)
            elif n_samples < 10000:
                batch_size = 64
            elif n_samples < 100000:
                batch_size = 128
            else:
                batch_size = 256
                
            config.data.batch_size = max(1, batch_size)
            
        # Lazy loading para datasets grandes
        if n_samples > config.data.max_samples_in_memory:
            config.data.lazy_loading = True
            config.data.memory_efficient = True
            
        # Workers baseado no tamanho
        if n_samples > 50000:
            config.data.num_workers = min(8, config.data.num_workers)
            config.data.persistent_workers = True
        
        logger.info(f"   📊 Dados auto-configurados: {n_samples} samples → batch_size={config.data.batch_size}")
        return config
    
    def _auto_configure_model(self, config: UnifiedConfig,
                            n_samples: int, n_features: int,
                            n_classes: Optional[int]) -> UnifiedConfig:
        """Auto-configura modelo baseado nos dados."""
        
        # Ajustar arquitetura baseado no número de features
        if n_features < 50:
            # Features poucas - modelo menor
            config.model.hidden_layers = [64, 32]
        elif n_features < 200:
            # Features médias
            config.model.hidden_layers = [128, 64, 32]
        elif n_features < 1000:
            # Muitas features
            config.model.hidden_layers = [256, 128, 64]
        else:
            # Features demais - modelo mais profundo
            config.model.hidden_layers = [512, 256, 128, 64]
        
        # Ajustar regularização baseado no tamanho do dataset
        if n_samples < 1000:
            # Dataset pequeno - mais regularização
            config.model.dropout_rate = min(0.5, config.model.dropout_rate + 0.1)
            config.model.weight_decay = max(1e-3, config.model.weight_decay)
        elif n_samples > 100000:
            # Dataset grande - menos regularização
            config.model.dropout_rate = max(0.1, config.model.dropout_rate - 0.1)
            config.model.weight_decay = min(1e-5, config.model.weight_decay)
        
        logger.info(f"   🧠 Modelo auto-configurado: {n_features} features → {config.model.hidden_layers}")
        return config
    
    def _auto_configure_training(self, config: UnifiedConfig, n_samples: int) -> UnifiedConfig:
        """Auto-configura treinamento baseado no tamanho do dataset."""
        
        # Learning rate baseado no tamanho do batch e dataset
        batch_size = config.data.batch_size or 32
        lr_scale = (batch_size / 32) ** 0.5  # Escala com raiz do batch size
        
        if n_samples < 1000:
            # Dataset pequeno - LR menor, mais epochs
            config.model.learning_rate *= 0.5 * lr_scale
            config.training.max_epochs = min(200, config.training.max_epochs * 2)
        elif n_samples > 100000:
            # Dataset grande - LR maior, menos epochs relativos
            config.model.learning_rate *= 1.5 * lr_scale
        else:
            config.model.learning_rate *= lr_scale
        
        # Early stopping baseado no tamanho
        steps_per_epoch = n_samples // (batch_size)
        if steps_per_epoch < 10:
            # Poucas steps por epoch - patience maior
            config.training.patience = max(config.training.patience or 10, 20)
        
        logger.info(f"   🎯 Treinamento auto-configurado: lr={config.model.learning_rate:.2e}")
        return config
    
    def _auto_configure_memory(self, config: UnifiedConfig, 
                             available_memory_gb: float) -> UnifiedConfig:
        """Auto-configura baseado na memória disponível."""
        
        # Ajustar batch size baseado na memória
        current_batch = config.data.batch_size or 32
        
        if available_memory_gb < 4:
            # Pouca memória - reduzir batch size
            new_batch = min(current_batch, 32)
            config.data.lazy_loading = True
            config.data.memory_efficient = True
        elif available_memory_gb > 16:
            # Muita memória - aumentar batch size
            new_batch = max(current_batch, 128)
            config.data.lazy_loading = False
            config.data.memory_efficient = False
        else:
            new_batch = current_batch
        
        config.data.batch_size = new_batch
        config.training.batch_size = new_batch
        
        logger.info(f"   💾 Memória auto-configurada: {available_memory_gb:.1f}GB → batch_size={new_batch}")
        return config


class ConfigManager:
    """Gerenciador centralizado de configurações."""
    
    def __init__(self, 
                 device_manager: Optional[SmartDeviceManager] = None,
                 auto_validate: bool = True):
        self.validator = ConfigValidator()
        self.template_manager = ConfigTemplateManager()
        self.auto_configurator = AutoConfigurator(device_manager)
        self.auto_validate = auto_validate
        
        # Configuração atual
        self._current_config: Optional[UnifiedConfig] = None
        
        logger.info("🔧 ConfigManager inicializado")
    
    def create_config(self, 
                     template: str = "development",
                     **overrides) -> UnifiedConfig:
        """
        Cria nova configuração baseada em template.
        
        Args:
            template: Nome do template base
            **overrides: Sobrescritas de configuração
            
        Returns:
            Configuração criada
        """
        # Obter template base
        base_config = self.template_manager.get_template(template)
        if base_config is None:
            logger.warning(f"Template '{template}' não encontrado, usando development")
            base_config = self.template_manager.get_template("development")
        
        # Aplicar overrides
        config = self._apply_overrides(base_config, overrides)
        
        # Validar se necessário
        if self.auto_validate:
            is_valid, errors, warnings = self.validator.validate_config(config)
            
            if not is_valid:
                raise ValueError(f"Configuração inválida: {errors}")
            
            for warning in warnings:
                logger.warning(f"⚠️  {warning}")
        
        self._current_config = config
        logger.info(f"✅ Configuração criada baseada no template '{template}'")
        
        return config
    
    def load_config(self, path: Union[str, Path]) -> UnifiedConfig:
        """
        Carrega configuração de arquivo.
        
        Args:
            path: Caminho para arquivo de configuração
            
        Returns:
            Configuração carregada
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
        
        # Detectar formato
        suffix = path.suffix.lower()
        
        try:
            if suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                if yaml is None:
                    raise ValueError("PyYAML não instalado - instale com: pip install pyyaml")
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            elif suffix == '.toml':
                if toml is None:
                    raise ValueError("toml não instalado - instale com: pip install toml")
                with open(path, 'r', encoding='utf-8') as f:
                    data = toml.load(f)
            else:
                raise ValueError(f"Formato não suportado: {suffix}")
            
            # Converter para UnifiedConfig
            config = self._dict_to_config(data)
            
            # Validar
            if self.auto_validate:
                is_valid, errors, warnings = self.validator.validate_config(config)
                
                if not is_valid:
                    raise ValueError(f"Configuração inválida: {errors}")
                
                for warning in warnings:
                    logger.warning(f"⚠️  {warning}")
            
            self._current_config = config
            logger.info(f"✅ Configuração carregada de: {path}")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar configuração: {e}")
            raise
    
    def save_config(self, 
                   config: UnifiedConfig, 
                   path: Union[str, Path],
                   format: str = "auto") -> None:
        """
        Salva configuração em arquivo.
        
        Args:
            config: Configuração para salvar
            path: Caminho do arquivo
            format: Formato ("json", "yaml", "toml", "auto")
        """
        path = Path(path)
        
        # Auto-detectar formato
        if format == "auto":
            format = path.suffix.lower().lstrip('.')
            if format not in ['json', 'yaml', 'yml', 'toml']:
                format = 'json'
        
        # Converter para dict
        data = self._config_to_dict(config)
        
        # Criar diretório se necessário
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'json':
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                    
            elif format in ['yaml', 'yml']:
                if yaml is None:
                    raise ValueError("PyYAML não instalado - instale com: pip install pyyaml")
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
                             
            elif format == 'toml':
                if toml is None:
                    raise ValueError("toml não instalado - instale com: pip install toml") 
                with open(path, 'w', encoding='utf-8') as f:
                    toml.dump(data, f)
                    
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"✅ Configuração salva em: {path}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar configuração: {e}")
            raise
    
    def auto_configure(self, 
                      template: str = "development",
                      n_samples: Optional[int] = None,
                      n_features: Optional[int] = None,
                      n_classes: Optional[int] = None,
                      available_memory_gb: Optional[float] = None,
                      **overrides) -> UnifiedConfig:
        """
        Cria configuração com auto-configuração baseada em dados.
        
        Returns:
            Configuração otimizada automaticamente
        """
        # Criar configuração base
        base_config = self.create_config(template, **overrides)
        
        # Auto-configurar
        optimized_config = self.auto_configurator.auto_configure(
            base_config=base_config,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            available_memory_gb=available_memory_gb
        )
        
        self._current_config = optimized_config
        return optimized_config
    
    def validate_config(self, config: UnifiedConfig) -> Tuple[bool, List[str], List[str]]:
        """Valida configuração e retorna resultado detalhado."""
        return self.validator.validate_config(config)
    
    def get_current_config(self) -> Optional[UnifiedConfig]:
        """Retorna configuração atual."""
        return self._current_config
    
    def list_templates(self) -> List[str]:
        """Lista templates disponíveis."""
        return self.template_manager.list_templates()
    
    def _apply_overrides(self, config: UnifiedConfig, overrides: Dict[str, Any]) -> UnifiedConfig:
        """Aplica sobrescritas na configuração."""
        if not overrides:
            return config
        
        config_dict = self._config_to_dict(config)
        
        # Aplicar overrides com dot notation
        for key, value in overrides.items():
            self._set_nested_value(config_dict, key, value)
        
        return self._dict_to_config(config_dict)
    
    def _set_nested_value(self, d: Dict[str, Any], key: str, value: Any):
        """Define valor aninhado usando dot notation (ex: 'model.hidden_dims')."""
        keys = key.split('.')
        current = d
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _config_to_dict(self, config: UnifiedConfig) -> Dict[str, Any]:
        """Converte UnifiedConfig para dict."""
        return asdict(config)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> UnifiedConfig:
        """Converte dict para UnifiedConfig."""
        # Converter componentes individuais
        components = {}
        
        if 'model' in data:
            components['model'] = MLPConfig.from_dict(data['model'])
        
        if 'training' in data:
            components['training'] = TrainingConfig(**data['training'])
        
        if 'data' in data:
            components['data'] = DataConfig(**data['data'])
        
        if 'device' in data:
            components['device'] = DeviceConfig(**data['device'])
        
        if 'logging' in data:
            components['logging'] = LoggingConfig(**data['logging'])
        
        # Metadados
        metadata = {k: v for k, v in data.items() 
                   if k not in ['model', 'training', 'data', 'device', 'logging']}
        
        # Converter datetime se necessário
        if 'created_at' in metadata and isinstance(metadata['created_at'], str):
            try:
                metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
            except:
                metadata['created_at'] = datetime.now()
        
        return UnifiedConfig(**components, **metadata)
