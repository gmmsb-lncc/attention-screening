"""
Sistema simplificado de configuração - Substitui o config_manager complexo.
Foca apenas nas funcionalidades essenciais.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
from pathlib import Path

# Imports com fallbacks
try:
    from ..classifier import MLPConfig
    from ..core.trainer import TrainingConfig
except ImportError:
    from classifier import MLPConfig
    try:
        from core.trainer import TrainingConfig
    except ImportError:
        # Usar definição local se necessário
        @dataclass
        class TrainingConfig:
            max_epochs: int = 100
            patience: int = 10


@dataclass
class SimpleConfig:
    """Configuração simplificada e funcional."""
    
    # Configurações essenciais
    model: MLPConfig
    training: TrainingConfig
    
    # Parâmetros de dados  
    batch_size: int = 64
    test_size: float = 0.2
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            'model': self.model.to_dict(),
            'training': {
                'max_epochs': self.training.max_epochs,
                'patience': self.training.patience
            },
            'batch_size': self.batch_size,
            'test_size': self.test_size,
            'device': self.device
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimpleConfig':
        """Cria configuração a partir de dicionário."""
        model_config = MLPConfig.from_dict(data.get('model', {}))
        training_config = TrainingConfig(
            max_epochs=data.get('training', {}).get('max_epochs', 100),
            patience=data.get('training', {}).get('patience', 10)
        )
        
        return cls(
            model=model_config,
            training=training_config,
            batch_size=data.get('batch_size', 64),
            test_size=data.get('test_size', 0.2),
            device=data.get('device', 'auto')
        )
    
    def save(self, path: str) -> None:
        """Salva configuração em arquivo JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleConfig':
        """Carrega configuração de arquivo JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def auto_configure(self, template: str = "development", 
                      n_samples: int = 1000, n_features: int = 50, 
                      n_classes: int = 2, available_memory: Optional[float] = None) -> 'SimpleConfig':
        """Auto-configuração baseada nos dados e recursos disponíveis."""
        
        # Configurar modelo baseado no tamanho dos dados
        hidden_layers = []
        if n_features <= 50:
            hidden_layers = [128, 64, 32]
        elif n_features <= 200:
            hidden_layers = [256, 128, 64]
        else:
            hidden_layers = [512, 256, 128]
        
        # Configurar batch_size baseado no número de amostras
        if n_samples < 500:
            batch_size = 32
        elif n_samples < 2000:
            batch_size = 64
        else:
            batch_size = 128
        
        # Configurar epochs baseado no template
        if template == "development":
            max_epochs = 50
            patience = 10
        elif template == "production":
            max_epochs = 200
            patience = 20
        else:  # research
            max_epochs = 500
            patience = 50
        
        # Criar nova configuração otimizada
        try:
            from ..classifier import MLPConfig as MLP
            optimized_model = MLP()
            # Configurar atributos ao invés de passar no construtor
            optimized_model.hidden_dim = hidden_layers[0] if hidden_layers else 256
            optimized_model.n_layers = len(hidden_layers) if hidden_layers else 3
            optimized_model.dropout = 0.3
        except ImportError:
            # Fallback simples
            optimized_model = self.model
        
        return SimpleConfig(
            model=optimized_model,
            training=TrainingConfig(max_epochs=max_epochs, patience=patience),
            batch_size=batch_size,
            test_size=self.test_size,
            device=self.device
        )


def create_default_config() -> SimpleConfig:
    """Cria configuração padrão otimizada."""
    from ..classifier import MLPConfig as create_mlp_config
    
    return SimpleConfig(
        model=create_mlp_config(),
        training=TrainingConfig(max_epochs=100, patience=10),
        batch_size=64,
        test_size=0.2,
        device="auto"
    )


# Manter compatibilidade com código existente
UnifiedConfig = SimpleConfig  # Alias para compatibilidade
ConfigManager = SimpleConfig  # Alias para compatibilidade
