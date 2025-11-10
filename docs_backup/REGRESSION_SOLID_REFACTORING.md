# 🔧 Análise SOLID e Refatoração - Módulo Regression

**Data**: 2025-11-09  
**Status**: 🔍 ANÁLISE COMPLETA  
**Objetivo**: Identificar violações SOLID e planejar refatoração

---

## 📋 Sumário Executivo

### Situação Atual
- **Arquivos duplicados**: 3 (trainer, evaluator, models)
- **Violações SOLID**: 8+ identificadas
- **Código duplicado**: ~40% entre raiz e core/
- **Testes**: 0 (zero!)
- **Acoplamento**: Alto (hardcoded dependencies)

### Alvo (após refatoração)
- **Arquivos duplicados**: 0
- **Compliance SOLID**: 100%
- **Código duplicado**: 0%
- **Testes**: 86+ (todos os 12 níveis)
- **Acoplamento**: Baixo (dependency injection)

---

## 🔍 Análise Detalhada de Violações SOLID

### 1. **SRP (Single Responsibility Principle)** ❌

#### Violação #1: RegressionPipeline (430 linhas)
```python
class RegressionPipeline:
    """Faz TUDO:"""
    def load_data()          # Responsabilidade 1: Carregar dados
    def train_models()       # Responsabilidade 2: Treinar modelos
    def evaluate_on_test()   # Responsabilidade 3: Avaliar
    def print_results_summary()  # Responsabilidade 4: Formatar saída
    def save_results()       # Responsabilidade 5: Salvar arquivos
```

**Problema**: Uma classe com 5 responsabilidades diferentes!

**Solução**: Separar em classes focadas
```python
# Cada classe com UMA responsabilidade
class DataPreparer:           # Apenas preparar dados
class ModelTrainer:           # Apenas treinar
class ResultsEvaluator:       # Apenas avaliar
class ResultsFormatter:       # Apenas formatar
class ResultsPersister:       # Apenas salvar

class RegressionPipeline:     # Facade - orquestra todos
    def __init__(self, preparer, trainer, evaluator, formatter, persister):
        self.preparer = preparer
        self.trainer = trainer
        # ... dependency injection
```

---

#### Violação #2: Arquivos Duplicados (SRP violado na estrutura)
```
src/regression/
├── trainer.py              ❌ Duplicado!
├── evaluator.py            ❌ Duplicado!
├── models.py               ❌ Duplicado!
└── core/
    ├── trainer.py          ❌ Duplicado!
    ├── evaluator.py        ❌ Duplicado!
    └── ...
```

**Problema**: Mesma responsabilidade em dois lugares!

**Solução**: Manter apenas core/
```
src/regression/
├── core/                   ✅ Único lugar
│   ├── trainer.py
│   ├── evaluator.py
│   └── data_loader.py
└── models/                 ✅ Único lugar
    └── models.py
```

---

### 2. **OCP (Open/Closed Principle)** ❌

#### Violação #3: Sem Base Abstrata
```python
# modular_pipeline.py
models = RegressionModels.get_all_models()  # Dict de modelos sklearn

# Sem interface comum!
# Para adicionar novo modelo = modificar RegressionModels
```

**Problema**: Não é aberto para extensão, fechado para modificação.

**Solução**: Criar classe base abstrata
```python
# models/base_model.py
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    """Interface comum para todos os regressores"""
    
    @abstractmethod
    def fit(self, X, y):
        """Treinar modelo"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Fazer predições"""
        pass
    
    @abstractmethod
    def get_params(self):
        """Obter hiperparâmetros"""
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """Obter importância das features (se aplicável)"""
        pass

# models/sklearn_wrapper.py
class SklearnRegressorWrapper(BaseRegressor):
    """Wrapper para modelos sklearn"""
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    # ... implementar interface

# Agora adicionar novo modelo = criar novo wrapper
# SEM modificar código existente! ✅
```

---

#### Violação #4: Configs Hardcoded
```python
# models/models.py
@staticmethod
def get_all_models(random_state=42, verbose=False):
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=100,        # Hardcoded!
            max_depth=None,          # Hardcoded!
            random_state=random_state
        ),
        # ...
    }
```

**Problema**: Para mudar config = modificar código!

**Solução**: Configs externas
```python
# config/model_configs.py
@dataclass
class RandomForestConfig:
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    # ... defaults

# models/models.py
def get_all_models(configs: Dict[str, Any]):
    """Recebe configs externas"""
    rf_config = configs.get('random_forest', RandomForestConfig())
    
    models = {
        'RandomForest': RandomForestRegressor(
            **asdict(rf_config)  # Usa config externa
        )
    }
```

---

### 3. **LSP (Liskov Substitution Principle)** ❌

#### Violação #5: Modelos Não Intercambiáveis
```python
# Alguns modelos têm feature_importance, outros não
rf_model = models['RandomForest']
print(rf_model.feature_importances_)  # ✅ Funciona

ridge_model = models['Ridge']
print(ridge_model.feature_importances_)  # ❌ AttributeError!
```

**Problema**: Subtipos não são substituíveis!

**Solução**: Interface consistente
```python
class BaseRegressor(ABC):
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Retorna importância ou None se não aplicável"""
        pass

class SklearnTreeWrapper(BaseRegressor):
    def get_feature_importance(self):
        return self.model.feature_importances_  # ✅ Existe

class SklearnLinearWrapper(BaseRegressor):
    def get_feature_importance(self):
        return self.model.coef_  # ✅ Usa coeficientes

class SklearnSVRWrapper(BaseRegressor):
    def get_feature_importance(self):
        return None  # ✅ SVR não tem importância
```

---

### 4. **ISP (Interface Segregation Principle)** ❌

#### Violação #6: Interface Gorda no Evaluator
```python
class RegressionEvaluator:
    @staticmethod
    def calculate_metrics(...)      # Métricas
    @staticmethod
    def compare_models(...)         # Comparação
    @staticmethod
    def get_best_model(...)         # Seleção
    @staticmethod
    def save_predictions_csv(...)   # I/O!
    @staticmethod
    def print_metrics_summary(...)  # Formatação!
```

**Problema**: Clientes forçados a depender de métodos que não usam!

**Solução**: Segregar em interfaces específicas
```python
# Separar responsabilidades

class MetricsCalculator:
    """Apenas calcular métricas"""
    def calculate_regression_metrics(...)
    def calculate_residuals(...)

class ModelComparator:
    """Apenas comparar modelos"""
    def compare_models(...)
    def rank_by_metric(...)
    def get_best_model(...)

class PredictionsExporter:
    """Apenas exportar predições"""
    def save_to_csv(...)
    def save_to_json(...)

class MetricsFormatter:
    """Apenas formatar saída"""
    def format_metrics_table(...)
    def print_summary(...)
```

---

### 5. **DIP (Dependency Inversion Principle)** ❌

#### Violação #7: Dependências Concretas
```python
class RegressionPipeline:
    def __init__(self, ...):
        # Cria dependências internamente - violação DIP!
        self.data_manager = DataManager(...)          # Hardcoded
        self.metrics_calculator = MetricsCalculator() # Hardcoded
        self.evaluator = RegressionEvaluator()        # Hardcoded
```

**Problema**: Depende de classes concretas, não abstrações!

**Solução**: Dependency Injection
```python
class RegressionPipeline:
    def __init__(
        self,
        data_preparer: IDataPreparer,         # Abstração
        model_trainer: IModelTrainer,         # Abstração
        results_evaluator: IResultsEvaluator, # Abstração
        verbose: bool = True
    ):
        """Injeta dependências - não as cria!"""
        self.data_preparer = data_preparer
        self.trainer = model_trainer
        self.evaluator = results_evaluator
        self.verbose = verbose

# Factory para criar com defaults
class RegressionPipelineFactory:
    @staticmethod
    def create_default_pipeline(...):
        """Factory cria componentes padrão"""
        data_preparer = DataPreparer(...)
        trainer = ModelTrainer(...)
        evaluator = ResultsEvaluator(...)
        
        return RegressionPipeline(
            data_preparer,
            trainer,
            evaluator
        )
```

---

#### Violação #8: Import Concreto
```python
# modular_pipeline.py
try:
    from .core import RegressionEvaluator  # Classe concreta
    from .utils import MetricsCalculator   # Classe concreta
except ImportError:
    from core import RegressionEvaluator
    from utils import MetricsCalculator
```

**Problema**: Importa implementações, não interfaces!

**Solução**: Imports de abstrações
```python
# interfaces/regression_interfaces.py
from abc import ABC, abstractmethod

class IDataPreparer(ABC):
    @abstractmethod
    def load_data(self): pass

class IModelTrainer(ABC):
    @abstractmethod
    def train(self, X, y): pass

# modular_pipeline.py
from .interfaces import IDataPreparer, IModelTrainer

class RegressionPipeline:
    def __init__(
        self,
        data_preparer: IDataPreparer,  # Interface!
        trainer: IModelTrainer          # Interface!
    ):
        pass
```

---

## 🎯 Plano de Refatoração (Fase a Fase)

### **Fase 1: Limpeza (Dia 1, ~2h)** 🧹

#### 1.1 Remover Duplicados
```bash
# Decisão: Manter core/, remover raiz

# Backup primeiro
git checkout -b refactor/solid-regression
git add .
git commit -m "Checkpoint antes de refatoração SOLID"

# Remover duplicados
rm src/regression/trainer.py
rm src/regression/evaluator.py  
rm src/regression/models.py

# Atualizar imports em arquivos que usam eles
# modular_pipeline.py, modular_regression.py, etc.
```

#### 1.2 Verificar Dependências
```bash
# Buscar todos os imports dos arquivos removidos
cd src/regression
grep -r "from .trainer import" .
grep -r "from .evaluator import" .
grep -r "from .models import" .

# Atualizar para:
# from .core.trainer import RegressionTrainer
# from .core.evaluator import RegressionEvaluator
# from .models.models import RegressionModels
```

#### 1.3 Teste de Sanidade
```python
# Verificar que ainda importa
python -c "from regression.modular_pipeline import RegressionPipeline; print('✅ OK')"
```

**Commit**: `refactor: remove duplicate files (trainer, evaluator, models)`

---

### **Fase 2: Criar Abstrações (Dia 1-2, ~4h)** 🏗️

#### 2.1 Criar Interfaces
```python
# src/regression/interfaces/__init__.py
from .data_interfaces import IDataPreparer, IDataValidator
from .model_interfaces import IRegressor, IModelTrainer
from .evaluation_interfaces import IMetricsCalculator, IResultsEvaluator

__all__ = [
    'IDataPreparer', 'IDataValidator',
    'IRegressor', 'IModelTrainer',
    'IMetricsCalculator', 'IResultsEvaluator'
]
```

```python
# src/regression/interfaces/model_interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class IRegressor(ABC):
    """Interface base para todos os regressores"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'IRegressor':
        """Treinar modelo"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fazer predições"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Obter hiperparâmetros"""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'IRegressor':
        """Definir hiperparâmetros"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Obter importância das features (None se não aplicável)"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do modelo"""
        pass


class IModelTrainer(ABC):
    """Interface para treinadores de modelos"""
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Treinar modelo(s)"""
        pass
    
    @abstractmethod
    def get_trained_models(self) -> Dict[str, IRegressor]:
        """Obter modelos treinados"""
        pass
    
    @abstractmethod
    def get_best_model(self, metric: str = 'MAE') -> IRegressor:
        """Obter melhor modelo"""
        pass
```

```python
# src/regression/interfaces/data_interfaces.py
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class IDataPreparer(ABC):
    """Interface para preparação de dados"""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carregar embeddings e targets"""
        pass
    
    @abstractmethod
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """Dividir em train/val/test"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas dos dados"""
        pass


class IDataValidator(ABC):
    """Interface para validação de dados"""
    
    @abstractmethod
    def validate_shapes(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validar compatibilidade de shapes"""
        pass
    
    @abstractmethod
    def check_missing_values(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Verificar NaN/Inf"""
        pass
```

```python
# src/regression/interfaces/evaluation_interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class IMetricsCalculator(ABC):
    """Interface para cálculo de métricas"""
    
    @abstractmethod
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcular todas as métricas de regressão"""
        pass


class IResultsEvaluator(ABC):
    """Interface para avaliação de resultados"""
    
    @abstractmethod
    def evaluate_model(
        self,
        model: 'IRegressor',
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Avaliar um modelo"""
        pass
    
    @abstractmethod
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'MAE'
    ) -> Dict[str, Any]:
        """Comparar múltiplos modelos"""
        pass
```

**Commit**: `refactor: add SOLID interfaces (OCP, DIP)`

---

#### 2.2 Criar BaseRegressor
```python
# src/regression/models/base_regressor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class BaseRegressor(ABC):
    """
    Classe base abstrata para todos os regressores.
    
    Implementa Template Method pattern para treinamento padrão.
    Subclasses implementam métodos específicos.
    """
    
    def __init__(self, name: str, random_state: int = 42):
        self._name = name
        self.random_state = random_state
        self._is_fitted = False
    
    @property
    def name(self) -> str:
        """Nome do modelo"""
        return self._name
    
    @property
    def is_fitted(self) -> bool:
        """Se o modelo está treinado"""
        return self._is_fitted
    
    # Template Method
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegressor':
        """
        Template method para treinamento.
        
        Hook methods: _validate_data, _fit_impl, _post_fit
        """
        # Hook: validar dados
        self._validate_data(X, y)
        
        # Hook: treinar implementação
        self._fit_impl(X, y)
        
        # Marcar como treinado
        self._is_fitted = True
        
        # Hook: pós-processamento
        self._post_fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fazer predições (verifica se está treinado)"""
        if not self._is_fitted:
            raise RuntimeError(f'{self.name} não foi treinado ainda!')
        
        return self._predict_impl(X)
    
    # Hooks para subclasses implementarem
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Hook: validar dados antes de treinar"""
        if len(X) != len(y):
            raise ValueError(f'X e y têm tamanhos diferentes: {len(X)} vs {len(y)}')
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError('Dados contêm NaN!')
    
    @abstractmethod
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Hook: implementação específica do treinamento"""
        pass
    
    @abstractmethod
    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Hook: implementação específica da predição"""
        pass
    
    def _post_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Hook: pós-processamento (opcional)"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Obter hiperparâmetros"""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'BaseRegressor':
        """Definir hiperparâmetros"""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Obter importância das features.
        
        Default: None (modelos sem importância).
        Subclasses sobrescrevem se aplicável.
        """
        return None
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name="{self.name}")'
```

**Commit**: `refactor: add BaseRegressor with Template Method pattern`

---

#### 2.3 Criar Wrappers para Sklearn
```python
# src/regression/models/sklearn_wrappers.py
from typing import Any, Dict, Optional
import numpy as np
from .base_regressor import BaseRegressor

class SklearnRegressorWrapper(BaseRegressor):
    """Wrapper genérico para modelos sklearn"""
    
    def __init__(self, sklearn_model, name: str, random_state: int = 42):
        super().__init__(name, random_state)
        self.model = sklearn_model
    
    def _fit_impl(self, X: np.ndarray, y: np.ndarray) -> None:
        """Delega para modelo sklearn"""
        self.model.fit(X, y)
    
    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Delega para modelo sklearn"""
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Obter params do modelo sklearn"""
        return self.model.get_params()
    
    def set_params(self, **params) -> 'SklearnRegressorWrapper':
        """Definir params no modelo sklearn"""
        self.model.set_params(**params)
        return self


class SklearnTreeRegressorWrapper(SklearnRegressorWrapper):
    """Wrapper para modelos baseados em árvores (RF, GB)"""
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Tree models têm feature_importances_"""
        if self.is_fitted:
            return self.model.feature_importances_
        return None


class SklearnLinearRegressorWrapper(SklearnRegressorWrapper):
    """Wrapper para modelos lineares (Ridge, Lasso)"""
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Linear models usam coeficientes como importância"""
        if self.is_fitted:
            return np.abs(self.model.coef_)
        return None


class SklearnNeuralNetworkWrapper(SklearnRegressorWrapper):
    """Wrapper para MLPRegressor"""
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """MLP não tem importância interpretável"""
        return None
```

**Commit**: `refactor: add sklearn wrappers implementing BaseRegressor`

---

### **Fase 3: Refatorar Classes (Dia 2-3, ~6h)** 🔨

#### 3.1 Refatorar RegressionPipeline (SRP)
```python
# src/regression/pipeline/__init__.py
from .data_preparer import DataPreparer
from .model_trainer import ModelTrainer
from .results_evaluator import ResultsEvaluator
from .results_formatter import ResultsFormatter
from .results_persister import ResultsPersister
from .regression_pipeline import RegressionPipeline

__all__ = [
    'DataPreparer',
    'ModelTrainer',
    'ResultsEvaluator',
    'ResultsFormatter',
    'ResultsPersister',
    'RegressionPipeline'
]
```

```python
# src/regression/pipeline/data_preparer.py
from typing import Tuple
import numpy as np
from ..interfaces import IDataPreparer
from ..core.data_loader import DataManager

class DataPreparer(IDataPreparer):
    """
    SRP: Responsável APENAS por preparar dados.
    """
    
    def __init__(
        self,
        embeddings_path: str,
        targets_path: str,
        verbose: bool = True
    ):
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.verbose = verbose
        self._data_manager = DataManager(embeddings_path, targets_path)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carregar embeddings e targets"""
        if self.verbose:
            print('📊 Carregando dados...')
        
        X, y = self._data_manager.load_data()
        
        if self.verbose:
            print(f'   ✅ X: {X.shape}, y: {y.shape}')
        
        return X, y
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """Dividir em train/val/test"""
        return self._data_manager.split_data(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas dos dados"""
        return self._data_manager.get_stats()
```

```python
# src/regression/pipeline/model_trainer.py
from typing import Dict, Any, Optional
import numpy as np
from ..interfaces import IModelTrainer, IRegressor
from ..models import RegressionModels

class ModelTrainer(IModelTrainer):
    """
    SRP: Responsável APENAS por treinar modelos.
    """
    
    def __init__(
        self,
        models_dict: Optional[Dict[str, IRegressor]] = None,
        verbose: bool = True,
        random_state: int = 42
    ):
        self.verbose = verbose
        self.random_state = random_state
        
        # Se não fornecido, criar todos os modelos
        if models_dict is None:
            models_dict = RegressionModels.get_all_models(random_state)
        
        self.models = models_dict
        self.trained_models = {}
        self.training_times = {}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Treinar todos os modelos"""
        if self.verbose:
            print(f'🤖 Treinando {len(self.models)} modelos...')
        
        for name, model in self.models.items():
            if self.verbose:
                print(f'   🔄 {name}...')
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.trained_models[name] = model
            self.training_times[name] = training_time
            
            if self.verbose:
                print(f'      ✅ {training_time:.2f}s')
        
        return self.trained_models
    
    def get_trained_models(self) -> Dict[str, IRegressor]:
        """Obter modelos treinados"""
        return self.trained_models
    
    def get_best_model(self, metric: str = 'MAE') -> IRegressor:
        """Obter melhor modelo (precisa de avaliação primeiro!)"""
        # Implementar lógica de seleção
        pass
```

```python
# src/regression/pipeline/results_evaluator.py
from typing import Dict, Any
import numpy as np
from ..interfaces import IResultsEvaluator, IRegressor, IMetricsCalculator
from ..utils.metrics import MetricsCalculator

class ResultsEvaluator(IResultsEvaluator):
    """
    SRP: Responsável APENAS por avaliar resultados.
    """
    
    def __init__(
        self,
        metrics_calculator: Optional[IMetricsCalculator] = None,
        verbose: bool = True
    ):
        self.verbose = verbose
        
        # DIP: Injeta calculator (ou usa default)
        if metrics_calculator is None:
            metrics_calculator = MetricsCalculator()
        
        self.metrics_calculator = metrics_calculator
    
    def evaluate_model(
        self,
        model: IRegressor,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Avaliar um modelo"""
        y_pred = model.predict(X)
        
        metrics = self.metrics_calculator.calculate_regression_metrics(
            y, y_pred
        )
        
        return {
            'model_name': model.name,
            'metrics': metrics,
            'predictions': y_pred
        }
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'MAE'
    ) -> Dict[str, Any]:
        """Comparar múltiplos modelos"""
        # Extrair métrica de cada modelo
        model_metrics = {
            name: res['metrics'][metric]
            for name, res in results.items()
        }
        
        # Ordenar (menor é melhor para MAE)
        sorted_models = sorted(
            model_metrics.items(),
            key=lambda x: x[1]
        )
        
        return {
            'ranking': sorted_models,
            'best_model': sorted_models[0][0],
            'best_score': sorted_models[0][1]
        }
```

```python
# src/regression/pipeline/results_formatter.py
class ResultsFormatter:
    """
    SRP: Responsável APENAS por formatar resultados para exibição.
    """
    
    def format_metrics_table(self, results: Dict[str, Any]) -> str:
        """Formatar tabela de métricas"""
        pass
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Imprimir resumo formatado"""
        pass
```

```python
# src/regression/pipeline/results_persister.py
from pathlib import Path
import json

class ResultsPersister:
    """
    SRP: Responsável APENAS por salvar resultados em disco.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str) -> None:
        """Salvar métricas em JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_predictions(self, predictions: np.ndarray, filename: str) -> None:
        """Salvar predições em NPY"""
        filepath = self.output_dir / filename
        np.save(filepath, predictions)
```

```python
# src/regression/pipeline/regression_pipeline.py
from typing import Optional, Dict, Any
from ..interfaces import IDataPreparer, IModelTrainer, IResultsEvaluator

class RegressionPipeline:
    """
    FACADE PATTERN: Orquestra componentes modularizados.
    
    Agora MUITO mais simples - apenas coordena!
    """
    
    def __init__(
        self,
        data_preparer: IDataPreparer,
        model_trainer: IModelTrainer,
        results_evaluator: IResultsEvaluator,
        results_formatter: Optional['ResultsFormatter'] = None,
        results_persister: Optional['ResultsPersister'] = None,
        verbose: bool = True
    ):
        """
        DIP: Todas as dependências INJETADAS!
        """
        self.data_preparer = data_preparer
        self.model_trainer = model_trainer
        self.results_evaluator = results_evaluator
        self.results_formatter = results_formatter
        self.results_persister = results_persister
        self.verbose = verbose
    
    def run(self) -> Dict[str, Any]:
        """
        Executar pipeline completo.
        
        MUITO mais simples agora!
        """
        if self.verbose:
            print('🚀 PIPELINE DE REGRESSÃO')
        
        # Etapa 1: Preparar dados
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.data_preparer.split_data()
        
        # Etapa 2: Treinar modelos
        trained_models = self.model_trainer.train(X_train, y_train)
        
        # Etapa 3: Avaliar no teste
        test_results = {}
        for name, model in trained_models.items():
            test_results[name] = self.results_evaluator.evaluate_model(
                model, X_test, y_test
            )
        
        # Etapa 4: Comparar modelos
        comparison = self.results_evaluator.compare_models(test_results)
        
        # Etapa 5: Formatar (se fornecido)
        if self.results_formatter:
            self.results_formatter.print_summary(comparison)
        
        # Etapa 6: Salvar (se fornecido)
        if self.results_persister:
            self.results_persister.save_metrics(
                comparison, 'test_results.json'
            )
        
        return test_results
```

**Commit**: `refactor: split RegressionPipeline into focused classes (SRP)`

---

#### 3.2 Refatorar Factory com DIP
```python
# src/regression/factory/pipeline_factory.py
from ..pipeline import (
    DataPreparer, ModelTrainer, ResultsEvaluator,
    ResultsFormatter, ResultsPersister, RegressionPipeline
)
from ..models import RegressionModels

class RegressionPipelineFactory:
    """
    Factory para criar pipelines com configs padrão.
    
    Facilita criação sem expor detalhes de construção.
    """
    
    @staticmethod
    def create_default_pipeline(
        embeddings_path: str,
        targets_path: str,
        output_dir: str = 'results/regression',
        random_state: int = 42,
        verbose: bool = True
    ) -> RegressionPipeline:
        """Criar pipeline com configurações padrão"""
        
        # Criar componentes
        data_preparer = DataPreparer(
            embeddings_path,
            targets_path,
            verbose=verbose
        )
        
        models = RegressionModels.get_all_models(random_state)
        model_trainer = ModelTrainer(
            models_dict=models,
            verbose=verbose,
            random_state=random_state
        )
        
        results_evaluator = ResultsEvaluator(verbose=verbose)
        results_formatter = ResultsFormatter()
        results_persister = ResultsPersister(output_dir)
        
        # Criar pipeline (DIP - injeta tudo)
        return RegressionPipeline(
            data_preparer=data_preparer,
            model_trainer=model_trainer,
            results_evaluator=results_evaluator,
            results_formatter=results_formatter,
            results_persister=results_persister,
            verbose=verbose
        )
    
    @staticmethod
    def create_custom_pipeline(
        data_preparer: 'IDataPreparer',
        model_trainer: 'IModelTrainer',
        results_evaluator: 'IResultsEvaluator',
        **kwargs
    ) -> RegressionPipeline:
        """Criar pipeline com componentes customizados"""
        return RegressionPipeline(
            data_preparer=data_preparer,
            model_trainer=model_trainer,
            results_evaluator=results_evaluator,
            **kwargs
        )
```

**Commit**: `refactor: add factory with dependency injection (DIP)`

---

### **Fase 4: Nova Estrutura de Diretórios (Dia 3, ~1h)** 📁

```
src/regression/
├── __init__.py                      # Exports públicos
│
├── interfaces/                      # Abstrações (DIP)
│   ├── __init__.py
│   ├── data_interfaces.py          # IDataPreparer, IDataValidator
│   ├── model_interfaces.py         # IRegressor, IModelTrainer
│   └── evaluation_interfaces.py    # IMetricsCalculator, IResultsEvaluator
│
├── core/                            # Componentes core
│   ├── __init__.py
│   ├── data_loader.py              # DataManager (implementa IDataPreparer)
│   ├── data_validator.py           # DataValidator (implementa IDataValidator)
│   └── metrics_calculator.py       # MetricsCalculator (implementa IMetricsCalculator)
│
├── models/                          # Modelos
│   ├── __init__.py
│   ├── base_regressor.py           # BaseRegressor (ABC)
│   ├── sklearn_wrappers.py         # Wrappers sklearn → BaseRegressor
│   └── models.py                   # RegressionModels factory
│
├── pipeline/                        # Pipeline components (SRP)
│   ├── __init__.py
│   ├── data_preparer.py            # DataPreparer
│   ├── model_trainer.py            # ModelTrainer
│   ├── results_evaluator.py        # ResultsEvaluator
│   ├── results_formatter.py        # ResultsFormatter
│   ├── results_persister.py        # ResultsPersister
│   └── regression_pipeline.py      # RegressionPipeline (Facade)
│
├── factory/                         # Factories (criação)
│   ├── __init__.py
│   └── pipeline_factory.py         # RegressionPipelineFactory
│
├── config/                          # Configurações
│   ├── __init__.py
│   ├── regression_config.py        # RegressionConfig (dataclass)
│   └── model_configs.py            # Configs de cada modelo
│
├── utils/                           # Utilitários
│   ├── __init__.py
│   ├── metrics.py                  # Métricas utilitárias
│   ├── visualization.py            # Plots
│   └── logging_utils.py            # Logging
│
└── cli/                             # Interface CLI
    ├── __init__.py
    └── modular_regression.py       # CLI principal

# Remover da raiz (movidos)
# trainer.py         → REMOVIDO (usar core/)
# evaluator.py       → REMOVIDO (usar core/)
# models.py          → REMOVIDO (usar models/)
# logger.py          → MOVIDO para utils/logging_utils.py
# visualizer.py      → MOVIDO para utils/visualization.py
# validation.py      → MOVIDO para core/data_validator.py
# utils.py           → SPLIT em utils/*
```

**Commit**: `refactor: reorganize directory structure (clean architecture)`

---

## 📊 Comparação Antes vs Depois

### Antes da Refatoração ❌

```python
# Violações SOLID:

# 1. SRP violado
class RegressionPipeline:
    # 430 linhas, 5 responsabilidades

# 2. OCP violado
# Sem abstrações, tudo hardcoded

# 3. LSP violado
# Modelos não intercambiáveis

# 4. ISP violado
class RegressionEvaluator:
    # Interface gorda com métodos não relacionados

# 5. DIP violado
class RegressionPipeline:
    def __init__(self):
        self.data = DataManager(...)  # Cria internamente
```

**Métricas**:
- Arquivos duplicados: 3
- Linhas por classe: ~430 (RegressionPipeline)
- Acoplamento: Alto
- Testabilidade: Baixa
- Extensibilidade: Difícil

---

### Depois da Refatoração ✅

```python
# SOLID compliant:

# 1. SRP ✅
class DataPreparer:        # SRP: Apenas dados
class ModelTrainer:        # SRP: Apenas treino
class ResultsEvaluator:    # SRP: Apenas avaliação
class ResultsFormatter:    # SRP: Apenas formatação
class ResultsPersister:    # SRP: Apenas persistência

class RegressionPipeline:  # Facade: Orquestra
    # Apenas 50 linhas!

# 2. OCP ✅
class BaseRegressor(ABC):  # Interface abstrata
    # Extensível sem modificar

# 3. LSP ✅
# Todos os modelos intercambiáveis via BaseRegressor

# 4. ISP ✅
class IMetricsCalculator:  # Interface específica
class IResultsEvaluator:   # Interface específica
# Clientes dependem apenas do necessário

# 5. DIP ✅
class RegressionPipeline:
    def __init__(
        self,
        data_preparer: IDataPreparer,      # Injeta interface
        model_trainer: IModelTrainer,      # Injeta interface
        results_evaluator: IResultsEvaluator  # Injeta interface
    ):
        pass
```

**Métricas**:
- Arquivos duplicados: 0 ✅
- Linhas por classe: ~50-100 (focadas)
- Acoplamento: Baixo
- Testabilidade: Alta ✅
- Extensibilidade: Fácil ✅

---

## ✅ Checklist de Refatoração

### Fase 1: Limpeza ✅
- [ ] Remover `src/regression/trainer.py`
- [ ] Remover `src/regression/evaluator.py`
- [ ] Remover `src/regression/models.py`
- [ ] Atualizar imports em todos os arquivos
- [ ] Teste de sanidade (imports funcionam)
- [ ] Commit: "remove duplicate files"

### Fase 2: Abstrações ✅
- [ ] Criar `interfaces/data_interfaces.py`
- [ ] Criar `interfaces/model_interfaces.py`
- [ ] Criar `interfaces/evaluation_interfaces.py`
- [ ] Criar `models/base_regressor.py`
- [ ] Criar `models/sklearn_wrappers.py`
- [ ] Commit: "add SOLID interfaces and base classes"

### Fase 3: Refatorar Classes ✅
- [ ] Criar `pipeline/data_preparer.py`
- [ ] Criar `pipeline/model_trainer.py`
- [ ] Criar `pipeline/results_evaluator.py`
- [ ] Criar `pipeline/results_formatter.py`
- [ ] Criar `pipeline/results_persister.py`
- [ ] Refatorar `pipeline/regression_pipeline.py` (Facade)
- [ ] Criar `factory/pipeline_factory.py`
- [ ] Commit: "refactor pipeline with SRP and DIP"

### Fase 4: Reorganizar ✅
- [ ] Mover arquivos para nova estrutura
- [ ] Atualizar `__init__.py` em cada pasta
- [ ] Atualizar imports globais
- [ ] Commit: "reorganize directory structure"

### Fase 5: Validação ✅
- [ ] Executar pipeline refatorado
- [ ] Comparar resultados (deve ser idêntico)
- [ ] Verificar não há regressões
- [ ] Commit: "validate refactored implementation"

---

## 🎯 Próximos Passos

Após refatoração completa:

1. **Criar testes** (86 testes, 12 níveis)
2. **Documentação** (README.md completo)
3. **Merge** para branch principal

---

**Está pronto para começar a refatoração? Qual fase deseja que eu implemente primeiro?** 🚀
