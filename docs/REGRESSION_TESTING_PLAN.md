# 🎯 Plano de Testes - Módulo Regression

**Data**: 2025-11-09  
**Status**: 📋 PLANEJAMENTO  
**Objetivo**: 100% de cobertura de testes com princípios SOLID

---

## 📊 Análise do Estado Atual

### ✅ Estrutura Existente

```
src/regression/
├── __init__.py
├── config.py                  # RegressionConfig (dataclass)
├── modular_pipeline.py        # RegressionPipeline (classe principal)
├── modular_regression.py      # CLI interface
│
├── core/                      # Módulos core
│   ├── __init__.py
│   ├── data_loader.py        # DataManager
│   ├── trainer.py            # RegressionTrainer
│   └── evaluator.py          # RegressionEvaluator
│
├── models/                    # Modelos
│   ├── __init__.py
│   └── models.py             # RegressionModels (factory)
│
└── utils/                     # Utilitários
    ├── __init__.py
    └── metrics.py            # MetricsCalculator

# Arquivos legados (a revisar)
├── trainer.py                 # ⚠️ Duplicado com core/trainer.py
├── evaluator.py               # ⚠️ Duplicado com core/evaluator.py
├── logger.py                  # RegressionLogger
├── visualizer.py              # RegressionVisualizer
├── utils.py                   # Funções utilitárias legacy
├── validation.py              # Validação de dados
└── models.py                  # ⚠️ Duplicado com models/models.py
```

### 📈 Comparação com Classifier

| Aspecto | Classifier | Regression | Status |
|---------|-----------|------------|--------|
| **Estrutura modular** | ✅ Completa | ⚠️ Parcial | Melhorar |
| **Princípios SOLID** | ✅ 100% | ⚠️ ~60% | Aplicar |
| **Testes** | ✅ 93 (100%) | ❌ 0 | Criar |
| **Documentação** | ✅ README completo | ⚠️ Básica | Criar |
| **Arquivos duplicados** | ✅ Nenhum | ❌ 3 duplicados | Limpar |

---

## 🔍 Análise SOLID

### Problemas Identificados

#### 1. **SRP Violado** ❌
```python
# modular_pipeline.py
class RegressionPipeline:
    # Faz muitas coisas:
    # - Gerencia dados
    # - Treina modelos
    # - Avalia resultados
    # - Salva outputs
    # - Logging
    # - Visualização
    # ~430 linhas!
```

#### 2. **Arquivos Duplicados** ❌
- `trainer.py` vs `core/trainer.py`
- `evaluator.py` vs `core/evaluator.py`
- `models.py` vs `models/models.py`

#### 3. **Dependências Hardcoded** ❌
```python
# Viola DIP
self.data_manager = DataManager(...)  # Criado internamente
self.evaluator = RegressionEvaluator()  # Não injetado
```

#### 4. **Sem Abstrações** ❌
- Nenhuma classe base abstrata
- Modelos sem interface comum
- Não usa OCP

---

## 🎯 Estratégia de Refatoração

### Fase 1: Estrutura Base (Dias 1-2)

#### 1.1 Limpar Duplicados
```bash
# Mover para core/ e remover raiz
rm src/regression/trainer.py       # Usar core/trainer.py
rm src/regression/evaluator.py     # Usar core/evaluator.py  
rm src/regression/models.py        # Usar models/models.py
```

#### 1.2 Criar Abstrações (OCP + LSP)
```python
# models/base_model.py
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    """Classe base para todos os regressores"""
    
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
```

#### 1.3 Separar Responsabilidades (SRP)
```python
# Quebrar RegressionPipeline em:

class DataPreparer:           # SRP: Apenas preparar dados
    def load_data(self): pass
    def split_data(self): pass

class ModelRegistry:          # SRP: Apenas gerenciar modelos
    def register_model(self): pass
    def get_model(self): pass

class TrainingOrchestrator:   # SRP: Apenas orquestrar treino
    def train_all(self): pass
    def get_best_model(self): pass

class ResultsManager:         # SRP: Apenas gerenciar resultados
    def save_predictions(self): pass
    def save_metrics(self): pass

class RegressionPipeline:     # Facade - orquestra componentes
    def __init__(self, data_preparer, registry, orchestrator, results_mgr):
        # DIP: Injeta dependências
        pass
```

---

## 🧪 Plano de Testes (12 Níveis)

### **Level 1: Foundation (8 testes)** 🏗️

#### Test 1.1: Data Loading
```python
test_1_1_data_loader.py
- DataManager.load_embeddings()
- DataManager.load_targets()
- DataManager.validate_shapes()
- Cache handling
- Edge cases: empty, malformed
```

#### Test 1.2: Data Validation
```python
test_1_2_data_validation.py
- validate_regression_data()
- Check NaN/Inf
- Check feature dimensions
- Check target range
- Edge cases
```

#### Test 1.3: Data Splitting
```python
test_1_3_data_splitting.py
- Train/val/test split
- Stratification (se aplicável)
- Reproducibility (seed)
- Edge cases: tiny datasets
```

#### Test 1.4: Config Management
```python
test_1_4_config_manager.py
- RegressionConfig creation
- get_fast_config()
- get_production_config()
- Serialization (JSON)
- Validation
```

#### Test 1.5: Metrics Calculator
```python
test_1_5_metrics_calculator.py
- MSE, RMSE, MAE
- R², adjusted R²
- MAPE, SMAPE
- Explained variance
- Edge cases
```

#### Test 1.6: Utils Functions
```python
test_1_6_utils.py
- prepare_regression_targets()
- load_split_indices()
- save_split_indices()
- Cache functions
```

#### Test 1.7: Logger
```python
test_1_7_logger.py
- RegressionLogger initialization
- log_training_start()
- log_metrics()
- save_logs()
```

#### Test 1.8: Visualizer
```python
test_1_8_visualizer.py
- plot_predictions()
- plot_residuals()
- plot_feature_importance()
- Save plots
```

---

### **Level 2: Models (6 testes)** 🤖

#### Test 2.1: Model Factory
```python
test_2_1_model_factory.py
- RegressionModels.get_all_models()
- Individual model creation
- Model parameters
- Optional dependencies (XGBoost, LightGBM, CatBoost)
```

#### Test 2.2: Random Forest
```python
test_2_2_random_forest.py
- Training
- Prediction
- Feature importance
- Hyperparameters
```

#### Test 2.3: Gradient Boosting
```python
test_2_3_gradient_boosting.py
- Training
- Prediction
- Early stopping
- Learning curve
```

#### Test 2.4: Linear Models
```python
test_2_4_linear_models.py
- Ridge regression
- Lasso regression
- ElasticNet
- Coefficients
```

#### Test 2.5: Neural Network
```python
test_2_5_mlp_regressor.py
- MLPRegressor training
- Architecture validation
- Convergence
```

#### Test 2.6: Optional Models
```python
test_2_6_optional_models.py
- XGBoost (if available)
- LightGBM (if available)
- CatBoost (if available)
- Graceful degradation
```

---

### **Level 3: Training & Evaluation (25 testes)** 🎓

#### Test 3.1: Single Model Training
```python
test_3_1_single_training.py
- Train one model
- Check convergence
- Validate predictions
- Timing
```

#### Test 3.2: Multiple Models Training
```python
test_3_2_multi_training.py
- Train all models
- Compare results
- Best model selection
- Parallel training
```

#### Test 3.3: Evaluator Metrics
```python
test_3_3_evaluator_metrics.py
- RegressionEvaluator.evaluate()
- All metrics computation
- Ranking models
- Confidence intervals
```

#### Test 3.4: Cross-Validation (5 testes)
```python
test_3_4_cross_validation.py
- 5-fold CV
- KFold strategy
- Metrics aggregation
- Reproducibility
- Edge cases
```

#### Test 3.5: Model Persistence (5 testes)
```python
test_3_5_model_persistence.py
- Save model (joblib)
- Load model
- Save metadata
- Versioning
- Checkpoint recovery
```

#### Test 3.6: Early Stopping (5 testes)
```python
test_3_6_early_stopping.py
- GradientBoosting early stop
- Validation monitoring
- Best iteration
- Patience parameter
- Restore best
```

#### Test 3.7: Feature Importance (5 testes)
```python
test_3_7_feature_importance.py
- Random Forest importance
- Gradient Boosting importance
- Linear model coefficients
- Permutation importance
- Visualization
```

---

### **Level 4: Integration (10 testes)** 🔗

#### Test 4.1: Data → Model Pipeline
```python
test_4_1_data_model_pipeline.py
- Load → Transform → Train
- Full workflow
- Error propagation
```

#### Test 4.2: Model → Evaluation Pipeline
```python
test_4_2_model_eval_pipeline.py
- Train → Predict → Evaluate
- Multiple models
- Comparison
```

#### Test 4.3: Config → Pipeline Integration
```python
test_4_3_config_pipeline.py
- Load config → Create pipeline
- Apply all settings
- Reproducibility
```

#### Test 4.4: Full Pipeline
```python
test_4_4_full_pipeline.py
- RegressionPipeline.run()
- Load → Train → Evaluate → Save
- 5-7 steps complete
- Results verification
```

---

### **Level 5: Edge Cases (8 testes)** ⚠️

#### Test 5.1: Empty Data
```python
test_5_1_empty_data.py
- 0 samples
- Error handling
- Graceful failure
```

#### Test 5.2: Single Sample
```python
test_5_2_single_sample.py
- 1 sample only
- Cannot split
- Error messages
```

#### Test 5.3: Large Dataset
```python
test_5_3_large_dataset.py
- 100K+ samples
- Memory efficiency
- Performance
```

#### Test 5.4: High Dimensionality
```python
test_5_4_high_dimensions.py
- 10K+ features
- Curse of dimensionality
- Model behavior
```

#### Test 5.5: Outliers
```python
test_5_5_outliers.py
- Extreme values
- Robust metrics
- Model robustness
```

#### Test 5.6: Missing Values
```python
test_5_6_missing_values.py
- NaN handling
- Inf handling
- Imputation
```

#### Test 5.7: Constant Features
```python
test_5_7_constant_features.py
- Zero variance
- Feature selection
- Model behavior
```

#### Test 5.8: Correlated Features
```python
test_5_8_correlated_features.py
- High collinearity
- Multicollinearity
- Model stability
```

---

### **Level 6: Performance (4 testes)** ⚡

#### Test 6.1: Training Speed
```python
test_6_1_training_speed.py
- Benchmark all models
- Samples/second
- Scaling (1K → 10K → 100K)
```

#### Test 6.2: Prediction Speed
```python
test_6_2_prediction_speed.py
- Inference latency
- Batch prediction
- Single vs batch
```

#### Test 6.3: Memory Usage
```python
test_6_3_memory_usage.py
- Memory profiling
- Peak memory
- Memory leaks
```

#### Test 6.4: Parallel Training
```python
test_6_4_parallel_training.py
- n_jobs parameter
- Speedup factor
- Thread safety
```

---

### **Level 7: Serialization (3 testes)** 💾

#### Test 7.1: Model Checkpoints
```python
test_7_1_model_checkpoints.py
- Save/load models
- Joblib format
- Pickle compatibility
```

#### Test 7.2: Predictions Export
```python
test_7_2_predictions_export.py
- CSV export
- NPY export
- JSON export
```

#### Test 7.3: Metrics Export
```python
test_7_3_metrics_export.py
- JSON metrics
- Summary reports
- Comparison tables
```

---

### **Level 8: End-to-End (3 testes)** 🚀

#### Test 8.1: Complete Workflow
```python
test_8_1_complete_workflow.py
- Load data
- Train all models
- Evaluate
- Save results
- Generate plots
```

#### Test 8.2: Production Pipeline
```python
test_8_2_production_pipeline.py
- Load config
- Run pipeline
- Error handling
- Logging
```

#### Test 8.3: Reproducibility
```python
test_8_3_reproducibility.py
- Same seed → same results
- Multiple runs
- Determinism
```

---

### **Level 9: Hyperparameter Tuning (4 testes)** 🎯

#### Test 9.1: Grid Search
```python
test_9_1_grid_search.py
- Parameter grid
- Best params
- CV scores
```

#### Test 9.2: Random Search
```python
test_9_2_random_search.py
- Random sampling
- n_iter parameter
- Best estimator
```

#### Test 9.3: Optuna Integration
```python
test_9_3_optuna.py
- Study creation
- Optimization
- Pruning
- Best trial
```

#### Test 9.4: AutoML
```python
test_9_4_automl.py
- Automated search
- Model selection
- Ensemble
```

---

### **Level 10: Validation Strategies (4 testes)** ✅

#### Test 10.1: K-Fold CV
```python
test_10_1_kfold_cv.py
- KFold splitter
- 5-fold validation
- Metrics aggregation
```

#### Test 10.2: Stratified K-Fold
```python
test_10_2_stratified_kfold.py
- StratifiedKFold
- Target binning
- Balance preservation
```

#### Test 10.3: Time Series Split
```python
test_10_3_time_series_split.py
- TimeSeriesSplit
- Forward chaining
- No data leakage
```

#### Test 10.4: Leave-One-Out
```python
test_10_4_leave_one_out.py
- LOOCV
- Expensive but thorough
- Small datasets
```

---

### **Level 11: Feature Engineering (4 testes)** 🔬

#### Test 11.1: Scaling
```python
test_11_1_scaling.py
- StandardScaler
- MinMaxScaler
- RobustScaler
- Effect on models
```

#### Test 11.2: Feature Selection
```python
test_11_2_feature_selection.py
- Variance threshold
- Correlation filter
- Model-based selection
```

#### Test 11.3: Dimensionality Reduction
```python
test_11_3_dimensionality_reduction.py
- PCA
- Feature importance
- Information loss
```

#### Test 11.4: Polynomial Features
```python
test_11_4_polynomial_features.py
- Interaction terms
- Polynomial degree
- Complexity vs performance
```

---

### **Level 12: CLI & Pipeline (3 testes)** 🖥️

#### Test 12.1: CLI Arguments
```python
test_12_1_cli_arguments.py
- Parse arguments
- Validate inputs
- Help messages
```

#### Test 12.2: Pipeline Factory
```python
test_12_2_pipeline_factory.py
- run_regression_pipeline()
- Configuration loading
- Component creation
```

#### Test 12.3: Integration Test
```python
test_12_3_integration.py
- modular_regression.py
- Full execution
- Output validation
```

---

## 📊 Resumo do Plano

| Nível | Descrição | Testes | Prioridade |
|-------|-----------|--------|------------|
| **1** | Foundation | 8 | 🔴 CRÍTICA |
| **2** | Models | 6 | 🔴 CRÍTICA |
| **3** | Training & Evaluation | 25 | 🔴 CRÍTICA |
| **4** | Integration | 10 | 🟡 ALTA |
| **5** | Edge Cases | 8 | 🟡 ALTA |
| **6** | Performance | 4 | 🟢 MÉDIA |
| **7** | Serialization | 3 | 🟡 ALTA |
| **8** | End-to-End | 3 | 🔴 CRÍTICA |
| **9** | Hyperparameter Tuning | 4 | 🟢 MÉDIA |
| **10** | Validation Strategies | 4 | 🟡 ALTA |
| **11** | Feature Engineering | 4 | 🟢 MÉDIA |
| **12** | CLI & Pipeline | 3 | 🟡 ALTA |
| **TOTAL** | **12 níveis** | **86** | - |

---

## 🛠️ Cronograma de Implementação

### **Semana 1: Refatoração + Níveis 1-3**
- **Dia 1-2**: Refatoração SOLID
  - Limpar duplicados
  - Criar abstrações
  - Aplicar DIP/SRP
  
- **Dia 3-4**: Level 1 Foundation (8 testes)
  - Data loading, validation, config
  
- **Dia 5**: Level 2 Models (6 testes)
  - Model factory e testes individuais

- **Dia 6-7**: Level 3 Training & Evaluation (25 testes)
  - Treinamento, avaliação, CV

### **Semana 2: Níveis 4-8**
- **Dia 8-9**: Level 4 Integration (10 testes)
- **Dia 10**: Level 5 Edge Cases (8 testes)
- **Dia 11**: Level 6 Performance (4 testes)
- **Dia 12**: Level 7 Serialization (3 testes)
- **Dia 13-14**: Level 8 End-to-End (3 testes)

### **Semana 3: Níveis 9-12 + Documentação**
- **Dia 15**: Level 9 Hyperparameter Tuning (4 testes)
- **Dia 16**: Level 10 Validation Strategies (4 testes)
- **Dia 17**: Level 11 Feature Engineering (4 testes)
- **Dia 18**: Level 12 CLI & Pipeline (3 testes)
- **Dia 19-20**: Documentação completa
- **Dia 21**: Revisão final e merge

---

## 🎯 Critérios de Sucesso

### Mínimo (MVP)
- ✅ 70+ testes passando
- ✅ Níveis 1, 2, 3, 4, 8 completos
- ✅ Refatoração SOLID aplicada
- ✅ Duplicados removidos

### Ideal (Target)
- ✅ 86 testes passando (100%)
- ✅ Todos os 12 níveis completos
- ✅ README.md completo
- ✅ 100% SOLID compliance
- ✅ Documentação de APIs

### Excelência (Stretch)
- ✅ CI/CD configurado
- ✅ Coverage reports
- ✅ Integração ESM → Regression
- ✅ Notebooks de exemplo

---

## 📝 Próximos Passos Imediatos

1. **Criar diretório de testes**
   ```bash
   mkdir -p tests/regression_test
   ```

2. **Começar com Level 1.1** (Data Loading)
   ```bash
   touch tests/regression_test/test_1_1_data_loader.py
   ```

3. **Refatorar enquanto testa**
   - Identificar violações SOLID
   - Corrigir incrementalmente
   - Testar cada correção

---

**🚀 Pronto para começar! Quer iniciar com Level 1.1 ou preferir a refatoração SOLID primeiro?**
