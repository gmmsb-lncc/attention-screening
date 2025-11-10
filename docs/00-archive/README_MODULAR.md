# Módulo de Regressão Modular - DockTKinase

## 📋 Visão Geral

Este módulo implementa pipeline completo de regressão para predição de valores de atividade (Ki, Kd, IC50) seguindo o **mesmo padrão de modularização do classificador**.

## 🏗️ Arquitetura Modular

```
src/regression/
├── core/                      # Funcionalidades centrais
│   ├── __init__.py           # Exports principais
│   ├── data_loader.py        # Gerenciamento de dados (NOVO)
│   ├── evaluator.py          # Avaliação e métricas
│   └── trainer.py            # Treinamento de modelos
│
├── models/                    # Modelos de regressão
│   ├── __init__.py           # Exports de modelos
│   └── models.py             # Factory de modelos ML
│
├── utils/                     # Utilitários
│   ├── __init__.py           # Exports de utils
│   └── metrics.py            # Calculador de métricas (NOVO)
│
├── modular_pipeline.py        # Pipeline principal (NOVO)
├── modular_regression.py      # CLI interface (NOVO)
│
└── (arquivos originais mantidos para compatibilidade)
```

## ✨ Componentes Principais

### 1. **DataManager** (`core/data_loader.py`)
Gerenciamento inteligente de dados:
- ✅ Carregamento de embeddings (.npy/.npz)
- ✅ Carregamento de targets
- ✅ Cache em memória
- ✅ **Stratified split baseado em bins quantílicos**
- ✅ Auto-detecção de dimensões
- ✅ Estatísticas dos dados

```python
from regression.core import DataManager

manager = DataManager('embeddings.npy', 'targets.npy')
X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data()
stats = manager.get_stats()
```

### 2. **MetricsCalculator** (`utils/metrics.py`)
Cálculo completo de 15+ métricas:
- **Principais**: MAE, MSE, RMSE, R², MedianAE, MAPE
- **Avançadas**: Explained Variance, Max Error
- **Estatísticas**: Resíduos (média, std)
- **Percentis**: P25, P50, P75, P90, P95, P99
- **Normalizadas**: RMSE normalizado, CV-RMSE
- **Comparação**: Tabelas e rankings

```python
from regression.utils import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_all_metrics(y_true, y_pred, 'MyModel')
print(calculator.format_metrics_table(metrics))
```

### 3. **RegressionModels** (`models/models.py`)
Factory de 11+ modelos:
- RandomForest
- GradientBoosting
- XGBoost (opcional)
- LightGBM (opcional)
- CatBoost (opcional)
- Ridge, Lasso, ElasticNet
- SVR
- KNN
- DecisionTree
- MLP

```python
from regression.models import RegressionModels

models = RegressionModels.get_all_models(random_state=42)
# {'RandomForest': <model>, 'GradientBoosting': <model>, ...}
```

### 4. **RegressionTrainer** (`core/trainer.py`)
Treinamento e avaliação:
- ✅ Treina múltiplos modelos em paralelo
- ✅ Avaliação automática em validação
- ✅ Tracking de tempo de treino
- ✅ Ranking de modelos

```python
from regression.core import RegressionTrainer

trainer = RegressionTrainer(models_dict=models)
trainer.train_all(X_train, y_train, X_val, y_val)
```

### 5. **RegressionPipeline** (`modular_pipeline.py`)
Pipeline completo orquestrado:
- ✅ Carrega dados automaticamente
- ✅ Divide em treino/val/teste
- ✅ Treina todos os modelos
- ✅ Avalia e compara resultados
- ✅ Salva métricas e stats

```python
from regression.modular_pipeline import RegressionPipeline

pipeline = RegressionPipeline(
    embeddings_path='embeddings.npy',
    targets_path='targets.npy',
    output_dir='results/my_test'
)
results = pipeline.run()
```

## 🚀 Uso

### Via Python (API)

```python
from regression.modular_pipeline import RegressionPipeline

# Pipeline completo
pipeline = RegressionPipeline(
    embeddings_path='protein_embeddings.npy',
    targets_path='activity_targets.npy',
    output_dir='results/regression',
    models_to_train=['RandomForest', 'XGBoost', 'KNN'],  # Opcional
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    verbose=True
)

# Executar
results = pipeline.run()

# Acessar componentes individuais
pipeline.load_data()
pipeline.train_models()
pipeline.evaluate_on_test()
pipeline.save_results()
```

### Via CLI

```bash
# Treinar todos os modelos
python src/regression/modular_regression.py embeddings.npy targets.npy

# Modelos específicos
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --models RandomForest GradientBoosting XGBoost

# Configurar output
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --output results/my_experiment

# Configurar splits
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --test-size 0.15 --val-size 0.15

# Seed customizada
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --random-state 123

# Modo silencioso
python src/regression/modular_regression.py embeddings.npy targets.npy --quiet
```

## 📊 Output

O pipeline gera automaticamente:

```
results/regression/
├── metrics/
│   ├── test_metrics.json          # Métricas de teste
│   └── validation_metrics.json    # Métricas de validação
├── models/                         # (preparado para salvar modelos)
├── predictions/                    # (preparado para predições)
└── pipeline_stats.json            # Estatísticas do pipeline
```

### Exemplo de `test_metrics.json`:

```json
{
  "RandomForest": {
    "model_name": "RandomForest",
    "n_samples": 100,
    "MAE": 5.234,
    "RMSE": 8.456,
    "R2": 0.834,
    "MedianAE": 3.567,
    "MAPE": 12.45,
    "ExplainedVariance": 0.842,
    "MaxError": 25.678,
    "mean_residual": 0.123,
    "std_residual": 8.234,
    "error_p25": 2.345,
    "error_p50": 4.567,
    "error_p75": 7.890,
    "error_p90": 12.345,
    "error_p95": 15.678,
    "error_p99": 22.345
  }
}
```

## 🎯 Benefícios da Modularização

| Aspecto | Original | Modular |
|---------|----------|---------|
| **Organização** | Múltiplos arquivos desconexos | Estrutura clara (core/models/utils) |
| **Testabilidade** | Difícil testar componentes | Cada módulo testável individualmente |
| **Reusabilidade** | Código duplicado | Componentes reutilizáveis |
| **Manutenção** | Mudança afeta múltiplos arquivos | Mudança isolada em módulo |
| **Documentação** | Dispersa | Centralizada e organizada |
| **Compatibilidade** | - | 100% compatível com original |

## 🔄 Compatibilidade

### ✅ Mantém 100% de compatibilidade:
- Mesmos algoritmos e parâmetros
- Mesma divisão de dados (stratified)
- Mesmas métricas calculadas
- Mesmo formato de outputs

### ✨ Melhora:
- Organização do código
- Facilidade de teste
- Documentação
- Extensibilidade
- Manutenibilidade

## 🧪 Teste

```python
# Teste básico
from regression.modular_pipeline import run_regression_pipeline

results = run_regression_pipeline(
    embeddings_path='tests/data/test_embeddings.npy',
    targets_path='tests/data/test_targets.npy',
    output_dir='results/test',
    models=['RandomForest', 'Ridge'],
    random_state=42
)

print(f"Melhor MAE: {min(m['MAE'] for m in results.values()):.4f}")
print(f"Melhor R²: {max(m['R2'] for m in results.values()):.4f}")
```

## 📚 Padrões Aplicados

1. **Single Responsibility**: Cada módulo tem uma responsabilidade
2. **Factory Pattern**: `RegressionModels.get_all_models()`
3. **Strategy Pattern**: Diferentes calculadores podem ser injetados
4. **Composition**: Pipeline compõe componentes
5. **Dependency Injection**: Componentes injetados via construtor
6. **Lazy Loading**: Dados carregados apenas quando necessário
7. **Caching**: Embeddings/targets mantidos em memória

## 🔍 Comparação com Classificador

### Estrutura Idêntica:
```
classifier/                    regression/
├── core/                     ├── core/
│   ├── evaluator.py         │   ├── evaluator.py
│   ├── data_loader.py       │   ├── data_loader.py
│   └── trainer.py           │   └── trainer.py
├── models/                   ├── models/
│   └── mlp_classifier.py    │   └── models.py
├── utils/                    ├── utils/
│   └── metrics.py           │   └── metrics.py
└── modular_pipeline.py       └── modular_pipeline.py
```

### Mesmos Princípios:
- ✅ Modularização clara
- ✅ Separação de responsabilidades
- ✅ 100% compatibilidade
- ✅ Documentação completa
- ✅ Padrões de design aplicados

## 📝 Próximos Passos

1. ✅ Estrutura modular criada
2. ✅ Pipeline funcional
3. ✅ CLI interface
4. ✅ Documentação completa
5. ⏳ Testes unitários (próximo)
6. ⏳ Integração com pipeline principal
7. ⏳ Visualizações modularizadas

## 🤝 Contribuindo

Ao adicionar novos modelos ou funcionalidades:
1. Coloque modelos em `models/`
2. Coloque métricas em `utils/`
3. Coloque lógica de treino em `core/`
4. Mantenha pipeline separado da implementação
5. Siga o padrão do classificador

---

**Desenvolvido seguindo o padrão de modularização do classificador DockTKinase** 🚀
