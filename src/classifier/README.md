# Módulo de Classificação - DockTKinase

Pipeline modular de classificação binária para predição de atividade de compostos.

## 🎯 Funcionalidades

### Pipeline Multi-Modelo
- **13 algoritmos de classificação** (10 base + 3 opcionais)
- XGBoost é obrigatório e deve estar instalado
- Seleção automática do melhor modelo baseado em ROC-AUC
- Métricas completas: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, etc.
- Divisão estratificada em treino/validação/teste

### Modelos Suportados

#### Modelos Base (10 - sempre disponíveis):
1. **RandomForest** - Random Forest com balanceamento de classes
2. **GradientBoosting** - Gradient Boosting sequencial
3. **LogisticRegression** - Baseline linear com regularização L2
4. **LinearSVC** - Linear Support Vector Classifier (100-1000x mais rápido que SVC-RBF)
5. **ExtraTrees** - Extremely Randomized Trees (mais rápido que Random Forest)
6. **KNN** - K-Nearest Neighbors com distância ponderada
7. **MLP** - Multi-Layer Perceptron (sklearn)
8. **NaiveBayes** - Gaussian Naive Bayes
9. **DecisionTree** - Árvore de decisão única (baseline interpretável)
10. **AdaBoost** - Adaptive Boosting (boosting clássico)

#### Modelos de Gradient Boosting:
11. **XGBoost** - Extreme Gradient Boosting (⚠️ OBRIGATÓRIO - deve estar instalado)
12. **LightGBM** - Light Gradient Boosting Machine (opcional)
13. **CatBoost** - Categorical Boosting (opcional)

## 📦 Uso

### Pipeline Multi-Modelo (Recomendado)

```python
from classifier.multi_model_pipeline import MultiModelClassificationPipeline

# Criar pipeline
pipeline = MultiModelClassificationPipeline(
    embeddings_path='embeddings.npy',
    labels_path='labels.npy',
    output_dir='results/classification',
    random_state=42
)

# Executar (treina todos os 10 modelos)
results = pipeline.run()

# Resultados automáticos:
# - Treinamento de 13 modelos (10 base + XGBoost obrigatório + 2 opcionais)
# - Avaliação em validação e teste
# - Seleção do melhor modelo
# - Métricas salvas em JSON
```

### Pipeline MLP (PyTorch - Legacy)

```python
from classifier.modular_pipeline import MLPEmbeddingPipeline

pipeline = MLPEmbeddingPipeline(
    embeddings_path='embeddings.npy',
    labels_path='labels.npy',
    batch_size=64,
    epochs=50
)

pipeline.train()
```

## 📊 Saída

### Estrutura de Diretórios
```
results/classification_multi_model/
├── metrics/
│   ├── test_metrics.json          # Métricas de teste por modelo
│   └── validation_metrics.json    # Métricas de validação
├── models/                         # Modelos treinados
└── pipeline_stats.json            # Estatísticas do pipeline
```

### Exemplo de Saída
```
📊 RESUMO DOS RESULTADOS
================================================================================
Modelo                    Acc     Prec      Rec       F1    ROC-AUC
--------------------------------------------------------------------------------
🥇 XGBoost                0.8900   0.8750   0.8500   0.8623     0.9200
🥈 LightGBM               0.8850   0.8700   0.8450   0.8573     0.9180
🥉 RandomForest           0.8700   0.8600   0.8300   0.8448     0.9050
   CatBoost               0.8650   0.8550   0.8250   0.8397     0.9000
   GradientBoosting       0.8600   0.8500   0.8200   0.8347     0.8950
   ...

🏆 MELHOR MODELO: XGBoost
   ROC-AUC: 0.9200
   F1-Score: 0.8623
   Accuracy: 0.8900
```

## 🔧 Componentes

### Core
- `sklearn_trainer.py` - Treinador multi-modelo para sklearn
- `sklearn_data_manager.py` - Gerenciador de dados
- `trainer.py` - Treinador PyTorch (MLP)
- `evaluator.py` - Avaliador de modelos
- `cross_validator.py` - Validação cruzada

### Models
- `classifiers.py` - Factory de 10 modelos de classificação
- `mlp_classifier.py` - Modelo MLP PyTorch

### Pipelines
- `multi_model_pipeline.py` - Pipeline multi-modelo (sklearn)
- `modular_pipeline.py` - Pipeline MLP (PyTorch)

## 📈 Comparação: Multi-Modelo vs MLP

| Aspecto | Multi-Modelo | MLP Único |
|---------|-------------|-----------|
| Algoritmos | 10 modelos | 1 modelo |
| Melhor Performance | XGBoost/LightGBM | MLP |
| Tempo de Treino | ~18s (todos) | ~30s |
| ROC-AUC Típico | 0.90-0.92 | 0.85-0.89 |
| Interpretabilidade | Alta (alguns) | Baixa |
| Robustez | Alta (ensemble) | Média |

## 🚀 Exemplo Completo

```python
from classifier.multi_model_pipeline import MultiModelClassificationPipeline

# Pipeline completo
pipeline = MultiModelClassificationPipeline(
    embeddings_path='concatenated_embeddings/embeddings.npy',
    labels_path='concatenated_embeddings/labels.npy',
    output_dir='results/kinase_classification',
    test_size=0.1,      # 10% teste
    val_size=0.1,       # 10% validação
    random_state=42,
    verbose=True
)

# Executar
results = pipeline.run()

# Melhor modelo
best_model = max(results.items(), key=lambda x: x[1]['ROC_AUC'])
print(f"Melhor: {best_model[0]} - ROC-AUC: {best_model[1]['ROC_AUC']:.4f}")
```

## 🧪 Teste

```bash
# Testar pipeline multi-modelo
python tests/test_multi_model_classification.py

# Saída esperada:
# ✅ 10 modelos treinados
# ✅ Comparação de performance
# ✅ Seleção automática do melhor
```

## 📝 Notas

- **Recomendação**: Use `multi_model_pipeline.py` para projetos novos
- **Legacy**: `modular_pipeline.py` (MLP) mantido para compatibilidade
- **Performance**: XGBoost e LightGBM geralmente têm melhor desempenho
- **Balanceamento**: Todos os modelos usam `class_weight='balanced'`
- **Estratificação**: Split mantém proporção de classes em todos os conjuntos
