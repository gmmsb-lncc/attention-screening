# 🏆 Guia de Comparação de Classificadores

## 📊 Visão Geral

O script `compare_classifiers.py` treina e compara **automaticamente** múltiplos algoritmos de machine learning para encontrar o **melhor classificador** para seus dados.

> **🆕 NOVO**: O DockTKinase agora suporta **dual pipeline system**:
> - **Classification Pipeline**: Predição binária (ATIVO/INATIVO) - Este guia
> - **Regression Pipeline**: Predição quantitativa (Ki, Kd, IC50) - Ver `run_regression_pipeline.py`

## 🎯 Modelos Testados

| Modelo | Tipo | Descrição | Vantagens |
|--------|------|-----------|-----------|
| **RandomForest** | Ensemble | 100 árvores de decisão | Robusto, rápido, interpretável |
| **MLP_Small** | Neural Network | 2 camadas (128, 64) | Rápido, captura não-linearidades |
| **MLP_Large** | Neural Network | 3 camadas (256, 128, 64) | Mais capacidade de aprendizado |
| **SVM_Linear** | Kernel | Kernel linear | Rápido, eficiente em alta dimensão |
| **SVM_RBF** | Kernel | Kernel RBF (Gaussiano) | Captura padrões complexos |
| **GradientBoosting** | Ensemble | Boosting sequencial | Alta acurácia, mas mais lento |
| **LogisticRegression** | Linear | Regressão logística | Simples, interpretável, baseline |
| **KNN** | Instance-based | K vizinhos mais próximos | Não-paramétrico, simples |
| **XGBoost** ⭐ | Gradient Boosting | Otimizado | Estado-da-arte, muito rápido |

## 🚀 Como Usar

### Teste Rápido (100 amostras)
```bash
python compare_classifiers.py \
    --dataset human \
    --max-samples 100 \
    --output-dir tests/comparison_quick
```

### Teste Médio (1000 amostras)
```bash
python compare_classifiers.py \
    --dataset human \
    --max-samples 1000 \
    --model esm2_t12_35M_UR50D \
    --output-dir tests/comparison_1k
```

### Produção (dataset completo)
```bash
python compare_classifiers.py \
    --dataset human \
    --model esm2_t36_3B_UR50D \
    --device cuda \
    --output-dir tests/comparison_full
```

### Com Labels Customizados (IC50)
```bash
python compare_classifiers.py \
    --dataset human \
    --label-method ic50 \
    --label-threshold 1000 \
    --max-samples 1000
```

## 📈 Interpretando Resultados

### Arquivo: `comparison_table.txt`

```
Modelo                 Val F1  Val Acc  Test F1 Test Acc    Tempo
--------------------------------------------------------------------------------
MLP_Small              0.6400   0.7000   0.3333   0.5000    0.08s
RandomForest           0.4000   0.4000   0.4949   0.5000    0.16s
```

**Interpretação**:

1. **Val F1** (F1-Score Validação): Métrica principal para ranking
   - Melhor modelo: maior Val F1
   - Balanço entre precision e recall

2. **Val Acc** (Acurácia Validação): Proporção de predições corretas

3. **Test F1** / **Test Acc**: Métricas no conjunto **nunca visto**
   - ⚠️ Se Test << Val: possível **overfitting**
   - ✅ Se Test ≈ Val: modelo **generaliza bem**

4. **Tempo**: Tempo de treinamento
   - Importante para produção

### Arquivo: `classifier_comparison.json`

Contém detalhes completos:
```json
{
  "results": [
    {
      "name": "MLP_Small",
      "params": {
        "hidden_layers": [128, 64],
        "activation": "relu"
      },
      "train_time": 0.08,
      "train_accuracy": 0.55,
      "validation": {
        "accuracy": 0.70,
        "f1": 0.64,
        "roc_auc": 0.72
      },
      "test": {
        "accuracy": 0.50,
        "f1": 0.33,
        "roc_auc": 0.45
      }
    }
  ]
}
```

## 🎓 Escolhendo o Melhor Modelo

### Critérios de Seleção

1. **Val F1-Score**: Métrica principal (já ordenado automaticamente)

2. **Generalização** (Test ≈ Val):
   ```
   ✅ MLP_Small:  Val=0.64, Test=0.33  → Diferença 0.31
   ❌ XGBoost:    Val=0.51, Test=0.20  → Diferença 0.31 (pior)
   ```

3. **Tempo de Treinamento**:
   - Produção: considerar tempo
   - Pesquisa: focar em performance

4. **Complexidade**:
   - Simples: LogisticRegression, SVM_Linear
   - Intermediário: RandomForest, KNN
   - Complexo: MLP, XGBoost, GradientBoosting

## 🔬 Exemplo de Análise

### Cenário: 100 amostras, embeddings ESM-2

**Ranking**:
```
🥇 MLP_Small          (Val F1: 0.64) → VENCEDOR
🥈 GradientBoosting   (Val F1: 0.60)
🥉 MLP_Large          (Val F1: 0.56)
4. XGBoost            (Val F1: 0.51)
5. RandomForest       (Val F1: 0.40)
```

**Análise**:
- ✅ **MLP_Small é o melhor**: F1=0.64, rápido (0.08s)
- ⚠️ **GradientBoosting**: F1=0.60, mas 10x mais lento (1.01s)
- ⚠️ **RandomForest**: Overfitting? (Train=0.98, Val=0.40)

**Recomendação**:
→ Usar **MLP_Small** para este dataset

## 🛠️ Argumentos CLI

```bash
python compare_classifiers.py [OPÇÕES]

Dataset:
  --dataset {human,non_human,all}    Dataset a usar
  --max-samples INT                  Limitar amostras

Embeddings:
  --model MODEL                      Modelo ESM-2
  --device {cpu,cuda,auto}          Device

Labels:
  --label-method {pchembl,ic50,ki,kd,auto}
  --label-threshold FLOAT            Threshold customizado

Split:
  --val-size FLOAT                   Validação (default: 0.1)
  --test-size FLOAT                  Teste (default: 0.1)

Output:
  --output-dir DIR                   Diretório de saída
  --seed INT                         Random seed
  --quiet                            Modo silencioso
```

## 📊 Workflow Recomendado

### 1. Teste Rápido (Exploração)
```bash
# 100 amostras, modelo pequeno
python compare_classifiers.py \
    --max-samples 100 \
    --model esm2_t6_8M_UR50D
```
**Tempo**: ~5 minutos  
**Objetivo**: Identificar top 3 modelos

### 2. Validação Média (Refinamento)
```bash
# 1000 amostras, modelo médio
python compare_classifiers.py \
    --max-samples 1000 \
    --model esm2_t12_35M_UR50D
```
**Tempo**: ~30 minutos  
**Objetivo**: Confirmar melhor modelo

### 3. Produção Final (Treinamento)
```bash
# Dataset completo, modelo grande, GPU
python compare_classifiers.py \
    --dataset human \
    --model esm2_t36_3B_UR50D \
    --device cuda
```
**Tempo**: 2-4 horas  
**Objetivo**: Modelo final de produção

## 🔍 Troubleshooting

### Erro: KNN falhou
```
❌ ERRO: 'NoneType' object has no attribute 'split'
```
**Solução**: Bug conhecido, será corrigido. Não afeta outros modelos.

### Todos os modelos ruins (F1 < 0.5)
**Causas possíveis**:
1. Poucas amostras (< 1000)
2. Dataset desbalanceado
3. Labels incorretos
4. Embeddings de baixa qualidade

**Soluções**:
- Aumentar `--max-samples`
- Usar modelo ESM-2 maior
- Verificar `--label-method` e `--label-threshold`

### Overfitting (Train >> Val)
```
Train Acc: 0.98
Val Acc:   0.40
```
**Soluções**:
- Aumentar dataset
- Usar regularização
- Escolher modelo mais simples

## 🎯 Próximos Passos

Após identificar o melhor modelo:

1. **Retreinar com dataset completo** (Classification):
   ```bash
   python run_complete_pipeline.py \
       --dataset human \
       --model esm2_t36_3B_UR50D \
       --device cuda
   ```
   (Modifique o código para usar o classificador escolhido)

2. **OU Usar Regression Pipeline** (Predição Quantitativa):
   ```bash
   # Para predições de valores numéricos (Ki, Kd, IC50)
   python run_regression_pipeline.py \
       --dataset data/kinase_all.tsv \
       --activity-type ki \
       --models random_forest xgboost \
       --output-dir results/regression_ki
   ```
   Ver documentação em `src/regression/README_IMPROVEMENTS.md`

3. **Hyperparameter tuning**:
   - GridSearchCV
   - RandomizedSearchCV
   - Optuna

4. **Cross-validation**:
   - K-fold para estabilidade
   - Stratified K-fold

5. **Feature engineering**:
   - Diferentes métodos de pooling de embeddings
   - Combinação de features

---

## 🔄 Comparação: Classification vs Regression

### Quando usar Classification?
- ✅ Decisão binária: ATIVO/INATIVO
- ✅ Screening inicial de compostos
- ✅ Priorização de candidatos
- ✅ Análise exploratória rápida

### Quando usar Regression?
- ✅ Predição de valores exatos (Ki, Kd, IC50)
- ✅ Otimização quantitativa de compostos
- ✅ Análise de relação estrutura-atividade (SAR)
- ✅ Modelagem farmacocinética

### Dual Pipeline Workflow Recomendado:
```bash
# 1. Classification para screening inicial
python compare_classifiers.py --max-samples 10000

# 2. Regression para compostos ativos
python run_regression_pipeline.py \
    --dataset data/active_compounds.tsv \
    --activity-type ki \
    --models xgboost random_forest
```

---

## 📚 Referências

- **RandomForest**: Breiman (2001)
- **SVM**: Cortes & Vapnik (1995)
- **MLP**: Backpropagation neural networks
- **XGBoost**: Chen & Guestrin (2016)
- **GradientBoosting**: Friedman (2001)

---

**Autor**: DockTKinase Pipeline  
**Data**: Outubro 2025  
**Versão**: 2.0 - Dual Pipeline System  
**Sistema**: 17 modelos ML total (6 classifiers + 11 regressors)

````
