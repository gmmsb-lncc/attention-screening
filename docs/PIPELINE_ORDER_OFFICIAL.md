# Ordem Oficial do Pipeline DockTKinase

**Data:** 07 de Novembro de 2025  
**Branch:** regression  
**Validado:** ✅ Sim (test_pipeline_official_order.py)

---

## 🎯 Ordem Correta do Pipeline

### ❌ Ordem INCORRETA (antiga proposta)
```
dados → embeddings → estratificação → matriz → classificação
```

**Por que está errada:**
- Labels criados APÓS embeddings
- "Estratificação" confundida como preprocessamento
- "Matriz" como etapa separada (na verdade é concatenação de embeddings)

---

### ✅ Ordem OFICIAL (validada)

```
1. 📂 DADOS          → Carregar dataset (TSV)
2. 🏷️  LABELS         → Criar labels (pchembl >= 6.0)
3. 🧬 EMBEDDINGS     → Gerar embeddings (ESM-2 + SMI-TED)
4. 🔀 SPLIT          → Split estratificado (train/val/test)
5. 🤖 TREINO         → Treinar classificadores/regressores
6. 📊 VALIDAÇÃO      → Avaliar em conjunto de validação
7. 📊 TESTE          → Avaliar em conjunto de teste final
```

---

## 📋 Detalhamento de Cada Passo

### **Passo 1: DADOS**
**Função:** `load_dataset()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1375

```python
df = self.load_dataset()
```

**Input:**
- Arquivo TSV com colunas obrigatórias

**Colunas obrigatórias:**
- **Classificação:** `Ligand_SMILES`, `Target_Seq`, `Y`
- **Regressão:** `Ligand_SMILES`, `Target_Seq`, `Ki`, `Kd`, `IC50`

**Output:**
- DataFrame pandas com os dados

---

### **Passo 2: LABELS** ⭐ CRÍTICO - ANTES DE EMBEDDINGS!
**Função:** `create_labels()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1376

```python
y, df = self.create_labels(df)
```

**Regra para classificação:**
```python
# pchembl_value >= 6.0 → ativo (1)
# pchembl_value < 6.0  → inativo (0)
threshold = 6.0
y = (df['pchembl_value'] >= threshold).astype(int)
```

**Métodos disponíveis:**
- `'pchembl'`: Baseado em pchembl_value
- `'kd'`: Baseado em Kd
- `'ki'`: Baseado em Ki
- `'ic50'`: Baseado em IC50
- `'auto'`: Seleciona automaticamente

**Output:**
- `y`: Array numpy com labels
- `df`: DataFrame atualizado

**⚠️ IMPORTANTE:** Labels são criados ANTES de gerar embeddings!

---

### **Passo 3: EMBEDDINGS**
**Função:** `generate_embeddings()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1377

```python
X = self.generate_embeddings(df, batch_size=8)
```

**Modelos usados:**
1. **Proteínas:** ESM-2 (Meta AI)
   - Modelo padrão: `esm2_t33_650M_UR50D`
   - Modelo rápido: `esm2_t6_8M_UR50D` (recomendado para testes)
   - Output: 320 dimensões (8M) ou 1280 dimensões (650M)

2. **Ligantes:** SMI-TED (FM4M)
   - Modelo: `smi-ted-Light_40.pt`
   - Output: 384 dimensões

**Processo:**
1. Gerar embeddings de proteínas (ESM-2)
2. Gerar embeddings de ligantes (SMI-TED)
3. **Concatenar** embeddings: `[protein_emb | ligand_emb]`

**Output:**
- `X`: Array numpy (n_samples × n_features)
- Dimensões típicas:
  - ESM-2 (8M) + SMI-TED: 320 + 384 = **704 dims**
  - ESM-2 (650M) + SMI-TED: 1280 + 384 = **1664 dims**

---

### **Passo 4: SPLIT ESTRATIFICADO**
**Função:** `stratified_split()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1378

```python
X_train, X_val, X_test, y_train, y_val, y_test = self.stratified_split(X, y)
```

**Configuração padrão:**
- **Train:** 80%
- **Validation:** 10%
- **Test:** 10%

**Estratificação:**
- Mantém proporção de classes em cada split
- Validado com teste chi-quadrado
- Garante balanceamento

**Output:**
- 6 arrays: X_train, X_val, X_test, y_train, y_val, y_test

---

### **Passo 5: TREINO**
**Função:** `train_classifier()` ou `train_regressor()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1379

#### **Classificação:**
```python
clf = self.train_classifier(X_train, y_train, X_val, y_val)
```

**Modelos disponíveis (6):**
1. RandomForest
2. GradientBoosting
3. XGBoost
4. SVM
5. KNN
6. MLP (Neural Network)

#### **Regressão:**
```python
reg = self.train_regressor(X_train, y_train, X_val, y_val)
```

**Modelos disponíveis (11):**
1. RandomForest
2. GradientBoosting
3. XGBoost
4. LinearRegression
5. Ridge
6. Lasso
7. ElasticNet
8. SVR
9. KNN
10. DecisionTree
11. MLP (Neural Network)

**Output:**
- Modelo treinado (scikit-learn/xgboost object)

---

### **Passo 6: VALIDAÇÃO**
**Função:** `evaluate_classifier()` ou `evaluate_regressor()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1380

```python
val_metrics = self.evaluate_classifier(clf, X_val, y_val)
```

**Métricas de Classificação:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

**Métricas de Regressão:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Explained Variance

**Output:**
- Dicionário com todas as métricas

---

### **Passo 7: TESTE FINAL**
**Função:** `evaluate_classifier()` ou `evaluate_regressor()`  
**Arquivo:** `run_complete_pipeline.py` linha ~1381

```python
test_metrics = self.evaluate_classifier(clf, X_test, y_test)
```

**Output:**
- Métricas finais no conjunto de teste
- Mesmas métricas do passo de validação

---

## 🔍 Diferenças Principais

| Aspecto | Ordem Incorreta | Ordem Oficial |
|---------|----------------|---------------|
| **Labels** | Após embeddings | **ANTES** de embeddings ⭐ |
| **Estratificação** | Fase de preprocessamento | Split train/val/test |
| **Matriz** | Etapa separada | Concatenação automática em embeddings |
| **Validação** | Não mencionada | Etapa explícita (passo 6) |

---

## 📝 Código de Referência

### Pipeline Completo (Classificação)

```python
from run_complete_pipeline import CompletePipeline

# Inicializar pipeline
pipeline = CompletePipeline(
    dataset_path="data/kinase_data.tsv",
    protein_model="esm2_t6_8M_UR50D",  # Modelo rápido para testes
    label_method="pchembl",
    test_size=0.1,
    val_size=0.1,
    output_dir="pipeline_output"
)

# Executar pipeline completo (7 passos)
pipeline.run()
```

### Ordem de Execução Interna

```python
def run(self):
    # 1. DADOS
    df = self.load_dataset()
    
    # 2. LABELS (ANTES DE EMBEDDINGS!)
    y, df = self.create_labels(df)
    
    # 3. EMBEDDINGS
    X = self.generate_embeddings(df, batch_size=8)
    
    # 4. SPLIT
    X_train, X_val, X_test, y_train, y_val, y_test = self.stratified_split(X, y)
    
    # 5. TREINO
    clf = self.train_classifier(X_train, y_train, X_val, y_val)
    
    # 6. VALIDAÇÃO
    val_metrics = self.evaluate_classifier(clf, X_val, y_val)
    
    # 7. TESTE
    test_metrics = self.evaluate_classifier(clf, X_test, y_test)
    
    return {
        'model': clf,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
```

---

## 🧪 Validação

**Arquivo de teste:** `tests/test_pipeline_official_order.py`

**Status:** ✅ VALIDADO

**Execução:**
```bash
source env/bin/activate
python tests/test_pipeline_official_order.py
```

**Resultado:**
- ✅ Pipeline executado com ordem correta
- ✅ Labels criados ANTES de embeddings
- ✅ Split estratificado funcionando
- ✅ 3 modelos treinados com sucesso
- ✅ Avaliação em validação e teste

---

## 📚 Referências

**Arquivos principais:**
- `run_complete_pipeline.py`: Pipeline oficial de classificação
- `run_regression_pipeline.py`: Pipeline oficial de regressão
- `src/build/pipeline/build_pipeline.py`: Pipeline modular
- `docs/PIPELINE_GUIDE.md`: Guia completo do pipeline
- `docs/EXECUTION_GUIDE.md`: Guia de execução

**Função `run()` completa:**
- Arquivo: `run_complete_pipeline.py`
- Linhas: 1373-1473
- Validado em: 07/11/2025

---

## ⚠️ Erros Comuns a Evitar

1. **❌ Criar labels APÓS embeddings**
   - Labels devem ser criados ANTES para manter integridade

2. **❌ Confundir estratificação com preprocessamento**
   - Estratificação = split train/val/test
   - Não é uma etapa de transformação de dados

3. **❌ Criar matriz separadamente**
   - Matriz é a concatenação automática de embeddings
   - Não precisa de etapa separada

4. **❌ Pular validação**
   - Validação é essencial antes do teste final
   - Ajuda a detectar overfitting

---

## ✅ Checklist de Validação

Ao implementar ou modificar o pipeline, verifique:

- [ ] Labels criados ANTES de embeddings
- [ ] Embeddings gerados (proteína + ligante)
- [ ] Split estratificado (mantém proporção de classes)
- [ ] Conjunto de validação separado do teste
- [ ] Avaliação em validação ANTES do teste final
- [ ] Métricas apropriadas calculadas
- [ ] Resultados salvos corretamente

---

**Última atualização:** 07 de Novembro de 2025  
**Validado por:** test_pipeline_official_order.py (✅ PASSOU)
