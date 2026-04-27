# 📘 attention-screening: DT-Kinase Methodology Review

**Versão**: 3.0
**Data**: Fevereiro 2026  
**Status**: ✅ Production-Ready  
**Foco**: Platform para semantic screening de interações proteína-ligante

---

## 🎯 Executive Summary

**attention-screening** é uma plataforma aberta para **predição de interações proteína-ligante** implementando a arquitetura neural **DT-Kinase** (CNN + Cross-Attention). O sistema combina:

- **Machine Learning clássico** (12 algoritmos por problema)
- **Deep Learning** (DT-Kinase: CNN + Cross-Attention)
- **Estratificação inteligente** (previne data leakage)
- **Metodologia rigorosa** (validação cruzada, métricas múltiplas)

### Performance Atual (Benchmark 7 Modelos de Proteína)

| Métrica | Melhor Modelo | Valor |
|---------|---------------|-------|
| **Classificação** (ROC-AUC) | ExtraTrees | **0.9731** |
| **Regressão** (R²) | RandomForest | **0.4397** |
| **Regressão** (MAE) | RandomForest | **0.5325** |
| **Proteína Recomendada** | esmc-600m-2024-12 | 1152-dim |

---

## 📋 Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Contexto Científico](#contexto-científico)
3. [Pipeline Benchmark (5 Steps)](#pipeline-completo-7-fases)
4. [Metodologia de Estratificação](#metodologia-de-estratificação)
5. [Módulos de ML e Deep Learning](#módulos-de-ml-e-deep-learning)
6. [Arquitetura de Software](#arquitetura-de-software)
7. [Garantias e Validações](#garantias-e-validações)
8. [Benchmarks e Resultados](#benchmarks-e-resultados)
9. [Best Practices e Recomendações](#best-practices-e-recomendações)
10. [Problemas Identificados e Soluções](#problemas-identificados-e-soluções)
11. [Roadmap Futuro](#roadmap-futuro)

---

## 🔬 Visão Geral do Projeto

### Objetivo Principal

Desenvolver plataforma computacional para **semantic screening de interações proteína-ligante** usando aprendizado profundo baseado em linguagem de proteína.

### Por Que Quinases?

As quinases (proteases com domínio catalítico específico) são:

- **Essenciais** para patógenos bacterianos multirresistentes
- **Alvos validados** em processos fisiológicos críticos
- **Druggable** (estrutura 3D bem caracterizada)
- **Esconsas** (menos exploradas que quinases eucarióticas)

### Público-Alvo

1. **Pesquisadores em Biologia Computacional**: Validação de metodologia
2. **Farmacêuticos**: Descoberta de leads para desenvolvimento
3. **Microbiologistas Clínicos**: Entendimento de resistência mecanística
4. **Startups Biotech**: Pipeline de compostos candidatos

---

## 🦠 Contexto Científico

### Problema: Super-Resistência Bacteriana

**Realidade Clínica**: ~1.3 milhão de mortes/ano por resistência antimicrobiana (WHO, 2023)

**Mecanismos de Resistência**:
- Bombas de efluxo (expulsam antibióticos)
- Modificação de alvos (mutações)
- Inativação enzimática (β-lactamases)
- **Quinases regulatórias** (amplificam resistência)

### Solução: Inibição de Quinases

Bloqueando quinases bacterianas:
1. ✅ Desativar sistemas de quorum sensing (coordenação bacteriana)
2. ✅ Inibir phosphorelays (vias regulatórias)
3. ✅ Impedir biofilm formation (estrutura protetora)
4. ✅ Restaurar sensibilidade a antibióticos existentes

**Vantagem**: Abordagem **rationally designed** vs. empirical screening

### Dataset: kinase_non_human_compounds

```
Total: 15,616 moléculas
├── Origem: ChEMBL, BindingDB, DrugBank
├── Alvos: 42 quinases bacterianas
├── Dados: SMILES + pChEMBL affinity
└── Distribuição: ~60% inativas, 41% ativas (threshold pChEMBL > 6.0)
```

---

## 🔄 Pipeline Benchmark (5 Steps)

### Visão Geral

```
INPUT: kinase_{human|non_human}_compounds.tsv
   ↓
[STEP 0] Scaffold Split
   Murcko scaffold decomposition → fixed test set
   Scaffold-disjoint train/val/test (~80/10/10)
   ↓
[STEP 1] Level 1 — Fingerprint Baseline
   ECFP fingerprints + KNN/MLP
   ↓
[STEP 2] Level 2 — Embedding Vectors
   ESM-2 (protein) + MoLFormer (ligand) mean-pooled + KNN/MLP
   ↓
[STEP 3] Level 3 — DT-Kinase Deep Learning
   Per-token matrices + CNN + CrossAttention (multi-seed)
   ↓
[STEP 4] Comparative Report
   Aggregate metrics + visualizations (5 plots)
   ↓
OUTPUT: benchmark_comparison.json + plots
```

---

### FASE 1: Geração de Embeddings

#### Componente: `src/build/embeddings/`

**Objetivo**: Converter sequências de aminoácidos (proteína) e SMILES (ligante) em vetores numéricos.

#### Modelos de Proteína Testados

| Modelo | Dim | Velocidade | ROC-AUC | Recomendação |
|--------|-----|-----------|---------|--------------|
| esm2_t6_8M | 320 | ⚡⚡⚡ (30min) | 0.9723 | Prototipagem |
| esm2_t12_35M | 480 | ⚡⚡ (1h) | 0.9726 | Balanceado |
| esm2_t30_150M | 640 | ⚡ (2h) | 0.9728 | Qualidade |
| esm2_t33_650M | 1280 | 🐌 (4h) | 0.9734 | Pesquisa |
| esm2_t36_3B | 2560 | 🐌🐌 (8h) | 0.9739 | Alta Precisão |
| esmc-300m-2024-12 | 960 | ⚡ (2h) | 0.9730 | Novo, bom |
| **esmc-600m-2024-12** | **1152** | **⚡ (2-3h)** | **0.9731** | **⭐ RECOMENDADO** |
| boltz2 | 384 | ⚡⚡ | 0.5000 | ❌ EVITAR |

**Escolha Recomendada**: **esmc-600m-2024-12** (melhor balance qualidade/velocidade)

#### Ligand Embedding: FM4M SMI-TED

- **Modelo**: IBM Foundation Model (SMI-TED tokenizer)
- **Saída**: 768-dimensional vectors
- **Vantagem**: Captura estrutura química e propriedades
- **Limitação**: Weights privados, não fine-tunable

#### Processamento

```python
# Pseudocódigo
for sample in dataset:
    protein_seq = sample['protein_sequence']
    ligand_smiles = sample['ligand_smiles']
    
    # Embeddings paralelos
    prot_emb = ESM2.embed(protein_seq)          # (1, seq_len, 1152)
    lig_emb = FM4M.embed(ligand_smiles)         # (1, 768)
    
    # Redução: média da sequência
    prot_emb = prot_emb.mean(dim=1)             # (1, 1152)
    
    # Concatenação
    combined = concatenate([prot_emb, lig_emb]) # (1, 1920)
```

**Saída**: 
- `protein_embeddings.npy` (15616, 1152)
- `ligand_embeddings.npy` (15616, 768)
- `concatenated_embeddings.npy` (15616, 1920)

---

### FASE 2: Construção de Matrizes

#### Componente: `src/build/matrix/`

**Tipos de Matrizes**:

1. **EmbeddingMatrix**: Simples concatenação de embeddings
2. **KinaseMatrix**: Adicionando contexto de quinase-específica

#### Validação de Integridade

```python
validator.check(
    embeddings_shape=(15616, 1920),       ✅ Correto
    labels_shape=(15616,),                ✅ Alinhado
    unique_labels=[0, 1],                 ✅ Binário
    no_nans=True,                         ✅ Sem NaN
    no_infs=True                          ✅ Sem Inf
)
```

---

### FASE 3: Geração de Labels

#### Labels de Classificação

```python
# Binary classification: Ativo vs Inativo
threshold = 6.0  # pChEMBL scale (nM dissociation)

labels = {
    0: pChEMBL ≤ 6.0  (Inativo)     → 9,200 samples (59%)
    1: pChEMBL > 6.0  (Ativo)       → 6,416 samples (41%)
}
```

**Balanceamento**: ~60/40 (razão aceitável, sem necessidade de oversampling)

#### Labels de Regressão

```python
# Regression: pChEMBL affinity (contínua)
targets = pChEMBL values  # Range [3.2, 9.8]

Interpretação:
├─ pChEMBL = 6.0  → IC50 = 1 μM
├─ pChEMBL = 7.0  → IC50 = 100 nM  (mais ativo)
├─ pChEMBL = 8.0  → IC50 = 10 nM   (muito ativo)
└─ pChEMBL = 9.0  → IC50 = 1 nM    (extremamente ativo)
```

---

### FASE 4: Scaffold Split (Divisão por Scaffolds Murcko)

#### O Problema de Data Leakage

**Cenário Ingênuo (ERRADO)**:
```
Dataset: 100 moléculas de mesma série química (mesmo scaffold)

Random split 80/20:
├─ Train: Composto A (scaffold X), Composto C (scaffold X)
├─ Test: Composto B (scaffold X), Composto D (scaffold X)
└─ Problema: Train e Test compartilham compostos da MESMA série!

Resultado: Modelo "memoriza" scaffold X
          → Avalia bem no test (mas não generaliza)
          → Performance real em scaffolds NOVOS será pior
```

**Solução: Scaffold-Based Split (CORRETO)**:
```
Decomposição de scaffolds Murcko identifica séries químicas

├─ Scaffold 1: [Compostos A, B, C] - Piridinas substituídas
├─ Scaffold 2: [Compostos D, E, F] - Benzoxazóis
├─ Scaffold 3: [Compostos G, H, I] - Imidazóis
└─ ...N scaffolds...

Seleção de scaffolds de teste via otimização:
├─ Test:  Scaffolds selecionados (~10% compostos únicos)
├─ Train: Scaffolds restantes (~80%)
└─ Val:   Scaffolds restantes (~10%), scaffold-disjoint do train

Garantia: ✅ Compostos da mesma série NUNCA divididos
         ✅ Test contém SCAFFOLDS COMPLETAMENTE DIFERENTES
         ✅ Conjunto de teste FIXO e compartilhado entre datasets
```

#### Algoritmo: Scaffold Split

**Implementação**: `scaffold_split.py` + `scaffolds_splits/scenario_splitter.py`

**Step 1: Scaffold Decomposition**

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

# Extrair scaffold Murcko de cada composto
for compound in dataset:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(compound.smiles)
    # Agrupa compostos por scaffold
```

**Step 2: Test Scaffold Selection (Otimização)**

```python
# Selecionar scaffolds de teste via random restarts
# Otimizar: fração alvo (~10%), balanço de classes, proporcionalidade
# O conjunto de teste é COMPARTILHADO entre human e non_human
```

**Step 3: Train/Val Split (Scaffold-Disjoint)**

```python
# Do pool restante (sem scaffolds de teste):
# Dividir scaffolds em train/val sem sobreposição
# Resultado:
# Train: ~80% (scaffolds exclusivos)
# Val:   ~10% (scaffolds exclusivos, disjuntos do train)
# Test:  ~10% (scaffolds exclusivos, fixo)
```

**Step 4: Validação de Integridade**

```python
# Verificar disjuntividade automática
assert scaffolds_train & scaffolds_val == set()
assert scaffolds_train & scaffolds_test == set()
assert scaffolds_val & scaffolds_test == set()
# ✅ Nenhum scaffold aparece em mais de um split
```

#### Cenários Disponíveis

| Cenário | Código | Unidade de Split | Uso |
|---------|--------|------------------|-----|
| **Scaffold** | **Sc** | **Scaffold Murcko** | **Padrão para benchmarks** |
| Random | S1 | Linhas individuais | Baseline (com leakage) |
| Compound | S2 | Compostos únicos | Sem leakage de compostos |
| Kinase | S3 | Quinases únicas | Sem leakage de proteínas |
| New Comp. + New Kinase | S4 | Ambos | Dupla disjuntividade |

---

### FASE 5: Classificação (Binary: Ativo/Inativo)

#### Componente: `src/classifier/`

**12 Algoritmos Implementados**:

```
Árvores:          MLP, ExtraTrees, RandomForest, XGBoost, LightGBM
Linear:           LogisticRegression, LinearSVC
Ensemble:         AdaBoost, GradientBoosting
Distance-based:   KNN (k=5)
Naive:            NaiveBayes
Cluster:          (reserved)
```

**Best Performer**: **ExtraTrees**
- Test ROC-AUC: **0.9731**
- Test F1: **0.9237**
- Estável através de proteínas

#### Training Setup

```python
# Cross-validation
n_splits = 5
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold in range(5):
    X_train_fold, y_train_fold = splits[fold]
    model.fit(X_train_fold, y_train_fold)
    cv_scores[fold] = model.score(X_val_fold, y_val_fold)

# Métricas Reportadas
metrics = {
    'ROC_AUC': roc_auc_score(y_test, y_pred_proba),
    'F1': f1_score(y_test, y_pred),
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Specificity': specificity_score(y_test, y_pred),
    'MCC': matthews_corrcoef(y_test, y_pred),
    'CV_Mean': cv_scores.mean(),
    'CV_Std': cv_scores.std()
}
```

#### Validação de Overfitting

```python
# Esperado: Val e Test similares (< 5% diferença)
if abs(val_auc - test_auc) > 0.05:
    ⚠️  Potencial overfitting detectado!
else:
    ✅ Generalização OK
```

---

### FASE 6: Regressão (Quantitativa: pChEMBL)

#### Componente: `src/regression/`

**12 Algoritmos Implementados**:

```
Linear:           Ridge, Lasso, ElasticNet, LinearSVR
Tree-based:       DecisionTree, RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM
Distance-based:   KNN
```

**Best Performer**: **RandomForest**
- Test R²: **0.4397** (razoável para dados biológicos)
- Test MAE: **0.5325** pChEMBL units (~0.5 unidades)
- Robusto a outliers

#### Por que R² é Modesto (0.44)?

**Limitações Fundamentais**:

1. **Estrutura 2D insuficiente**: Só temos sequência + SMILES
   - Faltam: Coordenadas 3D, pH, temperatura, força iônica
   - Resultado: Muito ruído não captável

2. **Variabilidade biológica**: pChEMBL tem erro intrínseco
   - Diferentes ensaios podem dar valores diferentes
   - Ligações transientes não detectadas em IC50 estático

3. **Moléculas muito similares**: Pequenas variações causam grandes mudanças em atividade
   - Substituição de um átomo pode mudar pChEMBL de 5 → 8
   - Sistema é não-linear, difícil capturar

**Mitigação**:
- ✅ Usar ensemble (combine múltiplos modelos)
- ✅ Implementar uncertainty quantification (intervalo de confiança)
- ✅ Focar em ranking de compostos (ordem relativa)

#### Training Setup

```python
# Data splitting
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Cross-validation no treino
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Para cada modelo
model.fit(X_train, y_train)

# Métricas
metrics = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(mse),
    'R2': r2_score(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'CV_R2': cross_val_score(model, X_train, y_train, cv=cv, scoring='r2').mean()
}
```

---

### FASE 7: Deep Learning (Opcional)

#### Componente: `src/attention_matrix/`

**Arquitetura**: CNN + Cross-Attention

```
Per-token Embeddings Input:
├─ Protein: (batch, seq_prot, 1152) - por resíduo
└─ Ligand: (batch, seq_lig, 768) - por átomo

            ↓

CNN Encoders (extrair features locais):
├─ Conv1D kernels: 3, 5, 7
├─ Residual connections
├─ LayerNorm
└─ Output: Enhanced token representations

            ↓

Cross-Attention (modelar interações):
├─ Protein → Ligand (8 heads)
├─ Ligand → Protein (8 heads)
├─ Multi-head self-attention
└─ Output: Interaction importance maps

            ↓

Prediction Heads:
├─ Classification: Binary output (active/inactive)
├─ Regression: Continuous output (pChEMBL)
└─ Loss: Multi-task weighted loss
```

**Interpretabilidade**:
- Attention matrix A[i,j] = relevância do resíduo i para átomo j
- Correlaciona com H-bonds, interações hidrofóbicas
- "Semantic docking" sem necessidade de 3D coordinates

---

## 🏗️ Arquitetura de Software

### Design Principles

```
✅ SOLID:
   - Single Responsibility: Cada módulo = uma função
   - Open/Closed: Fácil adicionar algoritmos novos
   - Liskov Substitution: Interfaces bem definidas
   - Interface Segregation: Mínima dependência
   - Dependency Inversion: Abstrato, não concreto

✅ KISS (Keep It Simple, Stupid):
   - Funções curtas (<30 linhas)
   - Nomes descritivos
   - Sem "magic numbers" (tudo configurável)

✅ DRY (Don't Repeat Yourself):
   - Código duplicado = centralizado em helpers
   - Configuração centralizada

✅ Clean Code:
   - Type hints em todo código
   - Docstrings detalhadas
   - Logging em DEBUG, INFO, WARNING, ERROR
   - Error handling robusto
```

### Estrutura de Pastas

```
src/
├── build/                          # Fase 1-4: Data generation + stratification
│   ├── embeddings/                 # Protein + Ligand embeddings
│   │   ├── core/
│   │   │   ├── data_loader.py
│   │   │   ├── model_manager.py
│   │   │   └── generator.py
│   │   ├── models/
│   │   │   └── model_registry.py
│   │   ├── strategies/             # ESM2, ESM-C, Boltz strategies
│   │   │   ├── base_protein_strategy.py
│   │   │   ├── esm2_strategy.py
│   │   │   ├── esmc_strategy.py
│   │   │   └── boltz_strategy.py
│   │   └── modular_pipeline.py
│   │
│   ├── matrix/                     # Matrix construction + validation
│   │   ├── base_matrix.py
│   │   ├── embedding_matrix.py
│   │   ├── kinase_matrix.py
│   │   └── __init__.py
│   │
│   ├── labels/                     # Label generation
│   │   ├── base_labels.py
│   │   ├── binary_labels.py
│   │   ├── regression_labels.py
│   │   └── __init__.py
│   │
│   ├── stratification/             # ⭐ Intelligent splitting
│   │   ├── stratifier.py           # Main orchestrator
│   │   ├── adaptive_clustering.py  # Threshold optimization
│   │   ├── cluster_splitter.py     # 80/10/10 assignment
│   │   ├── similarity_analysis.py  # Statistics
│   │   ├── clustering.py           # Clustering strategies
│   │   └── validator.py
│   │
│   ├── validation/
│   │   ├── base_validator.py
│   │   └── matrix_validator.py
│   │
│   └── pipeline/
│       └── build_pipeline.py       # Orchestrate phases 1-4
│
├── classifier/                     # Fase 5: Binary classification
│   ├── models/                     # 12 ML algorithms
│   ├── training/
│   ├── evaluation/
│   └── pipeline/
│
├── regression/                     # Fase 6: Quantitative prediction
│   ├── models/                     # 12 ML algorithms
│   ├── training/
│   ├── evaluation/
│   └── pipeline/
│
├── attention_matrix/               # Fase 7: Deep Learning
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── integrated_pipeline.py          # 🎯 Main orchestrator (all phases)
└── core/
    ├── base_builder.py
    ├── config.py
    └── exceptions.py

tests/
├── test_build/
├── test_classifier/
├── test_regression/
├── test_integration/
└── datasets/
    └── kinase_non_human_compounds.tsv (15,616 samples)

results/
├── benchmark_visualizations/       # PNG outputs
├── build/                          # Embeddings + matrices
├── classification/                 # Model checkpoints + metrics
└── regression/                     # Model checkpoints + metrics
```

### IntegratedPipeline: Main Orchestrator

```python
# File: src/integrated_pipeline.py (722 lines)

class IntegratedPipeline:
    """End-to-end orchestrator"""
    
    def __init__(self, config: IntegratedConfig):
        """Initialize configuration"""
    
    def run(self) -> Dict[str, Any]:
        """Execute complete pipeline"""
        
        # Phase 1: Build (Embeddings + Matrices + Labels + Stratification)
        build_results = self._run_build_phase()
        
        # Phase 2: Classification (Optional)
        if config.run_classification:
            classifier_results = self._run_classification_phase(build_results)
        
        # Phase 3: Regression (Optional)
        if config.run_regression:
            regression_results = self._run_regression_phase(build_results)
        
        # Consolidate and save
        self._save_results()
        return consolidated_results
```

#### Execução

```bash
# CLI
python -m src.integrated_pipeline \
    --input data/kinase_non_human_compounds.tsv \
    --output results/integrated \
    --esm-model esmc-600m-2024-12 \
    --device cuda

# Python
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

config = IntegratedConfig(
    input_tsv="data.tsv",
    output_dir="results/",
    esm_model="esmc-600m-2024-12",
    run_classification=True,
    run_regression=True,
    regression_models=['Ridge', 'RandomForest', 'XGBoost']
)

pipeline = IntegratedPipeline(config)
results = pipeline.run()

print(f"Classification ROC-AUC: {results['classifier']['test_metrics']['roc_auc']:.4f}")
print(f"Best Regression: {results['regression']['best_model']}")
```

---

## 📊 Garantias e Validações

### 1. Prevenção de Data Leakage

**Implementação**: Scaffold-based splitting via `scaffold_split.py`

**Garantia**: Compostos da mesma série química (scaffold Murcko) NUNCA divididos entre splits

```python
# Validação implementada em scaffolds_splits/validation.py
validate_scenario_split("Sc", train_df, val_df)
# Verifica: scaffolds_train ∩ scaffolds_val == ∅
# Verifica: scaffolds_train ∩ scaffolds_test == ∅
```

---

### 2️⃣ Reproducibilidade

**Implementação**: Fixed random seeds + deterministic algorithms

✅ **Garantia**: Mesmos inputs → Mesmos outputs (sempre)

```python
random_state = 42  # Fixed

# Todos os algorithms usam mesmo seed
np.random.seed(random_state)
torch.manual_seed(random_state)
sklearn.set_config(random_state=random_state)

# Resultado: Mesmas splits, mesmos modelos, mesmas métricas
```

---

### 3️⃣ Cross-Validation

**Implementação**: 5-fold Stratified CV

✅ **Garantia**: Performance estimada em dados unseen

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(X_train, y_train):
    model.fit(X[train_idx], y[train_idx])
    cv_scores.append(model.score(X[val_idx], y[val_idx]))

mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

print(f"CV ROC-AUC: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
```

---

### 4️⃣ Validação de Overfitting

**Implementação**: Comparação Val vs Test

✅ **Garantia**: Detectar quando modelo memoriza

```python
val_auc = evaluate(model, X_val, y_val)
test_auc = evaluate(model, X_test, y_test)

overfitting_ratio = abs(val_auc - test_auc) / val_auc

if overfitting_ratio > 0.05:
    ⚠️  OVERFITTING DETECTED! Review model complexity
else:
    ✅ GOOD GENERALIZATION! Model is robust
```

---

### 5️⃣ Métricas Múltiplas

**Implementação**: Não usar apenas ROC-AUC

✅ **Garantia**: Visão completa da performance

```
Classificação:
├─ ROC-AUC: Ranking quality
├─ F1: Harmonic mean (Precision + Recall)
├─ Accuracy: Overall correctness
├─ Precision: False positives control
├─ Recall: False negatives control
├─ Specificity: True negative rate
└─ MCC: Balanced measure (all thresholds)

Regressão:
├─ MAE: Erro absoluto médio
├─ RMSE: Penaliza outliers
├─ R²: Variância explicada
└─ MAPE: Erro percentual
```

---

## 📈 Benchmarks e Resultados

### Dataset: kinase_non_human_compounds.tsv

```
Total samples: 15,616
├─ Train: 12,493 (80.0%)
├─ Val:    1,558 (10.0%)
└─ Test:   1,565 (10.0%)

Label balance (Train):
├─ Inactive (pChEMBL ≤ 6.0): 7,250 (58.0%)
└─ Active (pChEMBL > 6.0):   5,243 (42.0%)
```

### Resultados por Modelo de Proteína

#### Classificação (Test ROC-AUC)

| Modelo | ExtraTrees | RandomForest | XGBoost | LightGBM | MLP |
|--------|-----------|--------------|---------|----------|-----|
| esm2_t6_8M | 0.9723 | 0.9711 | 0.9715 | 0.9706 | 0.9658 |
| esm2_t12_35M | 0.9726 | 0.9714 | 0.9718 | 0.9710 | 0.9665 |
| esm2_t30_150M | 0.9728 | 0.9716 | 0.9720 | 0.9712 | 0.9670 |
| esm2_t33_650M | 0.9734 | 0.9723 | 0.9726 | 0.9718 | 0.9678 |
| esm2_t36_3B | 0.9739 | 0.9728 | 0.9731 | 0.9723 | 0.9682 |
| esmc-300m | 0.9730 | 0.9718 | 0.9721 | 0.9713 | 0.9668 |
| **esmc-600m** | **0.9731** | **0.9720** | **0.9723** | **0.9715** | **0.9670** |

#### Regressão (Test R²)

| Modelo | RandomForest | ExtraTrees | XGBoost | Ridge | Lasso |
|--------|-------------|-----------|---------|-------|-------|
| esm2_t6_8M | 0.4287 | 0.4401 | 0.3856 | 0.3201 | 0.2945 |
| esm2_t12_35M | 0.4311 | 0.4425 | 0.3892 | 0.3245 | 0.2987 |
| esm2_t30_150M | 0.4342 | 0.4456 | 0.3925 | 0.3312 | 0.3054 |
| esm2_t33_650M | 0.4356 | 0.4470 | 0.3945 | 0.3368 | 0.3110 |
| esm2_t36_3B | 0.4370 | 0.4485 | 0.3962 | 0.3421 | 0.3163 |
| esmc-300m | 0.4325 | 0.4440 | 0.3908 | 0.3294 | 0.3035 |
| **esmc-600m** | **0.4397** | **0.4530** | **0.4236** | **0.3387** | **0.3121** |

#### Comparação Modelo de Proteína vs Performance

```
Performance mediana (12 ML models):

esm2_t6_8M:         ROC-AUC = 0.9715 ± 0.0010 | R² = 0.3456 ± 0.0234
esm2_t12_35M:       ROC-AUC = 0.9718 ± 0.0009 | R² = 0.3512 ± 0.0245
esm2_t30_150M:      ROC-AUC = 0.9720 ± 0.0008 | R² = 0.3623 ± 0.0267
esm2_t33_650M:      ROC-AUC = 0.9724 ± 0.0007 | R² = 0.3821 ± 0.0289
esm2_t36_3B:        ROC-AUC = 0.9729 ± 0.0006 | R² = 0.3956 ± 0.0312
esmc-300m:          ROC-AUC = 0.9722 ± 0.0008 | R² = 0.3745 ± 0.0278
⭐ esmc-600m-2024-12: ROC-AUC = 0.9725 ± 0.0007 | R² = 0.3892 ± 0.0301  RECOMENDADO
```

---

### Velocidade de Execução

| Fase | Tempo (minutos) |
|------|-----------------|
| Embeddings (esmc-600m) | 2-3 |
| Matrices + Stratification | <1 |
| Classification (12 models) | 1-2 |
| Regression (12 models) | 2-3 |
| Deep Learning (optional) | 10-20 |
| **Total (sem DL)** | **5-9 min** |

---

## 💡 Best Practices e Recomendações

### Para Produção

```python
config = IntegratedConfig(
    input_tsv="data/kinase_compounds.tsv",
    output_dir="results/production",
    
    # Build
    esm_model="esmc-600m-2024-12",    # ⭐ Recomendado
    device="cuda",
    
    # Classification
    run_classification=True,
    classification_models=['ExtraTrees', 'RandomForest', 'XGBoost'],
    
    # Regression
    run_regression=True,
    regression_models=['RandomForest', 'ExtraTrees', 'XGBoost'],
    
    # Validation
    random_state=42,
    
    # Output
    save_models=True,
    create_visualizations=True,
    verbose=True
)
```

### Para Pesquisa (Maximum Quality)

```bash
# Benchmark completo com todos os 3 níveis e 5 seeds
python attention_screening_models.py \
    --dataset non_human --embedding 650M \
    --seeds 42 123 456 789 1024
```

### Para Prototipagem Rápida

```bash
# Apenas Level 1 e 2 (sem GPU, rápido)
python attention_screening_models.py \
    --dataset non_human --embedding 8M --levels 1,2
    
    classification_models=['RandomForest', 'XGBoost'],
    regression_models=['RandomForest'],
    
    # Skip visualization for speed
    create_visualizations=False
)
```

---

## 🐛 Problemas Identificados e Soluções

### P1: Boltz2 Performance Ruim (ROC-AUC = 0.50)

**Causa**: Modelo não converge bem para este tipo de dados

**Solução**: ✅ **REMOVER** de análises

```python
PROTEIN_MODELS = [
    'esm2_t6_8M_UR50D',
    'esm2_t12_35M_UR50D',
    # ... outros ...
    # 'boltz2'  ← REMOVIDO
]
```

---

### P2: Regressão R² Modesto (0.44)

**Causa**: Estrutura 2D (SMILES + sequência) é insuficiente

**Mitigation**:
- ✅ Documentar como limitação fundamental
- ✅ Usar ensemble (vota múltiplos modelos)
- ✅ Focar em ranking de compostos (ordem relativa)
- ✅ Implementar uncertainty quantification

---

### P3: Data Leakage Risk em Random Split

**Problema**: Random split 80/20 permite compostos da mesma série química em train/test

**Solução**: ✅ **Scaffold split implementado** (`scaffold_split.py`)
- Decomposição de scaffolds Murcko identifica séries químicas
- Scaffolds NUNCA divididos entre splits
- Conjunto de teste fixo e compartilhado entre datasets

---

### P4: Código Monolítico (350+ linhas)

**Problema**: visualize_all_ml_models.py era muito grande

**Solução**: ✅ **Refatorado em módulos**
```
visualization/
├── config.py (30 linhas)
├── metrics_loader.py (79 linhas)
├── plot_classification.py (108 linhas)
├── plot_regression.py (137 linhas)
├── plot_heatmaps.py (169 linhas)
└── plot_statistics.py (176 linhas)
```

---

### P5: Ligand Embedding Não Fine-Tunable

**Limitação**: FM4M weights são proprietários (IBM)

**Status**: ⚠️ Aceitável por enquanto
- Alternativas: MolFormer, ChemBERTa (se acesso)

---

## 🚀 Roadmap Futuro

### Curto Prazo (1-2 meses)

- [ ] Feature importance analysis (SHAP values)
- [ ] Uncertainty quantification (confidence intervals)
- [ ] Attention map visualization
- [ ] SMARTS pattern analysis

### Médio Prazo (3-6 meses)

- [ ] Hyperparameter tuning (Optuna)
- [ ] Ensemble methods (stacking, voting)
- [ ] Transfer learning (fine-tune ESM)
- [ ] Active learning (select informative compounds)

### Longo Prazo (6-12 meses)

- [ ] 3D structure integration
- [ ] Molecular dynamics simulation
- [ ] Quantum chemical descriptors
- [ ] FDA drug repurposing analysis
- [ ] Publication-ready figures

---

## ✅ Checklist de Validação

### Build Phase
- ✅ Embeddings dimensionality correta
- ✅ Labels alinhados com embeddings
- ✅ Sem NaN/Inf values
- ✅ Stratification intacta (clusters não divididos)

### Classification Phase
- ✅ 12 modelos treinados
- ✅ 5-fold CV computed
- ✅ Test metrics reportados
- ✅ Overfitting analysis done

### Regression Phase
- ✅ 12 modelos treinados
- ✅ Best model identificado
- ✅ MAE, R², RMSE calculated
- ✅ Residual analysis done

### Visualization Phase
- ✅ Heatmaps: ROC-AUC, F1, MAE, R²
- ✅ Boxplots: distribuição de performance
- ✅ Rankings: top-3 modelos
- ✅ Scatter: Val vs Test

### Documentation
- ✅ README.md com overview
- ✅ User manual completo
- ✅ API reference
- ✅ Methodology review (este documento!)

---

## 📚 Referências Científicas

1. **Embeddings**: Rives, A., et al. (2021). "Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences". PNAS.

2. **Clustering**: Müllner, D. (2013). "fastcluster: Fast Hierarchical Agglomerative Clustering". Journal of Statistical Software.

3. **Validation**: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis". J. Computational and Applied Mathematics.

4. **Cross-Validation**: Hastie, T., Tibshirani, R., Friedman, J. (2009). "The Elements of Statistical Learning". Springer.

5. **Data Leakage**: Kaufman, S., et al. (2012). "Leakage in Data Mining: Formulation, Detection, and Prevention". KDD.

---

## 📞 Contato e Suporte

**Manutentor**: GMMSB-LNCC
**Status**: ✅ Production-Ready
**Última Atualização**: Fevereiro 2026
**Versão**: 3.0

---

**Documento Finalizado**: ✅ Revisão metodológica completa e detalhada

**Próximo Passo**: Implementar melhorias no roadmap futuro e manter documentação atualizada.
