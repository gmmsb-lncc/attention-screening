# DockTKinase (semantic-screening) — Relatório Completo de Fluxo de Trabalho

**Autor**: Claude/Copilot  
**Data**: Janeiro 2025  
**Versão**: 2.1  
**Destinatário**: Leitor leigo em bioinformática e aprendizado de máquina

---

## Sumário Executivo

Este relatório documenta o fluxo de trabalho completo do **semantic-screening** (também conhecido como **DockTKinase**), uma plataforma de código aberto para predição de interações proteína-ligante usando aprendizado profundo. O sistema prediz se uma molécula (fármaco candidato) irá se ligar a uma proteína alvo (como quinases) e com qual intensidade — informação crítica para descoberta de novos medicamentos.

**Objetivo central**: Dado um par (proteína, molécula), prever:
1. **Classificação binária**: A molécula é ativa ou inativa contra a proteína?
2. **Regressão quantitativa**: Qual é a afinidade de ligação (IC50/Ki em nM)?

---

## 1. Visão Geral da Arquitetura

### 1.1 O que o Sistema Faz

O semantic-screening transforma dados textuais (sequências de aminoácidos e códigos moleculares SMILES) em predições de atividade biológica. Pense nele como um "tradutor" que:

1. **Lê** a sequência de uma proteína (ex: `MTEYKLVVVGAGGVGKSALTIQLIQ...`)
2. **Lê** o código SMILES de uma molécula (ex: `CC(=O)Nc1ccc(O)cc1`)
3. **Converte** ambos em vetores numéricos ("embeddings")
4. **Aprende** padrões de interação usando redes neurais
5. **Prediz** se há atividade e qual a intensidade

### 1.2 Por que Usar Embeddings?

**Embeddings** são representações numéricas densas que capturam informação semântica. Assim como palavras podem ser representadas como vetores (Word2Vec), proteínas e moléculas podem ser representadas como vetores:

| Entrada | Modelo de Embedding | Dimensão | Analogia |
|---------|---------------------|----------|----------|
| Proteína (sequência) | ESM-2 (Meta AI) | 320 a 5120 | "Dicionário de proteínas" |
| Molécula (SMILES) | SMI-TED (IBM) | 768 | "Dicionário de química" |

Proteínas similares terão vetores próximos; moléculas similares terão vetores próximos. O modelo aprende a "combinar" esses vetores para prever interações.

---

## 2. Pipeline Completo — Três Fases Principais

O sistema executa três fases sequenciais, orquestradas pelo arquivo `run_complete_pipeline.py`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE COMPLETO                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [FASE 1: BUILD]                                                        │
│  ├── 1.1 Carregar dados TSV (proteínas + moléculas + atividade)        │
│  ├── 1.2 Gerar embeddings de proteínas (ESM-2/ESM-C)                   │
│  ├── 1.3 Gerar embeddings de moléculas (SMI-TED)                       │
│  ├── 1.4 Concatenar embeddings [proteína | molécula]                   │
│  ├── 1.5 Estratificação inteligente (evitar data leakage)              │
│  └── 1.6 Dividir em treino/validação/teste                             │
│                           │                                             │
│                           ▼                                             │
│  [FASE 2: CLASSIFICAÇÃO]                                                │
│  ├── 2.1 Treinar 10-12 modelos de classificação                        │
│  ├── 2.2 Avaliar métricas (ROC-AUC, F1, Precisão, Recall)             │
│  └── 2.3 Selecionar melhor modelo                                      │
│                           │                                             │
│                           ▼                                             │
│  [FASE 3: REGRESSÃO]                                                    │
│  ├── 3.1 Converter valores para escala pChEMBL (log-transformação)     │
│  ├── 3.2 Treinar 10-12 modelos de regressão                            │
│  ├── 3.3 Avaliar métricas (MAE, R², RMSE)                              │
│  └── 3.4 Selecionar melhor modelo                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. FASE 1: BUILD (Construção de Embeddings)

### 3.1 Entrada de Dados

O pipeline espera um arquivo TSV (Tab-Separated Values) com colunas:

| Coluna | Descrição | Exemplo |
|--------|-----------|---------|
| `target_sequence` | Sequência de aminoácidos | MTEYKLVVVGAGGVGK... |
| `canonical_smiles` | Código SMILES da molécula | CC(=O)Nc1ccc(O)cc1 |
| `standard_value` | Atividade em nM | 150.0 |
| `standard_type` | Tipo de ensaio | IC50, Ki, Kd |

**Fonte dos dados**: ChEMBL 35 (base pública de bioatividade)

### 3.2 Geração de Embeddings de Proteínas

**Arquivo**: `src/build/embeddings/protein_embedding.py`

O sistema utiliza modelos de linguagem de proteínas pré-treinados:

| Modelo | Parâmetros | Dimensão | Uso Recomendado |
|--------|------------|----------|-----------------|
| `esm2_t6_8M_UR50D` | 8M | 320 | Testes rápidos |
| `esm2_t33_650M_UR50D` | 650M | 1280 | Balanço performance/custo |
| **`esm2_t36_3B_UR50D`** | 3B | 2560 | **Padrão (melhor qualidade)** |
| `esm2_t48_15B_UR50D` | 15B | 5120 | Máxima qualidade (requer GPU 48GB) |

**Processo**:
1. Tokenizar sequência de aminoácidos
2. Passar pelo modelo ESM-2 (transformer com bilhões de parâmetros)
3. Extrair representação da última camada
4. Aplicar mean-pooling sobre a sequência
5. Salvar vetor de dimensão `d` para cada proteína

### 3.3 Geração de Embeddings de Moléculas

**Arquivo**: `src/build/embeddings/ligand_embedding.py`

Utiliza o modelo **SMI-TED** (IBM Research):

- **Entrada**: Código SMILES (ex: `CC(=O)Nc1ccc(O)cc1`)
- **Saída**: Vetor de 768 dimensões
- **Modelo**: SMI-TED Light (~100M parâmetros)

**Características do SMI-TED**:
- Treinado em milhões de moléculas
- Captura propriedades químicas (polaridade, aromaticidade, grupos funcionais)
- Representa a "semântica" da molécula no espaço vetorial

### 3.4 Concatenação de Embeddings

**Arquivo**: `src/build/matrix/embedding_matrix.py`

Para cada par (proteína, molécula):

```
embedding_final = [embedding_proteína | embedding_molécula]
                = [2560-dim          | 768-dim          ]
                = 3328-dim vetor concatenado
```

Este vetor representa o "contexto de interação" e será usado pelos classificadores/regressores.

### 3.5 Estratificação Inteligente

**Arquivo**: `src/build/stratification/stratifier.py`

**Problema**: Se proteínas/moléculas similares aparecem em treino e teste, o modelo pode "memorizar" em vez de generalizar (data leakage).

**Solução**: Agrupamento por similaridade antes da divisão:

```
┌─────────────────────────────────────────────────────────────┐
│                   ESTRATIFICAÇÃO                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Calcular similaridade de cosseno entre embeddings       │
│ 2. Agrupar amostras similares em clusters (K-means++)      │
│ 3. Dividir CLUSTERS (não amostras) em treino/val/teste     │
│ 4. Garantir que proteínas similares não "vazem"            │
└─────────────────────────────────────────────────────────────┘
```

**Métodos de threshold disponíveis**:

| Método | Descrição |
|--------|-----------|
| `target` | Otimiza para número alvo de clusters |
| `silhouette` | Maximiza coesão intra-cluster |
| `elbow` | Ponto de inflexão na curva de distorção |
| `leakage_aware` | Minimiza vazamento entre splits |

### 3.6 Divisão dos Dados

**Proporções padrão**:
- Treino: 80%
- Validação: 10%
- Teste: 10%

**Saídas da Fase Build**:

```
build/
├── embedding_matrix.npy      # Matriz [N_amostras × 3328]
├── binary_labels.npy         # Rótulos 0/1 (ativo/inativo)
├── interaction_labels.npy    # Valores contínuos (nM ou pChEMBL)
└── splits/
    ├── train_indices.npy     # Índices do conjunto de treino
    ├── val_indices.npy       # Índices do conjunto de validação
    └── test_indices.npy      # Índices do conjunto de teste
```

---

## 4. FASE 2: CLASSIFICAÇÃO

### 4.1 Objetivo

Prever se um composto é **ativo** ou **inativo** contra a proteína alvo.

**Threshold padrão**: IC50 < 1000 nM → Ativo (label = 1)

### 4.2 Modelos Treinados

**Arquivo**: `src/classifier/multi_model_pipeline.py`

O sistema treina automaticamente 10-12 modelos diferentes:

| Modelo | Tipo | Tempo Aprox. | Características |
|--------|------|--------------|-----------------|
| NaiveBayes | Probabilístico | ~2s | Baseline rápido |
| DecisionTree | Árvore | ~5s | Interpretável |
| LogisticRegression | Linear | ~10s | Baseline linear |
| LinearSVC | SVM linear | ~15s | Escalável |
| **LightGBM** | Gradient Boosting | ~20s | **Rápido e preciso** |
| **XGBoost** | Gradient Boosting | ~25s | **Estado da arte** |
| ExtraTrees | Ensemble | ~40s | Robusto |
| RandomForest | Ensemble | ~60s | Clássico |
| AdaBoost | Boosting | ~80s | Adaptativo |
| KNN | Instância | ~120s | Não-paramétrico |
| GradientBoosting | Sklearn | ~180s | Implementação sklearn |
| MLP | Rede Neural | ~300s | Deep Learning |

### 4.3 Métricas de Avaliação

| Métrica | Descrição | Fórmula |
|---------|-----------|---------|
| **ROC-AUC** | Área sob curva ROC | Principal métrica |
| F1 | Média harmônica de precisão e recall | 2×(P×R)/(P+R) |
| Accuracy | Taxa de acertos | (TP+TN)/Total |
| Precision | Proporção de positivos corretos | TP/(TP+FP) |
| Recall | Proporção de positivos encontrados | TP/(TP+FN) |

### 4.4 Seleção do Melhor Modelo

O modelo com maior **ROC-AUC no conjunto de teste** é selecionado como vencedor.

---

## 5. FASE 3: REGRESSÃO

### 5.1 Objetivo

Prever o **valor exato de afinidade** (IC50/Ki em nM ou pChEMBL).

### 5.2 Transformação de Escala

**Problema**: Valores de IC50 variam de 0.1 nM a 100.000 nM (5 ordens de magnitude).

**Solução**: Converter para escala pChEMBL (log-transformada):

```
pChEMBL = -log₁₀(IC50 em Molar)
        = -log₁₀(IC50_nM × 10⁻⁹)
        = 9 - log₁₀(IC50_nM)
```

**Exemplo**:
- IC50 = 100 nM → pChEMBL = 9 - 2 = 7.0
- IC50 = 10 nM → pChEMBL = 9 - 1 = 8.0
- IC50 = 1 nM → pChEMBL = 9 - 0 = 9.0

Valores maiores de pChEMBL = maior afinidade = melhor fármaco.

### 5.3 Modelos Treinados

**Arquivo**: `src/regression/modular_pipeline.py`

| Modelo | Tipo | Uso |
|--------|------|-----|
| Ridge | Regularização L2 | Baseline rápido |
| Lasso | Regularização L1 | Seleção de features |
| ElasticNet | L1 + L2 | Híbrido |
| DecisionTree | Árvore | Interpretável |
| LinearSVR | SVM linear | Escalável |
| **LightGBM** | Gradient Boosting | **Recomendado** |
| **XGBoost** | Gradient Boosting | **Estado da arte** |
| ExtraTrees | Ensemble | Robusto |
| RandomForest | Ensemble | Clássico |
| KNN | Instância | Não-paramétrico |
| GradientBoosting | Sklearn | Implementação base |
| MLP | Rede Neural | Deep Learning |

### 5.4 Métricas de Avaliação

| Métrica | Descrição | Objetivo |
|---------|-----------|----------|
| **MAE** | Erro Médio Absoluto | **Minimizar** |
| **R²** | Coeficiente de Determinação | **Maximizar** (até 1.0) |
| RMSE | Raiz do Erro Quadrático Médio | Minimizar |

---

## 6. Módulo Avançado: Cross-Attention (DT-Kinase)

### 6.1 O que É

Além dos classificadores tradicionais, o sistema implementa uma arquitetura de **Deep Learning** chamada **DT-Kinase** que modela explicitamente interações proteína-ligante.

**Arquivo principal**: `src/classifier/models/cross_attention_model.py`

### 6.2 Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENTRADA: Embeddings por posição                                            │
│  ├── Proteína: [seq_len × 2560] (um vetor por aminoácido)                   │
│  └── Molécula: [mol_len × 768] (um vetor por átomo/token)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 1: CODIFICAÇÃO CNN                                                   │
│  • Convoluções 1D com kernels {3, 5, 7}                                     │
│  • Captura padrões locais (motivos, domínios)                               │
│  • Residual connections + BatchNorm + GELU                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 2: CROSS-ATTENTION (Mecanismo de Atenção Cruzada)                   │
│  • Proteína → Molécula: "Quais átomos são relevantes para cada resíduo?"   │
│  • Molécula → Proteína: "Quais resíduos são relevantes para cada átomo?"   │
│  • 8 cabeças de atenção paralelas                                           │
│  • Saída: Matriz de atenção [resíduos × átomos]                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 3: POOLING E PREDIÇÃO                                                │
│  • Adaptive Average Pooling sobre sequência                                 │
│  • Concatenação: z = [z_proteína; z_molécula]                               │
│  • Multi-task: Classificação + Regressão simultâneas                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Configuração Padrão

```python
CrossAttentionAffinityModel(
    protein_dim=2560,        # ESM-2 3B
    ligand_dim=768,          # SMI-TED
    hidden_dim=256,          # Dimensão interna
    num_cnn_layers=3,        # Camadas CNN
    kernel_sizes=(3, 5, 7),  # Tamanhos de kernel
    num_cross_attn_layers=2, # Camadas de atenção
    num_heads=8,             # Cabeças de atenção
    ff_dim=1024,             # Feed-forward
    dropout=0.1,             # Regularização
    positional_encoding='sinusoidal'  # ou 'rope'
)
```

### 6.4 RoPE (Rotary Position Embedding)

Para sequências longas, o sistema suporta **RoPE**, que codifica posição por rotação:

```
x_rotated = x · cos(mθ) + rotate_half(x) · sin(mθ)
```

**Vantagens**:
- Sem limite de comprimento de sequência
- Melhor extrapolação para proteínas longas
- Posição relativa preservada na atenção

---

## 7. Execução do Pipeline

### 7.1 Comando Completo

```bash
python run_complete_pipeline.py \
    --input data/kinase_data.tsv \
    --output results/experiment_1 \
    --protein-model esm2_t36_3B_UR50D \
    --device auto \
    --seed 42
```

### 7.2 Parâmetros Principais

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--input` | Arquivo TSV de entrada | **Obrigatório** |
| `--output` | Diretório de saída | **Obrigatório** |
| `--protein-model` | Modelo ESM-2/ESM-C | esm2_t6_8M_UR50D |
| `--ligand-model` | Modelo de molécula | SMI-TED |
| `--device` | CPU/CUDA/MPS/auto | auto |
| `--seed` | Semente aleatória | 42 |
| `--test-size` | Proporção teste | 0.1 |
| `--val-size` | Proporção validação | 0.1 |
| `--stratifier-method` | Método de estratificação | target |
| `--no-classification` | Pular classificação | False |
| `--no-regression` | Pular regressão | False |

### 7.3 Saída do Pipeline

```
results/experiment_1/
├── build/
│   ├── proteins/                # Embeddings de proteínas
│   ├── ligands/                 # Embeddings de moléculas
│   ├── embedding_matrix.npy     # Matriz concatenada
│   ├── binary_labels.npy        # Rótulos binários
│   ├── interaction_labels.npy   # Valores de afinidade
│   └── splits/                  # Índices de divisão
│
├── classifier/
│   ├── models/                  # Modelos salvos (.joblib)
│   ├── metrics/                 # Métricas JSON
│   └── predictions/             # Predições
│
├── regression/
│   ├── models/                  # Modelos salvos (.joblib)
│   ├── metrics/                 # Métricas JSON
│   └── predictions/             # Predições
│
└── checkpoints/                 # Estado intermediário
```

---

## 8. Sistema de Checkpoints

O pipeline implementa **checkpointing** para resiliência:

**Arquivo**: `src/utils/checkpoint_manager.py`

- Salva estado após cada fase
- Permite retomar execução interrompida
- Evita reprocessamento de embeddings já gerados
- Pode ser desabilitado com `--no-checkpoints`

---

## 9. Dispositivos Suportados

| Dispositivo | Flag | Detectado Automaticamente |
|-------------|------|---------------------------|
| CPU | `--device cpu` | Sim (fallback) |
| NVIDIA GPU | `--device cuda` | Sim (CUDA) |
| Apple Silicon | `--device mps` | Sim (Metal) |
| Automático | `--device auto` | Detecta o melhor |

---

## 10. Dados de Treinamento

### 10.1 Fonte

**ChEMBL 35** (Dezembro 2024) — base pública curada pelo EMBL-EBI

### 10.2 Extração SQL

```sql
SELECT 
    cs.canonical_smiles,
    ts.sequence AS target_sequence,
    act.standard_value,
    act.standard_type
FROM activities act
JOIN assays ON act.assay_id = assays.assay_id
JOIN target_dictionary td ON assays.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences ts ON tc.component_id = ts.component_id
WHERE 
    td.pref_name LIKE '%kinase%'
    AND act.standard_type IN ('IC50', 'Ki', 'Kd')
    AND act.standard_units = 'nM'
    AND ts.organism = 'Homo sapiens'
```

### 10.3 Estatísticas

| Métrica | Valor |
|---------|-------|
| Total de registros | 491.329 |
| Quinases humanas | 489/518 (94%) |
| Compostos únicos | ~150.000 |
| Intervalo de afinidade | 0.001 nM — 100.000 nM |

---

## 11. Limitações e Considerações

### 11.1 Limitações Técnicas

1. **Memória GPU**: Modelos grandes (ESM-2 15B) requerem GPUs com 48+ GB
2. **Tempo de Processamento**: Geração de embeddings pode levar horas para datasets grandes
3. **Dependência de Modelos Pré-treinados**: Qualidade depende do ESM-2/SMI-TED

### 11.2 Limitações Científicas

1. **Sem Informação 3D**: Não usa estruturas cristalográficas (proposital)
2. **Generalização Limitada**: Melhor performance em proteínas similares ao treino
3. **Interpolação vs Extrapolação**: Predições para moléculas muito diferentes são menos confiáveis

---

## 12. Próximos Passos para Adaptações

Com base neste relatório, scripts de adaptação podem:

1. **Trocar Modelo de Proteína**: Usar Boltz-2 ou ESM-C via `--protein-model`
2. **Reutilizar Embeddings**: Via `--protein-embeddings-dir` e `--ligand-embeddings-dir`
3. **Customizar Estratificação**: Via `--stratifier-method` e `--stratifier-threshold`
4. **Treinar Apenas Subconjunto de Modelos**: Via `--classification-models` e `--regression-models`
5. **Modo Cross-Attention**: Usar `attention_matrix.py` para modelo de Deep Learning

---

## Glossário

| Termo | Definição |
|-------|-----------|
| **Embedding** | Representação vetorial numérica de alta dimensão |
| **ESM-2** | Evolutionary Scale Modeling 2 — modelo de linguagem de proteínas da Meta |
| **SMI-TED** | Molecular foundation model da IBM para moléculas |
| **SMILES** | Simplified Molecular Input Line Entry System — notação textual de moléculas |
| **IC50** | Concentração que inibe 50% da atividade enzimática |
| **pChEMBL** | Escala logarítmica de atividade (-log₁₀ M) |
| **Cross-Attention** | Mecanismo que relaciona duas sequências (proteína ↔ molécula) |
| **Data Leakage** | Vazamento de informação do teste para o treino |
| **Estratificação** | Divisão de dados respeitando estrutura de clusters |
| **ROC-AUC** | Área sob curva Receiver Operating Characteristic |
| **MAE** | Mean Absolute Error — erro médio absoluto |
| **R²** | Coeficiente de determinação (0 a 1) |

---

## Referências dos Arquivos Principais

| Arquivo | Responsabilidade |
|---------|------------------|
| `run_complete_pipeline.py` | CLI principal e orquestração |
| `src/integrated_pipeline.py` | Coordenação das 3 fases |
| `src/build/pipeline/build_pipeline.py` | Pipeline de embeddings |
| `src/build/embeddings/protein_embedding.py` | Wrapper ESM-2 |
| `src/build/embeddings/ligand_embedding.py` | Wrapper SMI-TED |
| `src/build/stratification/stratifier.py` | Estratificação por clusters |
| `src/classifier/multi_model_pipeline.py` | Pipeline de classificação |
| `src/regression/modular_pipeline.py` | Pipeline de regressão |
| `src/classifier/models/cross_attention_model.py` | Arquitetura DT-Kinase |
| `attention_matrix.py` | CLI para Cross-Attention |

---

*Este relatório foi gerado com base na análise completa do código-fonte do repositório semantic-screening/docktkinase em Janeiro de 2025.*
