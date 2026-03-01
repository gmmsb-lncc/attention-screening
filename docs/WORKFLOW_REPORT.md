# DockTKinase (semantic-screening) — Relatório Completo de Fluxo de Trabalho

**Autor**: GMMSB-LNCC
**Data**: Fevereiro 2026
**Versão**: 3.0 (Scaffold splits + Benchmark unificado)
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

## 2. Pipeline Completo — Benchmark Unificado

O sistema executa um pipeline de benchmark em 5 passos, orquestrado pelo script `semantic_screening_models_beta.py`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BENCHMARK UNIFICADO (3 NÍVEIS)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [STEP 0: SCAFFOLD SPLIT]                                               │
│  ├── Decompor moléculas em scaffolds Murcko                            │
│  ├── Selecionar scaffolds de teste (fixo, compartilhado)               │
│  └── Gerar train/val/test sem sobreposição de scaffolds                │
│                           │                                             │
│                           ▼                                             │
│  [STEP 1: LEVEL 1 — Fingerprint Baseline]                               │
│  ├── Extrair fingerprints moleculares (ECFP)                           │
│  ├── Treinar KNN + MLP no scaffold split                               │
│  └── Avaliar métricas (Accuracy, MCC, F1, AUC, ...)                   │
│                           │                                             │
│                           ▼                                             │
│  [STEP 2: LEVEL 2 — Embedding Vectors]                                  │
│  ├── Usar vetores ESM-2 (proteína) + MoLFormer (ligante) mean-pooled  │
│  ├── Treinar KNN + MLP no mesmo scaffold split                         │
│  └── Avaliar métricas (mesmo protocolo)                                │
│                           │                                             │
│                           ▼                                             │
│  [STEP 3: LEVEL 3 — DT-Kinase Deep Learning]                           │
│  ├── Usar matrizes per-token (ESM-2 + MoLFormer)                      │
│  ├── Treinar CNN + CrossAttention (multi-seed: 5 seeds)                │
│  └── Avaliar métricas com threshold otimizado no val                   │
│                           │                                             │
│                           ▼                                             │
│  [STEP 4: RELATÓRIO COMPARATIVO]                                        │
│  ├── Agregar métricas dos 3 níveis                                     │
│  ├── Gerar visualizações comparativas (5 gráficos)                     │
│  └── Salvar benchmark_comparison.json                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Objetivo do Benchmark

O benchmark responde duas perguntas científicas fundamentais:

1. **Level 1 vs Level 2**: Qual o ganho de usar embeddings de PLMs no lugar de fingerprints clássicos?
2. **Level 2 vs Level 3**: Qual o ganho de preservar contexto por-resíduo/por-átomo (matrizes) ao invés de vetores agregados?

Todos os 3 níveis usam **exatamente o mesmo scaffold split**, garantindo comparação justa.

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

### 3.5 Scaffold Split (Divisão por Scaffolds Murcko)

**Arquivo**: `scaffold_split.py` + `scaffolds_splits/scenario_splitter.py`

**Problema**: Se compostos da mesma série química aparecem em treino e teste, o modelo pode "memorizar" padrões de scaffolds em vez de generalizar (data leakage).

**Solução**: Divisão baseada em **scaffolds Murcko** — o esqueleto central (sistema de anéis) de cada molécula:

```
┌─────────────────────────────────────────────────────────────┐
│                  SCAFFOLD SPLIT                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Extrair scaffold Murcko de cada composto                 │
│ 2. Selecionar scaffolds de teste via otimização             │
│    (balanceando fração e distribuição de classes)            │
│ 3. Dividir scaffolds restantes em treino/validação          │
│ 4. Garantir: NENHUM scaffold aparece em >1 split            │
└─────────────────────────────────────────────────────────────┘
```

**Garantias**:
- Compostos da mesma série química NUNCA divididos entre splits
- Conjunto de teste fixo e compartilhado entre datasets (human/non_human)
- Distribuição de classes monitorada e otimizada

**Proporções padrão**:
- Treino: ~80%
- Validação: ~10%
- Teste: ~10%

**Saídas do Scaffold Split**:

```
scaffolds_splits/output/
├── manifest.json                    # Metadados do split
├── {dataset}_test.tsv               # Conjunto de teste fixo
├── {dataset}_train.tsv              # Treino (cenário padrão Sc)
├── {dataset}_val.tsv                # Validação (cenário padrão Sc)
├── scenarios/Sc/                    # Scaffold-disjoint train/val
│   ├── {dataset}_train.tsv
│   └── {dataset}_val.tsv
└── split_class_distribution_summary.csv
```

---

## 4. Os Três Níveis de Modelos

O benchmark unificado avalia modelos em três níveis de complexidade crescente:

### 4.1 Level 1 — Fingerprint Baseline

**Entrada**: Fingerprints moleculares (ECFP — Extended-Connectivity Fingerprints)
**Modelos**: KNN + MLP
**Arquivo**: `split_comparison_analysis.py` com `feature_type="fingerprint"`

Fingerprints são representações binárias clássicas que codificam subestruturas presentes na molécula. Servem como **baseline** para avaliar o ganho de usar embeddings de PLMs.

### 4.2 Level 2 — Embedding Vectors

**Entrada**: Vetores mean-pooled de ESM-2 (proteína) + MoLFormer (ligante)
**Modelos**: KNN + MLP
**Arquivo**: `split_comparison_analysis.py` com `feature_type="embedding"`

Usa vetores fixos gerados por modelos de linguagem pré-treinados. A comparação Level 1 vs Level 2 mede o **valor dos embeddings de PLMs** sobre fingerprints tradicionais.

### 4.3 Level 3 — DT-Kinase (CNN + CrossAttention)

**Entrada**: Matrizes per-token de ESM-2 (por resíduo) + MoLFormer (por átomo)
**Modelo**: CNN multi-escala + Cross-Attention bidirecional
**Arquivo**: `crossattention_split_analysis/experiment.py`

Preserva o contexto de cada resíduo e cada átomo, permitindo ao modelo aprender quais regiões da proteína interagem com quais partes do ligante. A comparação Level 2 vs Level 3 mede o **valor do contexto posicional**.

### 4.4 Métricas de Avaliação (todos os níveis)

| Métrica | Descrição | Uso |
|---------|-----------|-----|
| **MCC** | Matthews Correlation Coefficient | **Métrica primária de seleção** |
| AUC | Área sob curva ROC | Qualidade de ranking |
| F1 | Média harmônica de precisão e recall | Balanço P/R |
| Accuracy | Taxa de acertos | Visão geral |
| Precision | Proporção de positivos corretos | Controle de falsos positivos |
| Recall | Proporção de positivos encontrados | Controle de falsos negativos |

**Threshold de atividade**: pChEMBL >= 6.0 (IC50 <= 1000 nM) → Ativo

## 5. Escala pChEMBL

**Problema**: Valores de IC50 variam de 0.1 nM a 100.000 nM (5 ordens de magnitude).

**Solução**: Converter para escala pChEMBL (log-transformada):

```
pChEMBL = -log₁₀(IC50 em Molar)
        = -log₁₀(IC50_nM × 10⁻⁹)
        = 9 - log₁₀(IC50_nM)
```

**Exemplo**:
- IC50 = 1000 nM → pChEMBL = 6.0 (threshold ativo/inativo)
- IC50 = 100 nM → pChEMBL = 7.0
- IC50 = 10 nM → pChEMBL = 8.0
- IC50 = 1 nM → pChEMBL = 9.0

Valores maiores de pChEMBL = maior afinidade = melhor fármaco.

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

## 7. Execução do Benchmark

### 7.1 Comando Principal (Benchmark Unificado)

```bash
# Benchmark completo (3 níveis)
python semantic_screening_models_beta.py \
    --dataset non_human \
    --embedding 8M

# Apenas Level 1 e 2 (baseline rápido)
python semantic_screening_models_beta.py \
    --dataset non_human \
    --embedding 8M \
    --levels 1,2

# Apenas Level 3 com hiperparâmetros customizados
python semantic_screening_models_beta.py \
    --dataset non_human \
    --embedding 8M \
    --levels 3 \
    --epochs 100 \
    --batch_size 32
```

### 7.2 Parâmetros Principais

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--dataset` | Dataset (human, non_human) | **Obrigatório** |
| `--embedding` | Modelo ESM-2 (8M, 150M, 650M) | 8M |
| `--levels` | Níveis a executar (1,2,3) | 1,2,3 |
| `--output_dir` | Diretório de saída | auto |
| `--seeds` | Seeds para reprodutibilidade | [42,123,456,789,1024] |
| `--force` | Forçar recálculo | False |
| `--force_split` | Regenerar scaffold splits | False |
| `--epochs` | Épocas Level 3 | 500 |
| `--batch_size` | Batch size Level 3 | 32 |
| `--patience` | Early stopping (0=desabilitado) | 30 |

### 7.3 Saída do Benchmark

```
results/benchmark_non_human_8M/
├── level1_fingerprint/non_human/      # Resultados Level 1
│   └── split_comparison_results.json
├── level2_embedding_8M/non_human/     # Resultados Level 2
│   └── split_comparison_results.json
├── level3_cnn_crossattn_8M/           # Resultados Level 3
│   └── *_crossattention_analysis_results.json
├── benchmark_comparison.json          # Tabela unificada
├── benchmark_grouped_bar.png          # Comparativo barras
├── benchmark_radar.png                # Gráfico radar
├── benchmark_heatmap.png              # Mapa de calor
├── benchmark_mcc_ranking.png          # Ranking por MCC
└── benchmark_per_metric.png           # Comparativo por métrica
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

## 12. Próximos Passos

Com base neste pipeline, os próximos passos incluem:

1. **Expandir embeddings**: Testar novos modelos de proteína (ESM-C 600M, ESM-2 3B)
2. **Benchmark multi-embedding**: Rodar benchmark unificado com diferentes embeddings para comparar
3. **Análise de leakage**: Usar `scripts/data_leakage_analysis.py` para validar splits
4. **Multi-seed statistics**: Garantir significância estatística com 5+ seeds
5. **Cross-dataset**: Comparar performance human vs non_human no mesmo benchmark

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
| `semantic_screening_models_beta.py` | **Benchmark unificado (3 níveis)** |
| `scaffold_split.py` | Geração de scaffold splits |
| `scaffolds_splits/scenario_splitter.py` | Splitting por cenário (Sc, S1-S4) |
| `split_comparison_analysis.py` | Level 1 e 2 (KNN/MLP) |
| `crossattention_split_analysis/experiment.py` | Level 3 (CNN+CrossAttention) |
| `crossattention_split_analysis/config.py` | Configuração e constantes |
| `scripts/extract_ligand_vectors.py` | Mean-pool MoLFormer → vetores |
| `src/classifier/models/cross_attention_model.py` | Arquitetura DT-Kinase |
| `src/build/embeddings/strategies/` | Estratégias de embedding (ESM-2, ESM-C, MoLFormer) |

---

*Atualizado em Fevereiro de 2026. Reflete a metodologia de scaffold splits e benchmark unificado de 3 níveis.*
