# Análise de Data Leakage em Predição de Interação Proteína-Ligante

**Data**: 2026-02-17
**Datasets**: Non-Human (kinase_non_human_compounds.tsv) e Human (kinase_human_compounds.tsv)
**Protocolo**: kfold_cv_v5, 10-fold CV, seed=42
**Threshold de Afinidade**: pChEMBL >= 6.0 (IC50 <= 1000 nM)

---

## Índice de Figuras

| # | Nome do Arquivo | Non-Human | Human |
|---|-----------------|-----------|-------|
| 01 | `01_leakage_analysis.png` | [NH-01](#análise-2-diagnóstico-de-data-leakage) | [H-01](#análise-2-diagnóstico-de-data-leakage) |
| 02 | `02_baseline_comparison.png` | [NH-02](#análise-3-baselines-de-lookup) | [H-02](#análise-3-baselines-de-lookup) |
| 03 | `03_kinase_imbalance.png` | [NH-03](#análise-4-desbalanceamento-de-kinases) | [H-03](#análise-4-desbalanceamento-de-kinases) |
| 04 | `04_compound_consistency.png` | [NH-04](#análise-5-consistência-de-compostos) | [H-04](#análise-5-consistência-de-compostos) |
| 05 | `05_similarity_analysis.png` | [NH-05](#análise-6-similaridade-química) | [H-05](#análise-6-similaridade-química) |
| 06 | `06_split_comparison.png` | [NH-06](#análise-7-comparação-de-cenários-de-split) | [H-06](#análise-7-comparação-de-cenários-de-split) |
| 07 | `07_inflated_vs_real_performance.png` | [NH-07](#análise-8-performance-inflada-vs-real) | [H-07](#análise-8-performance-inflada-vs-real) |

**Paths dos resultados**:
- Non-Human: `results/non_human_13_02_2026/non_human/`
- Human: `results/human_13_02_2026/human/`

---

## ANÁLISE 1: Características dos Datasets

**Fonte**: `analysis_report.txt` (Non-Human e Human)

| Métrica | Non-Human | Human | Ratio |
|---------|-----------|-------|-------|
| **Total de amostras** | 14,080 | 473,760 | 1:34 |
| **Compostos únicos** | 7,428 | 136,003 | 1:18 |
| **Kinases únicas** | 114 | 517 | 1:4.5 |
| **Amostras/composto** | 1.9 | 3.5 | - |
| **Amostras/kinase** | 123 | 917 | - |

### Interpretação

O dataset **Human** é substancialmente maior e mais denso:
- **34x mais amostras** permite estimativas mais estáveis
- **4.5x mais kinases** representa maior diversidade de alvos
- **Maior densidade** (amostras/kinase) indica mais medições por alvo

**Implicação**: Resultados do dataset Human são mais confiáveis estatisticamente, mas ambos mostram padrões consistentes.

---

## ANÁLISE 2: Diagnóstico de Data Leakage

**Figuras**:
- `NH-01`: `results/non_human_13_02_2026/non_human/01_leakage_analysis.png`
- `H-01`: `results/human_13_02_2026/human/01_leakage_analysis.png`

**Fonte adicional**: `analysis_report.txt`

| Métrica | Non-Human (NH-01) | Human (H-01) |
|---------|-------------------|--------------|
| **Linhas de teste com composto vazado** | 65.1% | 82.0% |
| **Duplicatas exatas (composto+kinase)** | 39.6% | 28.3% |

### Interpretação

**Figura NH-01** e **Figura H-01** mostram a sobreposição de compostos entre conjuntos de treino e teste em random split:

- **Human tem MAIS leakage de compostos** (82% vs 65%) — a maioria dos compostos no teste já foi vista no treino
- **Non-Human tem mais duplicatas exatas** (39.6% vs 28.3%) — pares idênticos (composto, kinase) repetidos
- Em ambos os casos, o modelo pode "trapacear" memorizando associações já vistas

**Implicação**: Random split permite que o modelo memorize em vez de generalizar. Performance reportada é artificialmente inflada.

---

## ANÁLISE 3: Baselines de Lookup

**Figuras**:
- `NH-02`: `results/non_human_13_02_2026/non_human/02_baseline_comparison.png`
- `H-02`: `results/human_13_02_2026/human/02_baseline_comparison.png`

**Fonte adicional**: `analysis_report.txt`

| Baseline | Non-Human (NH-02) | Human (H-02) |
|----------|-------------------|--------------|
| **Lookup Composto** | Acc=0.788, MCC=0.593 | Acc=0.729, MCC=0.479 |
| **Lookup Kinase** | Acc=0.749, MCC=0.521 | Acc=0.682, MCC=0.365 |
| **Lookup Comp+Kin** | Acc=0.867, MCC=0.732 | Acc=0.828, MCC=0.656 |
| **KNN (random split)** | Acc=0.898, MCC=0.793 | Acc=0.845, MCC=0.690 |

### Interpretação

**Figura NH-02** e **Figura H-02** comparam baselines triviais com o modelo KNN:

- **Lookup simples já atinge MCC > 0.5** apenas "lembrando" o label mais frequente para cada composto/kinase
- **KNN supera lookup por margem pequena** (~0.06 MCC em ambos datasets)
- O baseline Comp+Kin (MCC 0.73/0.66) está **próximo do KNN** (MCC 0.79/0.69)

**Implicação**: Grande parte da performance do modelo vem de memorização, não de aprendizado de padrões químicos generalizáveis. Um dicionário de lookup quase iguala um modelo de ML.

---

## ANÁLISE 4: Desbalanceamento de Kinases

**Figuras**:
- `NH-03`: `results/non_human_13_02_2026/non_human/03_kinase_imbalance.png`
- `H-03`: `results/human_13_02_2026/human/03_kinase_imbalance.png`

**Fonte adicional**: `analysis_report.txt`

| Métrica | Non-Human (NH-03) | Human (H-03) |
|---------|-------------------|--------------|
| **Kinases desbalanceadas (>80% ou <20% ativos)** | 49 (43.0%) | 192 (37.2%) |

### Interpretação

**Figura NH-03** e **Figura H-03** mostram a distribuição das taxas de atividade por kinase:

- **43% das kinases em Non-Human** são fortemente desbalanceadas
- **37% das kinases em Human** são fortemente desbalanceadas
- Kinases com 100% ativos ou 0% ativos foram filtradas (monotonic filter)

**Implicação**: O desbalanceamento varia entre kinases. Modelos podem ter performance alta em kinases fáceis (extremas) e baixa em kinases difíceis (balanceadas). MCC é métrica apropriada por ser robusta a desbalanceamento.

---

## ANÁLISE 5: Consistência de Compostos

**Figuras**:
- `NH-04`: `results/non_human_13_02_2026/non_human/04_compound_consistency.png`
- `H-04`: `results/human_13_02_2026/human/04_compound_consistency.png`

**Fonte adicional**: `analysis_report.txt`

| Categoria | Non-Human (NH-04) | Human (H-04) |
|-----------|-------------------|--------------|
| **Uma kinase apenas** | 5,229 (80.9%) | 83,420 (69.2%) |
| **Perfeitamente consistente** | 905 (14.0%) | 23,570 (19.6%) |
| **Inconsistente** | 331 (5.1%) | 13,475 (11.2%) |

### Interpretação

**Figura NH-04** e **Figura H-04** analisam se o mesmo composto tem labels consistentes entre kinases:

- **Maioria dos compostos** foi testada contra apenas uma kinase (69-81%)
- **Compostos consistentes**: mesmo label (ativo/inativo) para todas as kinases testadas
- **Compostos inconsistentes**: ativo para algumas kinases, inativo para outras (comportamento esperado biologicamente)

**Implicação**: Poucos compostos são "pan-ativos" ou "pan-inativos". A seletividade química é real, e modelos precisam capturar interações específicas composto-kinase.

---

## ANÁLISE 6: Similaridade Química

**Figuras**:
- `NH-05`: `results/non_human_13_02_2026/non_human/05_similarity_analysis.png`
- `H-05`: `results/human_13_02_2026/human/05_similarity_analysis.png`

**Fonte adicional**: `analysis_report.txt`

| Faixa de Similaridade | Non-Human (NH-05) | Human (H-05) |
|-----------------------|-------------------|--------------|
| **Muito similar (Tanimoto > 0.8)** | 46.6% | 56.6% |
| **Similar (0.6-0.8)** | 43.7% | 37.2% |
| **Dissimilar (< 0.6)** | ~10% | ~6% |

### Interpretação

**Figura NH-05** e **Figura H-05** mostram a distribuição de similaridade de Tanimoto entre compostos de teste e seus vizinhos mais próximos no treino:

- **~90% dos compostos de teste** têm vizinho no treino com similaridade > 0.6
- **Human tem mais compostos muito similares** (56.6% vs 46.6% com Tanimoto > 0.8)
- Poucos compostos são verdadeiramente novos (dissimilares)

**Implicação**: Em random split, o modelo pode prever baseado em vizinhos químicos quase idênticos. Isso não testa capacidade de generalização para scaffolds novos.

---

## ANÁLISE 7: Comparação de Cenários de Split

**Figuras**:
- `NH-06`: `results/non_human_13_02_2026/non_human/06_split_comparison.png`
- `H-06`: `results/human_13_02_2026/human/06_split_comparison.png`

**Fonte adicional**: `split_comparison_results.json`

### Resultados por Cenário (MLP, média ± std)

| Cenário | Non-Human MCC (NH-06) | Human MCC (H-06) | Leakage % |
|---------|----------------------|------------------|-----------|
| **S1: Random Split** | 0.792 ± 0.017 | 0.781 ± 0.002 | 59-69% |
| **Scaffold Split** | 0.568 ± 0.060 | 0.579 ± 0.023 | 0% |
| **S2: Cold-Drug** | 0.674 ± 0.051 | 0.635 ± 0.014 | 0% |
| **S3: Cold-Target** | 0.303 ± 0.161 | 0.432 ± 0.033 | 39-55%* |
| **S4: True Generalization** | 0.233 ± 0.252 | 0.241 ± 0.066 | 0% |

*S3 tem leakage de compostos mas não de kinases

### Interpretação

**Figura NH-06** e **Figura H-06** mostram barras de Accuracy e MCC para KNN e MLP em cada cenário:

```
Hierarquia de Dificuldade (MCC decrescente):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

S1 Random    S2 Cold-Drug   Scaffold    S3 Cold-Target   S4 True Gen.
   0.79          0.67          0.57          0.37            0.24
    │             │              │             │               │
    └─────────────┴──────────────┴─────────────┴───────────────┘
         -15%          -28%          -53%           -70%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Observações-chave**:
1. **S1 → S2 (-15%)**: Remover leakage de compostos causa queda moderada
2. **S2 → S3 (-40%)**: Kinases novas são MUITO mais difíceis que compostos novos
3. **S4 (~0.24)**: Performance próxima ao acaso (MCC=0 seria aleatório)

**Implicação**: A tarefa de generalização verdadeira (S4) permanece essencialmente não resolvida com features simples (Morgan FP + one-hot kinase).

---

## ANÁLISE 8: Performance Inflada vs Real

**Figuras**:
- `NH-07`: `results/non_human_13_02_2026/non_human/07_inflated_vs_real_performance.png`
- `H-07`: `results/human_13_02_2026/human/07_inflated_vs_real_performance.png`

**Fonte adicional**: `split_comparison_results.json`

| Métrica | Non-Human (NH-07) | Human (H-07) |
|---------|-------------------|--------------|
| **S1 Random MCC** | 0.792 | 0.781 |
| **S4 True Gen. MCC** | 0.233 | 0.241 |
| **Fator de Inflação** | **3.40x** | **3.24x** |
| **Queda Percentual** | -70.6% | -69.1% |

### Interpretação

**Figura NH-07** e **Figura H-07** mostram comparação lado-a-lado de performance "Reportada" (S1) vs "Real" (S4):

- **Setas indicam queda de ~70%** entre performance inflada e real
- O fator de inflação é **consistente entre datasets** (~3.3x)
- Isso significa que MCC=0.79 em random split corresponde a MCC=0.24 em condições reais

**Implicação**: Resultados publicados usando apenas random split superestimam a capacidade preditiva por um fator de 3x. Comparações entre modelos usando S1 são potencialmente enganosas.

---

## ANÁLISE 9: Testes Estatísticos

**Fonte**: Calculado a partir de `split_comparison_results.json` (fold_results)

### Teste de Wilcoxon (S1 vs S4, MLP)

| Dataset | Estatística W | p-valor | Significância |
|---------|---------------|---------|---------------|
| Non-Human | 55.0 | 0.00098 | *** (p < 0.001) |
| Human | 55.0 | 0.00098 | *** (p < 0.001) |

### Tamanho do Efeito (Cohen's d)

| Dataset | Cohen's d | Interpretação |
|---------|-----------|---------------|
| Non-Human | 3.13 | Efeito GRANDE |
| Human | 11.53 | Efeito MUITO GRANDE |

### Interpretação

- A diferença S1 vs S4 é **estatisticamente significativa** (p < 0.001)
- Cohen's d > 0.8 indica efeito grande; valores de 3-11 são **extraordinariamente grandes**
- A queda de performance não é artefato de variância — é um efeito **robusto e reprodutível**

---

## ANÁLISE 10: Efeito Blind-Target (S2 vs S3)

**Fonte**: `split_comparison_results.json`

| Dataset | S2 Cold-Drug | S3 Cold-Target | Gap | % Perda |
|---------|--------------|----------------|-----|---------|
| Non-Human | 0.674 | 0.303 | 0.371 | **55.1%** |
| Human | 0.635 | 0.432 | 0.203 | **31.9%** |

### Interpretação

Comparando cenários onde apenas composto (S2) ou apenas kinase (S3) é nova:

- **S2 (cold-drug)**: Composto novo, kinase conhecida → modelo tem representação one-hot do alvo
- **S3 (cold-target)**: Kinase nova → one-hot é vetor de zeros (blind-target condition)

A perda de 32-55% demonstra que:
1. O modelo **depende fortemente da identidade do alvo**
2. One-hot encoding **não generaliza** para kinases novas
3. Human sofre menos (32% vs 55%) possivelmente por ter mais kinases similares

**Implicação**: Para predição em kinases novas, é necessário usar **embeddings de sequência** (ESM-2, ProtTrans) em vez de one-hot encoding.

---

## ANÁLISE 11: Estabilidade Inter-Fold

**Fonte**: `split_comparison_results.json` (campo `mcc_std`)

| Cenário | Non-Human std | Human std | Ratio NH/H |
|---------|---------------|-----------|------------|
| S1: Random | 0.017 | 0.002 | 8.5x |
| S2: Cold-Drug | 0.051 | 0.015 | 3.4x |
| S3: Cold-Target | 0.161 | 0.033 | 4.9x |
| S4: True Generalization | 0.252 | 0.066 | 3.8x |

### Interpretação

- **Human é 3-8x mais estável** que Non-Human em todos os cenários
- **Cenários mais difíceis** (S3, S4) têm **maior variância**
- Non-Human S4: std = 0.252 indica que MCC varia de -0.27 a +0.59 entre folds

**Implicação**:
- Datasets pequenos produzem estimativas instáveis e não confiáveis
- Para publicação científica, preferir dataset Human ou usar mais folds

---

## ANÁLISE 12: KNN vs MLP

**Fonte**: `split_comparison_results.json`

### Diferença MCC (MLP - KNN)

| Cenário | Non-Human | Human |
|---------|-----------|-------|
| S1: Random | +0.049 | +0.113 |
| Scaffold | +0.009 | +0.103 |
| S2: Cold-Drug | +0.025 | +0.101 |
| S3: Cold-Target | +0.061 | +0.034 |
| S4: True Generalization | +0.017 | +0.023 |

### Interpretação

- **MLP supera KNN em 100% dos cenários** (10/10)
- Vantagem é **maior em Human** (~0.10) que em Non-Human (~0.03)
- Vantagem **diminui em cenários difíceis** (S4: apenas +0.02)

**Implicação**: MLP aprende representações ligeiramente melhores, mas a vantagem é marginal quando a tarefa é genuinamente difícil. Ambos modelos falham similarmente em S4.

---

## CONCLUSÃO FINAL

### Síntese Visual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IMPACTO DO DATA LEAKAGE EM DTI PREDICTION                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Random Split (S1)           True Generalization (S4)                      │
│   [NH-06, H-06]               [NH-06, H-06]                                 │
│   ═══════════════════         ═══════════════════════                       │
│                                                                             │
│        MCC ≈ 0.79       ────── -70% ──────►    MCC ≈ 0.24                   │
│                                                                             │
│        "Excelente"                             "Marginal"                   │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  EVIDÊNCIAS [Figuras NH-01 a NH-07, H-01 a H-07]:                │      │
│   │                                                                  │      │
│   │  • 65-82% dos compostos de teste vazam do treino [NH-01, H-01]   │      │
│   │  • Lookup table simples atinge MCC > 0.5 [NH-02, H-02]           │      │
│   │  • ~90% dos compostos têm vizinho similar no treino [NH-05, H-05]│      │
│   │  • Queda de 70% é estatisticamente significativa (p < 0.001)     │      │
│   │  • Tamanho de efeito é GRANDE (Cohen's d > 3)                    │      │
│   │  • Fator de inflação: 3.3x [NH-07, H-07]                         │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Principais Conclusões

#### 1. Data Leakage é Prevalente e Grave
**Evidência**: Figuras NH-01, H-01 (leakage analysis) e NH-05, H-05 (similarity)
- 65-82% das amostras de teste contêm compostos já vistos no treino
- Random split infla métricas por **3.3x** em média
- Resultados publicados com random split são **sistematicamente otimistas**

#### 2. Hierarquia de Dificuldade é Consistente
**Evidência**: Figuras NH-06, H-06 (split comparison)
```
S1 Random > S2 Cold-Drug > Scaffold > S3 Cold-Target > S4 True Gen.
   0.79        0.67          0.57        0.37            0.24
```
Padrão idêntico em ambos datasets, validando a robustez dos achados.

#### 3. Cold-Target é Mais Difícil que Cold-Drug
**Evidência**: Comparação S2 vs S3 nas Figuras NH-06, H-06
- Perda de 32-55% ao prever para kinases novas
- One-hot encoding de kinases **não generaliza**
- Necessário usar **embeddings de sequência** (ESM-2, ProtTrans)

#### 4. Generalização Verdadeira Permanece Um Desafio
**Evidência**: Figuras NH-07, H-07 (inflated vs real)
- MCC ≈ 0.24 em S4 (ambos datasets)
- Apenas ligeiramente acima do acaso
- Morgan FP + One-hot são **insuficientes** para a tarefa

#### 5. Datasets Maiores São Mais Confiáveis
**Evidência**: Comparação de std entre datasets
- Human (34x maior) tem 4-5x menos variância inter-fold
- Resultados de Non-Human S4 (std=0.25) são instáveis

### Tabela-Resumo de Referência

| Achado | Figuras | Datasets |
|--------|---------|----------|
| Leakage de compostos | NH-01, H-01 | Ambos |
| Baselines de lookup | NH-02, H-02 | Ambos |
| Desbalanceamento | NH-03, H-03 | Ambos |
| Consistência | NH-04, H-04 | Ambos |
| Similaridade química | NH-05, H-05 | Ambos |
| Comparação de cenários | NH-06, H-06 | Ambos |
| Inflação de performance | NH-07, H-07 | Ambos |

### Implicações Práticas

| Para Pesquisadores | Para Practitioners |
|-------------------|-------------------|
| Sempre reportar S2/S3/S4, não apenas S1 | Não confiar em benchmarks com random split |
| Usar k-fold CV com splits corretos | Esperar performance real ~3x menor que publicada |
| Testar em kinases/compostos novos | Priorizar modelos avaliados em cold-start |
| Considerar embeddings de sequência | Combinar múltiplas fontes de evidência |

### Recomendação Final

> **Para avaliação científica rigorosa de modelos DTI, é MANDATÓRIO usar o cenário S4 (new_compound_new_kinase) como métrica principal de generalização. Resultados reportados apenas com random split (S1) superestimam a capacidade preditiva real em ~3.3x e não devem ser usados para comparar modelos ou tomar decisões.**

---

## Apêndice: Localização dos Arquivos

### Non-Human
```
results/non_human_13_02_2026/non_human/
├── 01_leakage_analysis.png          # NH-01
├── 02_baseline_comparison.png       # NH-02
├── 03_kinase_imbalance.png          # NH-03
├── 04_compound_consistency.png      # NH-04
├── 05_similarity_analysis.png       # NH-05
├── 06_split_comparison.png          # NH-06
├── 07_inflated_vs_real_performance.png  # NH-07
├── analysis_report.txt
├── split_comparison_results.json
└── README.md
```

### Human
```
results/human_13_02_2026/human/
├── 01_leakage_analysis.png          # H-01
├── 02_baseline_comparison.png       # H-02
├── 03_kinase_imbalance.png          # H-03
├── 04_compound_consistency.png      # H-04
├── 05_similarity_analysis.png       # H-05
├── 06_split_comparison.png          # H-06
├── 07_inflated_vs_real_performance.png  # H-07
├── analysis_report.txt
├── split_comparison_results.json
└── README.md
```
