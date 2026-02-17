# Análise: Impacto do Filtro de Compostos Monotônicos no Dataset Non-Human

**Data**: 2026-02-17
**Dataset**: Non-Human Kinases
**Comparação**: Antes vs Depois do filtro `--filter_monotonic_compounds`

---

## Índice de Figuras

As figuras estão organizadas por experimento. Prefixo **A-** = ANTES, **D-** = DEPOIS.

| ID | Figura | Descrição | Seção |
|----|--------|-----------|-------|
| **A-01** | `non_human_13_02_2026/.../01_leakage_analysis.png` | Análise de vazamento (ANTES) | §4 |
| **A-02** | `non_human_13_02_2026/.../02_baseline_comparison.png` | Comparação de baselines (ANTES) | §4.1 |
| **A-03** | `non_human_13_02_2026/.../03_kinase_imbalance.png` | Desbalanceamento de kinases (ANTES) | §6.3 |
| **A-04** | `non_human_13_02_2026/.../04_compound_consistency.png` | Consistência de compostos (ANTES) | §6.1, §6.2 |
| **A-05** | `non_human_13_02_2026/.../05_similarity_analysis.png` | Análise de similaridade (ANTES) | §6.3 |
| **A-06** | `non_human_13_02_2026/.../06_split_comparison.png` | Comparação de splits (ANTES) | §2 |
| **A-07** | `non_human_13_02_2026/.../07_inflated_vs_real_performance.png` | Inflação de métricas (ANTES) | §3 |
| **D-01** | `non_human_monotonic_17_02_2026/.../01_leakage_analysis.png` | Análise de vazamento (DEPOIS) | §4 |
| **D-02** | `non_human_monotonic_17_02_2026/.../02_baseline_comparison.png` | Comparação de baselines (DEPOIS) | §4.1 |
| **D-03** | `non_human_monotonic_17_02_2026/.../03_kinase_imbalance.png` | Desbalanceamento de kinases (DEPOIS) | §6.3 |
| **D-04** | `non_human_monotonic_17_02_2026/.../04_compound_consistency.png` | Consistência de compostos (DEPOIS) | §6.1, §6.2 |
| **D-05** | `non_human_monotonic_17_02_2026/.../05_similarity_analysis.png` | Análise de similaridade (DEPOIS) | §6.3 |
| **D-06** | `non_human_monotonic_17_02_2026/.../06_split_comparison.png` | Comparação de splits (DEPOIS) | §2 |
| **D-07** | `non_human_monotonic_17_02_2026/.../07_inflated_vs_real_performance.png` | Inflação de métricas (DEPOIS) | §3 |

---

## Resumo Executivo

Este documento compara os resultados do dataset Non-Human **antes** (apenas filtro de kinases monotônicas) e **depois** (filtro de kinases + compostos monotônicos) para avaliar o impacto na avaliação de modelos.

### Conclusão Principal

| Aspecto | Resultado |
|---------|-----------|
| **Amostras removidas** | 3,334 (23.7%) |
| **Impacto no Random Split** | MCC reduziu de 0.792 → 0.762 (-3.8%) |
| **Impacto no S4 (mais rigoroso)** | MCC reduziu de 0.233 → 0.164 (-29.6%) |
| **Fator de inflação** | Aumentou de 3.4x → 4.6x |

**Interpretação**: O filtro de compostos monotônicos remove casos "fáceis", tornando a avaliação mais rigorosa. A queda maior no S4 indica que esses compostos inflacionavam artificialmente as métricas de generalização.

---

## 1. Configuração dos Experimentos

### 1.1 Parâmetros Comuns

| Parâmetro | Valor |
|-----------|-------|
| Dataset | non_human |
| Protocolo | 10-fold CV (80/10/10) |
| Seed | 42 |
| Threshold pChEMBL | ≥ 6.0 |
| Filtro kinases monotônicas | ✅ Ativado |

### 1.2 Diferença entre Experimentos

| Experimento | Path | Filtro Compostos | Amostras | Compostos | Kinases |
|-------------|------|------------------|----------|-----------|---------|
| **ANTES** | `results/non_human_13_02_2026/` | ❌ Desativado | 14,080 | 7,436 | 114 |
| **DEPOIS** | `results/non_human_monotonic_17_02_2026/` | ✅ Ativado | 10,746 | 6,348 | 112 |
| **Diferença** | - | - | **-3,334 (-23.7%)** | **-1,088 (-14.6%)** | **-2 (-1.8%)** |

---

## 2. Comparação de Métricas por Cenário

> **Figuras de referência**: A-06, D-06 (`06_split_comparison.png`)

### 2.1 MCC (Matthews Correlation Coefficient) - Métrica Principal

| Cenário | KNN (Antes) | KNN (Depois) | Δ KNN | MLP (Antes) | MLP (Depois) | Δ MLP |
|---------|-------------|--------------|-------|-------------|--------------|-------|
| **Random (S1)** | 0.742 | 0.697 | -0.045 | **0.792** | **0.762** | **-0.030** |
| **Scaffold** | 0.560 | 0.549 | -0.011 | 0.568 | 0.580 | +0.012 |
| **Compound (S2)** | 0.648 | 0.631 | -0.017 | 0.674 | 0.694 | +0.020 |
| **Kinase (S3)** | 0.242 | -0.059 | -0.301 | 0.303 | 0.061 | -0.242 |
| **S4 (New C+K)** | 0.216 | -0.006 | -0.222 | 0.233 | 0.164 | -0.069 |

### 2.2 AUROC

| Cenário | KNN (Antes) | KNN (Depois) | Δ KNN | MLP (Antes) | MLP (Depois) | Δ MLP |
|---------|-------------|--------------|-------|-------------|--------------|-------|
| **Random (S1)** | 0.937 | 0.923 | -0.014 | **0.958** | **0.952** | **-0.006** |
| **Scaffold** | 0.846 | 0.845 | -0.001 | 0.860 | 0.874 | +0.014 |
| **Compound (S2)** | 0.891 | 0.883 | -0.008 | 0.913 | 0.917 | +0.004 |
| **Kinase (S3)** | 0.663 | 0.462 | -0.201 | 0.691 | 0.532 | -0.159 |
| **S4 (New C+K)** | 0.635 | 0.514 | -0.121 | 0.642 | 0.585 | -0.057 |

### 2.3 Visualização do Impacto (MLP MCC)

```
Cenário               ANTES      DEPOIS     Mudança
─────────────────────────────────────────────────────
Random (S1)      ████████████████  0.792
                 ███████████████   0.762    ▼ -3.8%

Compound (S2)    █████████████     0.674
                 ██████████████    0.694    ▲ +3.0%

Scaffold         ███████████       0.568
                 ████████████      0.580    ▲ +2.1%

Kinase (S3)      ██████            0.303
                 █                 0.061    ▼ -79.9%

S4 (New C+K)     █████             0.233
                 ███               0.164    ▼ -29.6%
─────────────────────────────────────────────────────
                 0.0   0.2   0.4   0.6   0.8   1.0
```

---

## 3. Análise do Fator de Inflação

> **Figuras de referência**: A-07, D-07 (`07_inflated_vs_real_performance.png`)

O **fator de inflação** mede quanto o Random Split superestima a performance real (S4).

### 3.1 Cálculo do Fator de Inflação (MLP MCC)

| Métrica | ANTES | DEPOIS |
|---------|-------|--------|
| Random MCC | 0.792 | 0.762 |
| S4 MCC | 0.233 | 0.164 |
| **Fator de Inflação** | **3.4x** | **4.6x** |

### 3.2 Interpretação

O fator de inflação **aumentou** após aplicar o filtro de compostos monotônicos. Isso significa:

1. **Os compostos monotônicos mascaravam a dificuldade real** do problema
2. **Sem eles, a diferença entre Random e S4 fica mais evidente**
3. **O modelo tem menos "atalhos" para memorização**

---

## 4. Análise por Tipo de Split

> **Figuras de referência**: A-01, D-01 (`01_leakage_analysis.png`) e A-02, D-02 (`02_baseline_comparison.png`)

### 4.1 Random Split (S1) - Baseline com Vazamento

| Métrica | ANTES | DEPOIS | Interpretação |
|---------|-------|--------|---------------|
| Vazamento de compostos | 59.1% | 47.4% | Menos compostos repetidos |
| MLP MCC | 0.792 | 0.762 | Performance ainda alta (vazamento) |
| MLP AUROC | 0.958 | 0.952 | Queda mínima |

**Conclusão**: O vazamento ainda domina, mas a performance caiu ligeiramente porque há menos casos triviais.

### 4.2 Compound Split (S2) - Cold-Drug

| Métrica | ANTES | DEPOIS | Interpretação |
|---------|-------|--------|---------------|
| Vazamento | 0% | 0% | Sem vazamento |
| MLP MCC | 0.674 | 0.694 | **Melhorou!** |
| MLP AUROC | 0.913 | 0.917 | **Melhorou!** |

**Conclusão**: A performance **melhorou** porque o modelo agora aprende padrões químicos genuínos, não memorização de compostos triviais.

### 4.3 Kinase Split (S3) - Cold-Target

| Métrica | ANTES | DEPOIS | Interpretação |
|---------|-------|--------|---------------|
| Vazamento de compostos | 39.2% | 16.2% | Reduziu significativamente |
| MLP MCC | 0.303 | 0.061 | **Colapsou para chance** |
| MLP AUROC | 0.691 | 0.532 | **Próximo de aleatório** |

**Conclusão**: Este é o resultado mais dramático. O modelo **não generaliza para kinases novas** no dataset non-human. Os compostos monotônicos estavam inflando artificialmente esta métrica.

### 4.4 S4 (New Compound + New Kinase) - Generalização Real

| Métrica | ANTES | DEPOIS | Interpretação |
|---------|-------|--------|---------------|
| Vazamento | 0% | 0% | Sem vazamento |
| MLP MCC | 0.233 | 0.164 | Queda de 29.6% |
| MLP AUROC | 0.642 | 0.585 | Queda de 8.9% |

**Conclusão**: A generalização real é **muito limitada** no dataset non-human, especialmente após remover casos triviais.

---

## 5. Estatísticas dos Splits

### 5.1 Tamanho das Partições

| Cenário | Train (Antes) | Train (Depois) | Δ |
|---------|---------------|----------------|---|
| Random | 11,264 | 8,597 | -2,667 |
| Compound | 11,264 | 8,597 | -2,667 |
| Scaffold | 11,264 | 8,597 | -2,667 |
| Kinase | 11,264 | 8,597 | -2,667 |
| S4 | 11,408 | 8,707 | -2,701 |

### 5.2 Vazamento de Compostos

| Cenário | Vazamento (Antes) | Vazamento (Depois) | Δ |
|---------|-------------------|--------------------|----|
| Random | 59.1% | 47.4% | -11.7pp |
| Kinase | 39.2% | 16.2% | -23.0pp |
| Compound | 0% | 0% | - |
| Scaffold | 0% | 0% | - |
| S4 | 0% | 0% | - |

---

## 6. Discussão

### 6.1 Por que o Split por Kinase (S3) Colapsou?

> **Figuras de referência**: A-04, D-04 (`04_compound_consistency.png`) - mostram a distribuição de compostos monotônicos

O MCC do S3 caiu de **0.303 → 0.061** (praticamente chance). Isso revela:

1. **Compostos monotônicos eram "âncoras"**: Compostos que sempre são ativos/inativos independente da kinase ajudavam o modelo a "acertar" mesmo em kinases novas
2. **Kinases non-human são muito diversas**: Kinases de diferentes espécies (bactérias, parasitas, fungos) têm pouca homologia
3. **O modelo não aprendeu features transferíveis**: Sem os atalhos, o modelo não consegue generalizar

### 6.2 Por que o Compound Split (S2) Melhorou?

O MCC do S2 subiu de **0.674 → 0.694**. Possíveis explicações:

1. **Menos ruído no treinamento**: Compostos monotônicos podem introduzir "ruído" no aprendizado
2. **Foco em padrões químicos**: Sem casos triviais, o modelo é forçado a aprender química
3. **Melhor calibração**: As predições ficam mais calibradas

### 6.3 Limitações do Dataset Non-Human

> **Figuras de referência**: A-03, D-03 (`03_kinase_imbalance.png`) e A-05, D-05 (`05_similarity_analysis.png`)

| Limitação | Impacto | Figura |
|-----------|---------|--------|
| **Poucas kinases** (112) | Dificulta generalização para kinases novas | A-03, D-03 |
| **93.6% compostos testados em 1 kinase** | Não há dados para aprender seletividade | A-04, D-04 |
| **Kinases de espécies diversas** | Baixa homologia entre targets | A-05, D-05 |
| **Alto desbalanceamento** | 53.6% das kinases são desbalanceadas | A-03, D-03 |

---

## 7. Recomendações

### 7.1 Para Avaliação Rigorosa

| Recomendação | Justificativa |
|--------------|---------------|
| ✅ **Use ambos os filtros** | Remove casos triviais |
| ✅ **Priorize o dataset Human** | Mais kinases, mais dados multi-kinase |
| ✅ **Avalie em S2/S3/S4** | Random Split é enganoso |
| ✅ **Reporte MCC, não apenas AUROC** | MCC é mais robusto a desbalanceamento |

### 7.2 Para Publicação Científica

```
Métricas a reportar:
1. Random Split (S1) - como baseline para comparação com literatura
2. Compound Split (S2) - avaliação cold-drug
3. S4 (New Comp + New Kinase) - generalização real
4. Fator de inflação (S1/S4) - quantifica o viés
```

---

## 8. Arquivos de Referência

| Arquivo | Descrição |
|---------|-----------|
| `results/non_human_13_02_2026/` | Resultados ANTES (só filtro kinases) |
| `results/non_human_monotonic_17_02_2026/` | Resultados DEPOIS (filtro kinases + compostos) |
| `KINASE_COMPOUND_EXTREME_PROFILES_REPORT.md` | Estatísticas de monotonia |
| `SPLIT_RESULTS.md` | Análise detalhada dos splits |

---

## 9. Conclusão Final

O filtro de compostos monotônicos **revela a verdadeira dificuldade** do problema de predição de interação proteína-ligante:

| Cenário | Conclusão |
|---------|-----------|
| **Random Split** | Ainda superestima performance (vazamento persiste) |
| **Compound Split** | Melhora ligeiramente (menos ruído) |
| **Kinase Split** | Colapsa para chance (modelo não generaliza) |
| **S4** | Cai 30% (cases triviais mascaravam dificuldade) |

**Mensagem principal**: O dataset Non-Human tem limitações estruturais que impedem boa generalização. Para avaliações rigorosas, **use o dataset Human com ambos os filtros ativados**.
