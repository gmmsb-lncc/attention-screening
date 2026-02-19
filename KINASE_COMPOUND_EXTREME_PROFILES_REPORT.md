# Relatório: Kinases e Compostos com Perfis Extremos de Atividade

**Data**: 2026-02-17
**Figuras de Referência**: `04_compound_consistency.png` (NH-04, H-04) e `03_kinase_imbalance.png` (NH-03, H-03)
**Datasets**: Non-Human e Human

---

## Sumário Executivo

Este relatório analisa kinases e compostos com perfis extremos de atividade (100% ativos ou 100% inativos), que podem influenciar artificialmente o treinamento de modelos.

### Achados Principais

| Métrica | Non-Human | Human |
|---------|-----------|-------|
| **Kinases monotônicas** | 117 (50.6%) | 73 (12.4%) |
| **Amostras em kinases monotônicas** | 1,536 (9.8%) | 1,953 (0.4%) |
| **Compostos pan-ativos/inativos** | 1,296 (75% dos multi-kinase) | 29,768 (64% dos multi-kinase) |
| **Amostras "triviais"** | 5,103 (32.7%) | 100,599 (21.1%) |

---

## PARTE 1: Análise de Kinases

### 1.1 Distribuição de Kinases por Perfil de Atividade

**Figuras**: NH-03 (`results/non_human_13_02_2026/non_human/03_kinase_imbalance.png`), H-03 (`results/human_13_02_2026/human/03_kinase_imbalance.png`)

#### Non-Human (231 kinases)

| Categoria | Kinases | % | Amostras | % |
|-----------|---------|---|----------|---|
| **100% Ativas** | 58 | 25.1% | 1,172 | 7.5% |
| >80% Ativas | 25 | 10.8% | 4,216 | 27.0% |
| Balanceadas (20-80%) | 65 | 28.1% | 8,355 | 53.5% |
| <20% Ativas | 24 | 10.4% | 1,509 | 9.7% |
| **100% Inativas** | 59 | 25.5% | 364 | 2.3% |
| **TOTAL MONOTÔNICAS** | **117** | **50.6%** | **1,536** | **9.8%** |
| TOTAL DESBALANCEADAS | 166 | 71.9% | 7,261 | 46.5% |

#### Human (590 kinases)

| Categoria | Kinases | % | Amostras | % |
|-----------|---------|---|----------|---|
| **100% Ativas** | 38 | 6.4% | 495 | 0.1% |
| >80% Ativas | 22 | 3.7% | 24,589 | 5.2% |
| Balanceadas (20-80%) | 326 | 55.3% | 385,536 | 81.0% |
| <20% Ativas | 169 | 28.6% | 63,635 | 13.4% |
| **100% Inativas** | 35 | 5.9% | 1,458 | 0.3% |
| **TOTAL MONOTÔNICAS** | **73** | **12.4%** | **1,953** | **0.4%** |
| TOTAL DESBALANCEADAS | 264 | 44.7% | 90,177 | 19.0% |

### 1.2 Lista de Kinases 100% Ativas (Top 20)

#### Non-Human

| Kinase | Amostras | Compostos |
|--------|----------|-----------|
| Calcium-dependent protein kinase 4 | 460 | 460 |
| Mitogen-activated protein kinase 3 | 90 | 90 |
| Serine/threonine-protein kinase pim-3 | 68 | 58 |
| Non-receptor tyrosine-protein kinase TYK2 | 24 | 24 |
| Tyrosine-protein kinase JAK1/JAK3 | 8 | 4 |
| Tyrosine-protein kinase JAK1/JAK2 | 6 | 3 |
| Aurora kinase B-B | 6 | 6 |
| MAP/microtubule affinity-regulating kinase 4 | 4 | 4 |
| Aurora kinase B-A | 3 | 3 |
| Tyrosine-protein kinase JAK1 | 3 | 3 |

#### Human

| Kinase | Amostras | Compostos |
|--------|----------|-----------|
| Ribosomal protein S6 kinase | 88 | 11 |
| Cyclin-dependent kinase 4/G1/S-specific cyclin-E1 | 62 | 31 |
| Inositol-trisphosphate 3-kinase B | 43 | 43 |
| E3 ubiquitin-protein ligase XIAP/Receptor-interacting serine/threonine-protein kinase 2 | 42 | 20 |
| Serine/threonine-protein kinase 19 | 25 | 17 |
| Cyclin-dependent kinase 1/G1/S-specific cyclin-D1 | 8 | 4 |
| RAF serine/threonine protein kinase | 6 | 2 |

### 1.3 Lista de Kinases 100% Inativas (Top 20)

#### Non-Human

| Kinase | Amostras | Compostos |
|--------|----------|-----------|
| KinA/Spo0F (sporulation kinase A) | 88 | 42 |
| Thymidylate kinase, putative | 19 | 19 |
| CaM kinase I alpha | 13 | 13 |
| Phosphoglycerate kinase 1 | 10 | 10 |
| Autoinducer 1 sensor kinase/phosphatase luxN | 10 | 9 |
| Ca2+/calmodulin-dependent protein kinase II | 8 | 2 |
| Proto-oncogene tyrosine-protein kinase ROS | 6 | 6 |
| CAI-1 autoinducer sensor kinase/phosphatase CqsS | 6 | 6 |
| Beta-adrenergic receptor kinase 2 | 6 | 6 |
| Thymidine kinase 2 | 5 | 5 |

#### Human

| Kinase | Amostras | Compostos |
|--------|----------|-----------|
| Choline/ethanolamine kinase | 44 | 44 |
| Phosphatidylinositol 4,5-bisphosphate 3-kinase/Pyruvate dehydrogenase kinase | 28 | 14 |
| PI4-kinase type II | 27 | 26 |
| Cyclin-dependent kinase 2-associated protein 1 | 24 | 24 |
| Serine/threonine-protein kinase WNK2 | 13 | 12 |
| Dual specificity MAPKK 1/MAPK 1/RAF kinase | 6 | 2 |
| Serine/threonine-protein kinase Chk1/2 | 4 | 2 |
| [3-methyl-2-oxobutanoate dehydrogenase] kinase | 3 | 3 |

### 1.4 Implicações das Kinases Monotônicas

**Problema**: Kinases com 100% de amostras ativas ou inativas não fornecem sinal discriminativo para o modelo aprender.

- **Non-Human**: 50.6% das kinases são monotônicas, mas representam apenas 9.8% das amostras
- **Human**: 12.4% das kinases são monotônicas, representando apenas 0.4% das amostras

**Ação tomada no script**: O filtro `monotonic_kinase_filter=True` remove essas kinases do treinamento, resultando em:
- Non-Human: 114 kinases (de 231) e 14,080 amostras (de 15,616)
- Human: 517 kinases (de 590) e 473,760 amostras (de 475,713)

---

## PARTE 2: Análise de Compostos

### 2.1 Compostos por Número de Kinases Testadas

**Figuras**: NH-04 (`results/non_human_13_02_2026/non_human/04_compound_consistency.png`), H-04 (`results/human_13_02_2026/human/04_compound_consistency.png`)

| Dataset | 1 Kinase | >1 Kinase | Total |
|---------|----------|-----------|-------|
| **Non-Human** | 6,401 (78.7%) | 1,730 (21.3%) | 8,131 |
| **Human** | 89,940 (66.0%) | 46,415 (34.0%) | 136,355 |

### 2.2 Consistência de Compostos Multi-Kinase

Análise de como compostos testados contra múltiplas kinases se comportam:

#### Non-Human (1,730 compostos multi-kinase)

| Categoria | Compostos | % | Amostras | % do Total |
|-----------|-----------|---|----------|------------|
| **Pan-ativos (100% ativos)** | 765 | 44.2% | 2,603 | 16.7% |
| **Pan-inativos (0% ativos)** | 531 | 30.7% | 1,636 | 10.5% |
| Inconsistentes (variam) | 434 | 25.1% | 2,023 | 13.0% |

#### Human (46,415 compostos multi-kinase)

| Categoria | Compostos | % | Amostras | % do Total |
|-----------|-----------|---|----------|------------|
| **Pan-ativos (100% ativos)** | 23,589 | 50.8% | 71,424 | 15.0% |
| **Pan-inativos (0% ativos)** | 6,179 | 13.3% | 27,741 | 5.8% |
| Inconsistentes (variam) | 16,647 | 35.9% | 255,225 | 53.7% |

### 2.3 Top Compostos Pan-Ativos

Compostos que são 100% ativos contra TODAS as kinases testadas:

#### Non-Human

| ChEMBL ID | Kinases Testadas | Amostras |
|-----------|------------------|----------|
| CHEMBL1803085 | 6 | 9 |
| CHEMBL26501 | 5 | 13 |
| CHEMBL3092862 | 5 | 6 |
| CHEMBL3092863 | 5 | 6 |
| CHEMBL5179655 | 5 | 6 |
| CHEMBL538718 | 5 | 13 |

#### Human

| ChEMBL ID | Kinases Testadas | Amostras |
|-----------|------------------|----------|
| CHEMBL4088216 | 251 | 254 |
| CHEMBL4549667 | 251 | 253 |
| CHEMBL215803 | 57 | 60 |
| CHEMBL1988581 | 54 | 54 |
| CHEMBL2443026 | 38 | 40 |
| CHEMBL4474690 | 36 | 44 |

**Nota**: Compostos como CHEMBL4088216 que são ativos contra 251 kinases são provavelmente inibidores promíscuos (pan-kinase inhibitors) ou artefatos experimentais.

### 2.4 Top Compostos Pan-Inativos

Compostos que são 0% ativos contra TODAS as kinases testadas:

#### Non-Human

| ChEMBL ID | Kinases Testadas | Amostras |
|-----------|------------------|----------|
| CHEMBL2037208 | 11 | 11 |
| CHEMBL2037226 | 11 | 11 |
| CHEMBL1803080 | 6 | 7 |
| CHEMBL1803081 | 6 | 9 |
| CHEMBL153548 | 4 | 17 |

#### Human

| ChEMBL ID | Kinases Testadas | Amostras |
|-----------|------------------|----------|
| CHEMBL2001019 | 219 | 261 |
| CHEMBL1201182 | 210 | 211 |
| CHEMBL2103839 | 201 | 201 |
| CHEMBL3991927 | 200 | 200 |
| CHEMBL2323775 | 198 | 198 |

**Nota**: Compostos inativos contra 200+ kinases são provavelmente controles negativos ou compostos com baixa afinidade geral.

---

## PARTE 3: Cruzamento Kinase × Composto

### 3.1 Sobreposição de Casos Extremos

| Categoria | Non-Human | % | Human | % |
|-----------|-----------|---|-------|---|
| Amostras em kinases monotônicas | 1,536 | 9.8% | 1,953 | 0.4% |
| Amostras com compostos pan-ativos/inativos | 4,239 | 27.1% | 99,165 | 20.8% |
| Amostras em AMBOS | 672 | 4.3% | 519 | 0.1% |
| **Total "triviais" (união)** | **5,103** | **32.7%** | **100,599** | **21.1%** |

### 3.2 Interpretação

#### Non-Human
- **32.7% das amostras são "triviais"**: o modelo pode prever corretamente apenas sabendo que a kinase é monotônica OU o composto é pan-ativo/inativo
- Isso contribui para a inflação de métricas em random split

#### Human
- **21.1% das amostras são "triviais"**: proporção menor, mas ainda significativa
- A maior proporção de compostos testados contra muitas kinases aumenta a quantidade de compostos pan-ativos/inativos

---

## PARTE 4: Impacto no Treinamento

### 4.1 Por que isso importa?

1. **Memorização vs Generalização**: Modelos podem "decorar" que certas kinases sempre produzem resultado positivo ou negativo, sem aprender features químicas relevantes.

2. **Inflação de Métricas**: Em random split, amostras triviais vazam entre train/test, permitindo que o modelo use atalhos.

3. **Viés de Seleção**: Kinases muito estudadas (com mais amostras) tendem a ser balanceadas; kinases pouco estudadas tendem a ser monotônicas.

### 4.2 Mitigações Implementadas

O script `split_comparison_analysis.py` implementa:

1. **`monotonic_kinase_filter=True`**: Remove kinases com 100% ativos ou 100% inativos
2. **Cenários de split corretos**: S2/S3/S4 garantem que compostos/kinases do teste não vazem para o treino
3. **Métricas robustas**: MCC é insensível a desbalanceamento de classes

### 4.3 Recomendações

| Recomendação | Justificativa |
|--------------|---------------|
| **Sempre usar o filtro monotônico** | Remove kinases sem sinal discriminativo |
| **Reportar distribuição de classes por kinase** | Transparência sobre desbalanceamento |
| **Considerar filtrar compostos pan-ativos** | Podem ser artefatos ou promíscuos |
| **Avaliar em cenários cold-start (S2/S3/S4)** | Única forma de medir generalização real |

---

## Arquivos Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `results/non_human_kinase_activity_profiles.csv` | Estatísticas por kinase (Non-Human) |
| `results/human_kinase_activity_profiles.csv` | Estatísticas por kinase (Human) |
| `results/non_human_compound_activity_profiles.csv` | Estatísticas por composto (Non-Human) |
| `results/human_compound_activity_profiles.csv` | Estatísticas por composto (Human) |

---

## Referências Cruzadas com Figuras

| Figura | Dataset | Conteúdo | Seção Relacionada |
|--------|---------|----------|-------------------|
| NH-03 | Non-Human | Histograma de taxa de atividade por kinase | Parte 1 |
| H-03 | Human | Histograma de taxa de atividade por kinase | Parte 1 |
| NH-04 | Non-Human | Consistência de compostos multi-kinase | Parte 2 |
| H-04 | Human | Consistência de compostos multi-kinase | Parte 2 |

---

## Conclusão

O dataset **Non-Human** é mais problemático:
- 50.6% das kinases são monotônicas
- 32.7% das amostras são triviais

O dataset **Human** é mais robusto:
- Apenas 12.4% das kinases são monotônicas
- 81% das amostras estão em kinases balanceadas

**Recomendação final**: Para avaliação rigorosa, usar o dataset Human com filtro monotônico ativado e cenários S2/S3/S4.
