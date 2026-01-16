# Relatório de Análise: Random vs Stratified Split

**Data:** 15 de Janeiro de 2026  
**Dataset:** Kinases Não-Humanas (ChEMBL)  
**Amostras:** 15,616 pares proteína-ligante  
**Classes:** 56.8% ativos / 43.2% inativos

---

## 📊 Resumo Executivo

Comparação entre estratégias de split de dados:
- **Random Split:** Divisão aleatória com seed fixa (420), mantendo proporção de classes
- **Stratified Split:** Divisão baseada em clustering de similaridade (Agglomerative Hierarchical)

### Resultado Principal

| Classificador | Random (μ AUC) | Stratified (μ AUC) | Δ (%) |
|---------------|----------------|---------------------|-------|
| **KNN**       | 0.9373         | 0.9440              | +0.72% |
| **MLP**       | 0.9586         | 0.9605              | +0.19% |

**Conclusão:** Delta **extremamente pequeno** (< 1%) indica que **ambas as estratégias são equivalentes** para este dataset específico.

---

## 🔬 Análise Detalhada

### 1. Verificação de Overfitting (Train vs Test Gap)

| Modelo | Classificador | Train AUC | Test AUC | Gap | Diagnóstico |
|--------|---------------|-----------|----------|-----|-------------|
| ESM2-8M | KNN | 0.9985 | 0.9412 | **5.73%** | ⚠️ Overfitting |
| ESM2-150M | KNN | 0.9993 | 0.9368 | **6.26%** | ⚠️ Overfitting |
| ESM2-3B | KNN | 0.9998 | 0.9338 | **6.60%** | ⚠️ Overfitting |
| ESM2-8M | MLP | 0.9964 | 0.9586 | 3.78% | ~ Leve |
| ESM2-150M | MLP | 0.9957 | 0.9554 | 4.03% | ~ Leve |
| ESM2-3B | MLP | 0.9957 | 0.9619 | 3.38% | ~ Leve |

**Observação:** KNN apresenta overfitting mais acentuado (memoriza treino), enquanto MLP generaliza melhor devido ao early stopping.

### 2. Médias por Métrica

#### KNN (média dos 3 modelos ESM-2)
| Métrica | Random | Stratified | Δ |
|---------|--------|------------|---|
| Accuracy | 0.8796 | 0.8899 | +1.16% |
| Precision | 0.8849 | 0.8853 | +0.04% |
| Recall | 0.9062 | 0.9241 | +1.98% |
| F1-Score | 0.8954 | 0.9042 | +0.98% |
| MCC | 0.7540 | 0.7759 | +2.90% |
| ROC-AUC | 0.9373 | 0.9440 | +0.72% |

#### MLP (média dos 3 modelos ESM-2)
| Métrica | Random | Stratified | Δ |
|---------|--------|------------|---|
| Accuracy | 0.8944 | 0.8940 | -0.05% |
| Precision | 0.9046 | 0.9014 | -0.36% |
| Recall | 0.9103 | 0.9110 | +0.08% |
| F1-Score | 0.9074 | 0.9062 | -0.14% |
| MCC | 0.7846 | 0.7843 | -0.03% |
| ROC-AUC | 0.9586 | 0.9605 | +0.19% |

---

## 🔍 Análise da Homogeneidade dos Dados

### Estatísticas de Similaridade de Cosseno

**Importante:** A similaridade é calculada sobre os **embeddings concatenados** (proteína ESM-2 + ligante SMI-TED), não apenas sobre sequências proteicas.

| Modelo | Dimensão | Clusters | Similaridade Média | Std Dev |
|--------|----------|----------|-------------------|---------|
| ESM2-8M | 1088 (320+768) | 145 | **0.9825** | 0.0075 |
| ESM2-150M | 1408 (640+768) | 150 | **0.9839** | 0.0058 |
| ESM2-3B | 3328 (2560+768) | 144 | **0.9821** | 0.0214 |

### Distribuição de Similaridade

```
Mínima:     ~0.93 - 0.94
P25:        ~0.978 - 0.981
P50:        ~0.984 - 0.985  (mediana)
P75:        ~0.987 - 0.988
P90:        ~0.990 - 0.991
P95:        ~0.992 - 0.993
```

**Implicação:** 75% dos pares de amostras têm similaridade > 0.987 — dataset extremamente homogêneo.

---

## 💡 Interpretação dos Resultados

### Por que o delta é tão pequeno?

#### Hipótese Inicial
> "Estratificação deveria **reduzir** métricas ao evitar data leakage"

#### Resultado Observado
- KNN: Stratified ligeiramente **melhor** (+0.72% AUC)
- MLP: Stratified praticamente **igual** (+0.19% AUC)

#### Explicações

1. **Não há leakage significativo no split aleatório**
   - Dataset muito homogêneo (similaridade média ~98%)
   - Mesmo split aleatório distribui amostras similares uniformemente
   - Porque **todas** as amostras são similares entre si

2. **Estratificação cria splits mais "balanceados"**
   - Clustering agrupa proteínas/ligantes similares
   - Divisão proporcional em cada cluster = melhor representação
   - Pode explicar leve melhora no KNN

3. **O problema é a natureza do dataset, não o split**
   - 15,616 pares com alta similaridade
   - Kinases não-humanas = família evolutivamente relacionada
   - Ligantes de kinases compartilham scaffolds comuns

---

## 🎯 Conclusões

### ✅ Positivo

1. **Pipeline estável e reprodutível**
   - Resultados consistentes independente da estratégia de split
   - Variância entre modelos ESM-2 é pequena (std ~0.003-0.004)

2. **Modelos generalizam razoavelmente**
   - ROC-AUC de 0.94-0.96 no teste
   - MLP supera KNN consistentemente

3. **Embeddings de qualidade**
   - Alta consistência entre ESM-2 8M, 150M e 3B
   - Modelo menor (8M) competitivo com maiores

### ⚠️ Atenção

1. **Overfitting detectado no KNN**
   - Gap train-test de 5.7-6.6%
   - KNN memoriza treino (comportamento esperado)

2. **Alta homogeneidade pode mascarar problemas**
   - Generalização para kinases "diferentes" não testada
   - Dataset pode ter viés de amostragem

3. **Validação externa necessária**
   - Testar em kinases humanas
   - Comparar com benchmarks públicos

---

## 📋 Recomendações para Tese

### 1. Documentação
- [x] Registrar alta similaridade intra-dataset
- [x] Documentar gap train-test por modelo
- [ ] Explicar escolha de split aleatório (justificado pelos resultados)

### 2. Validação Adicional
- [ ] Teste em dataset externo (kinases humanas)
- [ ] Leave-family-out cross-validation
- [ ] Comparação com benchmarks publicados

### 3. Análise Complementar
- [ ] Diversidade química dos ligantes (Tanimoto)
- [ ] Distribuição de famílias de kinases
- [ ] Identificação de possíveis outliers

---

## 📁 Arquivos Gerados

```
results/
├── baseline_random_split/
│   ├── baseline_results.json          # Resultados completos (random)
│   └── esm2_*/                         # Subdiretórios por modelo
│
└── stratified_baseline/
    ├── stratified_results.json        # Resultados completos (stratified)
    ├── comparison_random_vs_stratified.png  # Gráfico 2x3 comparativo
    ├── comparison_delta.png           # Gráfico de diferenças (Δ)
    └── esm2_*/                         # Subdiretórios por modelo
```

---

**Gerado automaticamente em:** 2026-01-15  
**Scripts:** `baseline.py` | `stratified_baseline.py`
