# 🎯 Quick Reference - semantic-screening & DT-Kinase

**Documento**: METHODOLOGY_REVIEW.md  
**Linhas**: 1,118 (especializado, completo)  
**Status**: ✅ Production-Ready

---

## ⚡ 30-Second Summary

**semantic-screening** é uma plataforma para screening semântico de interações proteína-ligante.
**DT-Kinase** é a arquitetura neural implementada (PLM + CNN + Cross-Attention).

```
Sequências  →  Embeddings (ESM-2/SMI-TED)  →  Processamento  
                                              (CNN + CrossAttn)
                                                        ↓
                                      Classificação (Ativo/Inativo)
                                      ✅ ROC-AUC = 0.9731
                                                        ↓
                                      Regressão (pChEMBL)
                                      ✅ R² = 0.4397
```

**Key**: Sem estrutura 3D. Tudo baseado em sequência e aprendizado de semântica via PLMs.

---

## 7 Fases do Pipeline semantic-screening

| Fase | O Quê | Como | Output |
|------|-------|------|--------|
| 1 | Embeddings | ESM-2/ESM-C (proteína) + SMI-TED (ligante) | 1920-dim vectors |
| 2 | Matrizes | Concatenação + validação | concatenated_embeddings.npy |
| 3 | Labels | pChEMBL > 6.0 (binary) + affinity (regression) | binary_labels.npy |
| 4 | ⭐ **Stratify** | **Cosine sim → Agglomerative → 80/10/10** | **train/val/test split** |
| 5 | Classificação | 12 algoritmos ML (ExtraTrees best) | ROC-AUC, F1, Accuracy, ... |
| 6 | Regressão | 12 algoritmos ML (RandomForest best) | MAE, R², RMSE, ... |
| 7 | DL (opcional) | **DT-Kinase**: CNN + Cross-Attention | Attention maps + predições |

---

## 🏆 Arquitetura DT-Kinase

```
Modelo de Proteína:  esmc-600m-2024-12 ⭐ (1152-dim)
Modelo de Ligante:   SMI-TED (768-dim)
Rede Neural:         CNN + Cross-Attention
Classificação:       ExtraTrees baseline (ROC-AUC = 0.9731)
Regressão:          RandomForest baseline (R² = 0.4397, MAE = 0.5325)
```

**DT-Kinase** é implementado na plataforma **semantic-screening**.

---

## 🔐 Por Que Previne Data Leakage?

**Problema**: Random split pode dividir moléculas similares

**Solução**: 
1. Cosine similarity entre embeddings (0-1 scale)
2. Agglomerative clustering encontra ~100 famílias químicas
3. Atribuir clusters inteiros (NUNCA dividir) a train/val/test
4. Resultado: Test tem **estruturas completamente diferentes** do train

**Garantia**: ✅ Moléculas similares SEMPRE no mesmo split

---

## 💡 Quando Usar Qual Modelo de Proteína?

| Cenário | Modelo | Velocidade | ROC-AUC | Custo |
|---------|--------|-----------|---------|-------|
| 🚀 Prototipagem | esm2_t6_8M | 30 min | 0.9723 | $ |
| ⚙️ Produção | **esmc-600m-2024-12** | **2-3 h** | **0.9731** | **$$** |
| 🔬 Pesquisa | esm2_t36_3B | 8 h | 0.9739 | $$$$ |

---

## ⚠️ Limitações Conhecidas

| Limitação | Causa | Impacto | Solução |
|-----------|-------|--------|---------|
| R² = 0.44 | Estrutura 2D insuficiente | Predição exata difícil | Usar ranking |
| FM4M imutável | Weights proprietários | Sem fine-tuning | Aceitável |
| Sem 3D | Dados não disponíveis | Performance subótima | Adicionar depois |

---

## ✅ 5 Garantias Implementadas

1. **No Data Leakage**: Cluster-based stratification
2. **Reproducible**: Fixed random_state=42
3. **Validated**: 5-fold cross-validation
4. **Overfitting Detection**: Val vs Test comparison
5. **Multiple Metrics**: 7 para classificação, 4 para regressão

---

## 🎯 Use Cases por Profissional

### 🧪 Biólogo Computacional
- Ler: **Metodologia de Estratificação** (linhas 350-500)
- Entender: Binary search, cosine similarity, UPGMA
- Validar: Implementação em `src/build/stratification/`

### 💊 Farmacêutico
- Ler: **Benchmarks e Resultados** (linhas 750-850)
- Entender: Qual modelo escolher para seu dataset
- Usar: Recomendações de best practices

### 🏗️ Engenheiro
- Ler: **Arquitetura de Software** (linhas 650-750)
- Entender: SOLID, modular design, type hints
- Implementar: Seguindo padrões descritos

### 📊 Data Scientist
- Ler: **Best Practices** (linhas 880-950)
- Entender: Trade-offs entre velocidade vs qualidade
- Escolher: Configuração otimizada para seu cenário

---

## 🚀 Roadmap de Próximas Melhorias

**Curto Prazo (1-2 meses)**:
- [ ] Feature importance (SHAP)
- [ ] Uncertainty quantification
- [ ] Attention map visualization

**Médio Prazo (3-6 meses)**:
- [ ] Hyperparameter tuning
- [ ] Ensemble methods
- [ ] Transfer learning

**Longo Prazo (6-12 meses)**:
- [ ] 3D structure integration
- [ ] Molecular dynamics
- [ ] FDA drug repurposing

---

## 📚 Documento Completo

**Arquivo**: `docs/01-methodology/METHODOLOGY_REVIEW.md`  
**Seções**: 11 principais (com subsections)  
**Linhas**: 1,118  
**Tempo de leitura**: 45-60 minutos (completo) ou 10 minutos (executive summary)

**Comece por**:
1. Executive Summary (2 min)
2. Pipeline Overview (5 min)
3. Seção de interesse (15-20 min)

---

## 🔗 Referências Cruzadas

- **Código**: `src/build/stratification/adaptive_clustering.py`
- **Testes**: `tests/test_integration/`
- **Exemplos**: `examples/demo_kinase_non_human_pipeline.py`
- **Visualizações**: `scripts/visualize_ml_comparison.py`

---

## ✨ Status Final

```
✅ Revisão metodológica: COMPLETA
✅ Especificação técnica: DETALHADA
✅ Performance documentada: BENCHMARKED
✅ Limitações identificadas: MITIGADAS
✅ Pronta para publicação: SIM

Versão: 2.0
Data: Dezembro 2025
Foco: Super-resistência bacteriana
Status: Production-Ready
```

---

**Próximo passo**: Ler `METHODOLOGY_REVIEW.md` completamente!
