# 📚 Methodology Documentation

Este diretório contém documentação detalhada sobre a metodologia do DockTKinase.

## 📄 Arquivos Principais

### 1. **METHODOLOGY_REVIEW.md** (1,118 linhas)
Revisão completa e especializada da metodologia do DockTKinase.

**Conteúdo**:
- ✅ Executive summary
- ✅ Contexto científico (super-resistência bacteriana)
- ✅ Pipeline benchmark (5 steps)
  0. **Scaffold split** (Murcko scaffold decomposition)
  1. Level 1: Fingerprint + KNN/MLP (baseline)
  2. Level 2: Embedding vectors + KNN/MLP
  3. Level 3: CNN + Cross-Attention (DT-Kinase, multi-seed)
  4. Comparative report + visualizations
- ✅ Arquitetura de software (SOLID, KISS, Clean Code)
- ✅ Garantias e validações (5 mecanismos principais)
- ✅ Benchmarks e resultados (7 modelos de proteína, 12 algoritmos)
- ✅ Best practices por cenário (produção, pesquisa, prototipagem)
- ✅ Problemas identificados e soluções
- ✅ Roadmap futuro

**Quando usar**: 
- 🎓 Entender metodologia completa
- 📖 Para publicações científicas
- 🔍 Revisar implementação detalhada
- 🏗️ Arquitetura de software

---

## 🎯 Quick Navigation

**Se você quer...**

| Objetivo | Seção | Linha |
|----------|-------|-------|
| Entender o sistema em 2 min | Executive Summary | 1-50 |
| Saber por que previne data leakage | Metodologia de Estratificação | ~350-500 |
| Comparar modelos de proteína | Benchmarks e Resultados | ~750-850 |
| Ver recomendações práticas | Best Practices | ~880-950 |
| Configurar para produção | Best Practices - Produção | ~910-935 |
| Entender arquitetura | Arquitetura de Software | ~650-750 |
| Checar validações | Garantias e Validações | ~650-750 |

---

## 📊 Documentação Relacionada

**Na pasta `docs/`**:
- `02-user-guide/stratification-methodology.md` - Detalhe técnico de estratificação
- `03-architecture/integrated-pipeline.md` - Integração de componentes
- `04-modules/` - Documentação por módulo
- `README.md` - Guia geral do projeto

---

## 🔑 Key Takeaways

### What Makes semantic-screening Unique

1. **Scaffold-Based Splitting**: Murcko scaffold decomposition prevents chemical series leakage
2. **3-Level Benchmark**: Fingerprints → PLM vectors → DT-Kinase (CNN+CrossAttention)
3. **Multi-Seed Evaluation**: 5 seeds for statistical significance
4. **Unified Orchestration**: Single script (`semantic_screening_models_beta.py`) coordinates all levels
5. **Interpretable Deep Learning**: CNN + Cross-Attention with attention maps

### 📈 Performance Summary

| Métrica | Melhor | Modelo |
|---------|--------|--------|
| **Classificação** | ROC-AUC = 0.9731 | ExtraTrees |
| **Regressão** | R² = 0.4397 | RandomForest |
| **Proteína** | 1152-dim | esmc-600m-2024-12 ⭐ |

### ⚠️ Key Limitations

1. **Regressão R² modesto (0.44)**: Estrutura 2D é insuficiente para predizer affinity exato
2. **Ligand embedding não fine-tunable**: FM4M weights proprietários
3. **Sem dados 3D**: Coordenadas cristalográficas aumentariam performance

---

## 🚀 Como Usar Este Documento

### Para Iniciantes
1. Leia Executive Summary (2 minutos)
2. Veja diagrama do Pipeline (5 minutos)
3. Leia "Visão Geral do Projeto" (10 minutos)
4. Explore seção de interesse específica

### Para Especialistas
1. Vá direto para "Metodologia de Estratificação" (80 linhas concentradas)
2. Examine "Arquitetura de Software"
3. Verifique "Garantias e Validações"
4. Compare resultados em "Benchmarks"

### Para Implementadores
1. Entenda fluxo em "Pipeline Completo (7 Fases)"
2. Estude "Arquitetura de Software"
3. Siga "Best Practices"
4. Implemente seguindo código em `src/`

---

## 📞 Referências

- **Código**: `src/build/`, `src/classifier/`, `src/regression/`
- **Testes**: `tests/test_integration/`
- **Exemplos**: `examples/demo_kinase_non_human_pipeline.py`
- **Visualizações**: `scripts/visualize_ml_comparison.py`

---

## 📌 Revisões e Updates

- **v3.0** (Fevereiro 2026): Scaffold splits + Benchmark unificado de 3 níveis
- **v2.0** (Dezembro 2025): Revisão completa com foco em super-resistência bacteriana
- **v1.0**: Documentação inicial

---

**Status**: Production-Ready
**Último update**: Fevereiro 2026
**Manutentor**: GMMSB-LNCC
