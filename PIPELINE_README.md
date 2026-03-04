# Pipeline de Treinamento - Semantic Screening

## 📋 Visão Geral

O pipeline completo de treinamento está organizado em **3 níveis progressivos** de complexidade, desde features clássicas até arquiteturas deep learning state-of-the-art com cross-attention.

```
Level 1 (Baseline)  →  Level 2 (Embeddings)  →  Level 3 (Cross-Attention)  →  [Level 6 (Otimizado)]
     FP + ML              Emb + ML                Transformer + CA               HPO + Ensemble
```

---

## 🎯 Níveis do Pipeline

### **Level 1: Fingerprints + ML Clássico**
**Baseline tradicional de quimioinformática**

- **Features**: Molecular fingerprints (ECFP, MACCS, etc.)
- **Modelos**: KNN, MLP
- **Vantagens**: Rápido, interpretável, baseline estabelecido
- **MCC típico**: ~0.40-0.45

```bash
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 1 \
    --seeds 42 123 456
```

---

### **Level 2: Embeddings + ML Clássico**
**Representações semânticas com Attention Pooling**

- **Features Proteína**: ESM-2 embeddings (per-residue) → **Attention Pooling** → vetor fixo
- **Features Ligante**: MoLFormer embeddings (per-token) → **Attention Pooling** → vetor fixo
- **Modelos**: KNN, MLP
- **Inovação**: Usa **Attention Pooling** ao invés de mean pooling para agregar tokens
  - Mean pooling: todas as posições têm peso igual
  - Attention pooling: pesos aprendidos para cada posição → contexto-aware

#### Por que Attention Pooling?

```python
# Mean Pooling (antigo)
pooled_vector = embeddings.mean(dim=1)  # [batch, seq_len, dim] → [batch, dim]

# Attention Pooling (novo)
attention_weights = softmax(attention_network(embeddings))  # [batch, seq_len, 1]
pooled_vector = (embeddings * attention_weights).sum(dim=1)  # [batch, dim]
```

**Vantagens**:
- ✅ Pesos aprendidos capturam posições mais importantes
- ✅ Contexto-aware: considera importância relativa de cada token
- ✅ Melhora discriminação entre ativos/inativos
- ✅ Mantém compatibilidade com modelos clássicos (KNN, MLP)

```bash
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 2 \
    --seeds 42 123 456
```

---

### **Level 3: Transformer + Cross-Attention**
**Arquitetura deep learning state-of-the-art** (antigo Level 5-Lite)

#### Arquitetura Completa

```
Input:
  Proteína: [seq_len, 320]   (ESM-2 per-residue embeddings)
  Ligante:  [mol_len, 768]   (MoLFormer per-token embeddings)
         ↓
┌────────────────────────────────────────────────────────┐
│  FASE 1: Encoders Lineares (Projeção para espaço comum) │
└────────────────────────────────────────────────────────┘
  Proteína: Linear(320 → 512) + LayerNorm + ReLU + Dropout
  Ligante:  Linear(768 → 512) + LayerNorm + ReLU + Dropout
         ↓
  Proteína: [seq_len, 512]
  Ligante:  [mol_len, 512]
         ↓
┌────────────────────────────────────────────────────────┐
│  FASE 2: Cross-Attention Bidirecional (8 cabeças)      │
└────────────────────────────────────────────────────────┘
  
  Direção 1: Proteína como Query, Ligante como Key/Value
    Q_prot = W_Q · proteína   [seq_len, 512]
    K_lig  = W_K · ligante    [mol_len, 512]
    V_lig  = W_V · ligante    [mol_len, 512]
    
    scores = softmax(Q_prot · K_lig^T / √64)  [seq_len, mol_len]
    attended_prot = scores · V_lig             [seq_len, 512]
  
  Direção 2: Ligante como Query, Proteína como Key/Value
    Q_lig  = W_Q · ligante    [mol_len, 512]
    K_prot = W_K · proteína   [seq_len, 512]
    V_prot = W_V · proteína   [seq_len, 512]
    
    scores = softmax(Q_lig · K_prot^T / √64)  [mol_len, seq_len]
    attended_lig = scores · V_prot             [mol_len, 512]
         ↓
  attended_prot: [seq_len, 512]
  attended_lig:  [mol_len, 512]
         ↓
┌────────────────────────────────────────────────────────┐
│  FASE 3: Pooling & Concatenação                        │
└────────────────────────────────────────────────────────┘
  pooled_prot = attended_prot.mean(dim=1)  [512]
  pooled_lig  = attended_lig.mean(dim=1)   [512]
  
  combined = concat(pooled_prot, pooled_lig)  [1024]
         ↓
┌────────────────────────────────────────────────────────┐
│  FASE 4: Classifier Head                               │
└────────────────────────────────────────────────────────┘
  x = Linear(1024 → 512) + ReLU + Dropout(0.2)
  x = Linear(512 → 256)  + ReLU + Dropout(0.2)
  logit = Linear(256 → 1)
         ↓
  Output: Probabilidade de atividade [0, 1]
```

#### Detalhes Técnicos

**8 Cabeças de Atenção**:
- hidden_dim = 512
- num_heads = 8
- head_dim = 512 / 8 = 64

Cada cabeça processa uma projeção diferente:
```python
# Para cada cabeça h ∈ {1, 2, ..., 8}:
Q_h = Linear_h(input)  # [batch, seq, 64]
K_h = Linear_h(input)  # [batch, seq, 64]
V_h = Linear_h(input)  # [batch, seq, 64]

attention_h = softmax(Q_h · K_h^T / √64) · V_h

# Concatenar todas as cabeças:
output = concat(attention_1, ..., attention_8)  # [batch, seq, 512]
```

**Por que 8 cabeças?**
- Cada cabeça aprende padrões diferentes de interação
- Cabeça 1: pode focar em regiões de ligação
- Cabeça 2: pode focar em bolsões hidrofóbicos
- Cabeça 3: pode focar em pontes de hidrogênio
- etc.

**Parâmetros do Modelo**: ~15.5M
- Encoders lineares: ~1.3M
- Cross-Attention: ~6.3M
- Classifier: ~7.9M

```bash
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 3 \
    --epochs 50 \
    --batch_size 32 \
    --patience 5 \
    --seeds 42 123 456
```

**Performance esperada**:
- MCC: 0.50-0.55
- AUC: 0.83-0.85
- Accuracy: 0.75-0.77

---

### **Level 6: Otimização com Hyperparameter Search** (Opcional)
**Busca automática de hiperparâmetros com Optuna**

- **Fase 1**: HPO com Optuna (20+ trials)
- **Fase 2**: Multi-seed com melhores hiperparâmetros
- **Fase 3**: Ensemble de modelos

```bash
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 6 \
    --opt \
    --n_trials 20 \
    --opt_timeout 48
```

---

## 🚀 Pipeline Completo

Execute todos os níveis sequencialmente:

```bash
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 1 2 3 \
    --epochs 50 \
    --batch_size 32 \
    --patience 5 \
    --seeds 42 123 456 789 1024
```

Isso gerará:
- `results/benchmark_human_8M/level1_fingerprint_*/` 
- `results/benchmark_human_8M/level2_embedding_8M/`
- `results/benchmark_human_8M/level3_crossatt_8M/`
- `results/benchmark_human_8M/benchmark_comparison_report.pdf` (comparação completa)

---

## 📊 Comparação de Performance

| Level | Arquitetura | MCC | AUC | Tempo/época | Parâmetros |
|-------|-------------|-----|-----|-------------|------------|
| 1 | FP + MLP | 0.428 | 0.79 | <1min | ~100K |
| 2 | Emb + Attention Pooling + MLP | 0.45-0.48 | 0.80-0.82 | ~2min | ~500K |
| 3 | Transformer + CrossAtt | **0.50-0.55** | **0.83-0.85** | ~15min | 15.5M |
| 6 | Optimized (HPO) | **0.55-0.60+** | **0.85-0.87** | varia | 20-30M |

---

## 🔧 Embeddings Suportados

| Shorthand | Modelo | Proteína Dim | Ligante Dim |
|-----------|--------|--------------|-------------|
| `8M` | esm2_t6_8M_UR50D | 320 | 768 |
| `150M` | esm2_t30_150M_UR50D | 640 | 768 |
| `650M` | esm2_t33_650M_UR50D | 1280 | 768 |

---

## 📝 Logs e Checkpoints

Cada nível salva:
- `checkpoint_seed{seed}_best.pt` - Melhor modelo
- `training_log_seed{seed}.json` - Histórico de treinamento
- `metrics_seed{seed}.json` - Métricas finais
- `*_analysis_results.json` - Resultados agregados

---

## ⚙️ Parâmetros Importantes

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 50 | Épocas máximas de treinamento |
| `--batch_size` | 32 | Tamanho do batch |
| `--patience` | 5 | Early stopping patience |
| `--learning_rate` | 1e-4 | Learning rate inicial |
| `--seeds` | [42, 123, 456, 789, 1024] | Seeds para múltiplas execuções |
| `--force` | False | Forçar reprocessamento |

---

## 📚 Documentação Complementar

- **LEVEL-5-LITE.md**: Detalhes técnicos completos do Level 3
- **LEVEL-6.md**: Especificação do Level 6 (HPO + Ensemble)
- **CLAUDE.md**: Guia para desenvolvimento
- **MCC_OPTIMIZATION_GUIDE.md**: Estratégias para otimização

---

## 🎓 Citação

Se usar este pipeline, cite:

```bibtex
@software{semantic_screening_2026,
  title={Semantic Screening: Deep Learning Pipeline for Protein-Ligand Interaction Prediction},
  author={GMMSB-LNCC},
  year={2026},
  url={https://github.com/gmmsb-lncc/semantic-screening}
}
```
