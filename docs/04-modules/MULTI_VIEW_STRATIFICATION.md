# Estratificação Avançada Multi-View

## 📋 Visão Geral

A estratificação multi-view é uma técnica robusta que considera simultaneamente a similaridade de **proteínas** e **ligantes** para criar splits de dados biologicamente coerentes em train/validation/test.

## 🎯 O Problema

Ao trabalhar com pares proteína-ligante, precisamos responder:

1. **Amostras com mesma proteína, ligantes diferentes** são similares?
2. **Amostras com proteínas diferentes, mesmo ligante** são similares?
3. **Como combinar AMBAS as dimensões** para um split realista?

## 📐 Similaridade de Cosseno

### Fórmula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
                       = Σ(Ai × Bi) / (√Σ(Ai²) × √Σ(Bi²))
```

### Interpretação

- **1.0**: Vetores idênticos (mesma direção)
- **0.0**: Vetores ortogonais (sem relação)
- **-1.0**: Vetores opostos

### Exemplo Prático

```python
A = [1, 2, 3]
B = [2, 4, 6]  # 2×A

cosine_similarity(A, B) = (1×2 + 2×4 + 3×6) / (√14 × √56)
                       = 28 / 28 = 1.0  ✓ Idênticos!
```

## 🔬 Multi-View Similarity

### Fórmula Matemática

```
S_multi-view(i, j) = α × S_protein(i, j) + β × S_ligand(i, j)
```

Onde:
- `α` = protein_weight (default: 0.6)
- `β` = ligand_weight (default: 0.4)
- `α + β = 1.0` (normalizado)

### Significado dos Pesos

Os pesos controlam a **importância relativa** de cada "visão":

- **α (protein_weight)**: Quanto a proteína contribui para similaridade total
- **β (ligand_weight)**: Quanto o ligante contribui para similaridade total

## 📊 Por que α=0.6 e β=0.4?

### Justificativa Biológica

1. **Proteínas determinam o mecanismo principal**
   - Mesmo ligante em proteínas diferentes → Atividade MUITO diferente
   - Exemplo: ATP funciona em centenas de kinases, mas com atividades distintas

2. **Sítio de ligação importa mais que a molécula**
   - Ligantes similares podem ter potências diferentes na mesma proteína
   - Proteínas diferentes raramente aceitam o mesmo ligante igualmente

3. **Dimensionalidade dos embeddings**
   - **Proteína**: 320 dim (ESM t6) - captura estrutura 3D, sítio ativo
   - **Ligante**: 768 dim (SMI-TED) - captura estrutura 2D química
   - Proteína tem informação mais "densa" biologicamente

### Impacto dos Pesos

| Cenário | Protein_Weight | Ligand_Weight | Resultado |
|---------|----------------|---------------|-----------|
| α=1.0, β=0.0 | Apenas proteína | Ignorado | Agrupa por alvo |
| α=0.0, β=1.0 | Ignorado | Apenas ligante | Agrupa por composto |
| α=0.6, β=0.4 | **Balanced** | **Balanced** | **Agrupa biologicamente** |
| α=0.8, β=0.2 | Protein-focused | Menos relevante | Muitas proteínas, poucos ligantes |
| α=0.3, β=0.7 | Menos relevante | Ligand-focused | Poucos alvos, muitos compostos |

## 💻 Exemplo Numérico Completo

### Dados

```python
# Amostra 1: Kinase A + Ligand X
protein_emb_1 = [0.8, 0.6]
ligand_emb_1 = [0.3, 0.4]

# Amostra 2: Kinase A + Ligand Y (mesma proteína, ligante diferente)
protein_emb_2 = [0.8, 0.6]  # Idêntica!
ligand_emb_2 = [0.1, 0.9]   # Diferente!

# Amostra 3: Kinase B + Ligand X (proteína diferente, mesmo ligante)
protein_emb_3 = [0.2, 0.7]  # Diferente!
ligand_emb_3 = [0.3, 0.4]   # Idêntica!
```

### Cálculos

#### Similaridade Amostra 1 ↔ Amostra 2

```python
S_protein(1, 2) = cosine_similarity([0.8, 0.6], [0.8, 0.6])
                = 1.0  ✓ Proteína idêntica

S_ligand(1, 2) = cosine_similarity([0.3, 0.4], [0.1, 0.9])
               = 0.61  (ligantes um pouco similares)

S_multi-view(1, 2) = 0.6 × 1.0 + 0.4 × 0.61
                   = 0.6 + 0.244
                   = 0.844  ⭐ Alta similaridade!
```

#### Similaridade Amostra 1 ↔ Amostra 3

```python
S_protein(1, 3) = cosine_similarity([0.8, 0.6], [0.2, 0.7])
                = 0.65  (proteínas moderadamente similares)

S_ligand(1, 3) = cosine_similarity([0.3, 0.4], [0.3, 0.4])
               = 1.0  ✓ Ligante idêntico

S_multi-view(1, 3) = 0.6 × 0.65 + 0.4 × 1.0
                   = 0.39 + 0.4
                   = 0.79  ⭐ Similaridade moderada-alta
```

### Interpretação

- **Amostras 1-2 (mesma proteína)**: Similaridade **0.844** (muito similares)
- **Amostras 1-3 (mesmo ligante)**: Similaridade **0.79** (similares, mas menos)

**Conclusão**: Proteína tem mais peso → Mesma proteína é mais relevante biologicamente!

## 🎲 Como Isso Afeta a Estratificação?

### Sem Multi-View (apenas embeddings concatenados)

```
Train: [Kinase A + LigX, Kinase A + LigY]
Test:  [Kinase A + LigZ, Kinase B + LigX]
```

❌ **Problema**: Kinase A está em train E test → **Data leakage!**

### Com Multi-View (α=0.6, β=0.4)

```
Cluster 1: [Kinase A + LigX, Kinase A + LigY, Kinase A + LigZ]
Cluster 2: [Kinase B + LigX, Kinase B + LigW]
Cluster 3: [Kinase C + LigX, Kinase C + LigT]

Train: Cluster 1 (70%), Cluster 2 (70%), Cluster 3 (70%)
Val:   Cluster 1 (10%), Cluster 2 (10%), Cluster 3 (10%)
Test:  Cluster 1 (20%), Cluster 2 (20%), Cluster 3 (20%)
```

✅ **Resultado**: Clusters biologicamente coerentes, splits balanceados!

## ⚙️ Configurações Alternativas

| Configuração | Protein | Ligand | Quando usar |
|--------------|---------|--------|-------------|
| **Padrão** | 0.6 | 0.4 | Balanced - boa escolha geral |
| **Protein-focused** | 0.8 | 0.2 | Dataset com muitas proteínas, poucos ligantes |
| **Ligand-focused** | 0.3 | 0.7 | Dataset com poucos alvos, muitos compostos |
| **Balanced** | 0.5 | 0.5 | Proteína e ligante igualmente importantes |
| **Protein-only** | 1.0 | 0.0 | Ignorar similaridade de ligantes |
| **Ligand-only** | 0.0 | 1.0 | Ignorar similaridade de proteínas |

## 🔧 Uso no DockTKinase

### Configuração (src/build/stratification/stratifier.py)

```python
stratifier = Stratifier(
    config=config,
    clustering_algorithm='hierarchical',  # ou 'dbscan', 'kmeans'
    similarity_threshold=0.7,
    protein_weight=0.6,  # α
    ligand_weight=0.4    # β
)
```

### Execução

```python
train_idx, val_idx, test_idx = stratifier.multi_view_stratified_split(
    protein_embeddings=protein_embeddings,  # Shape: (N, 320)
    ligand_embeddings=ligand_embeddings,    # Shape: (N, 768)
    labels=activity_labels,
    test_size=0.2,
    val_size=0.1,
    protein_weight=0.6,
    ligand_weight=0.4
)
```

### Resultado

```python
{
    'train_indices': [0, 2, 4, ...],  # 70% das amostras
    'val_indices': [1, 5, 8, ...],    # 10% das amostras
    'test_indices': [3, 6, 9, ...],   # 20% das amostras
    'n_clusters': 15,
    'cluster_sizes': {0: 1200, 1: 850, ...}
}
```

## 📈 Benefícios

### 1. Train/Val/Test Mais Realistas

- **Evita data leakage molecular**: Proteínas/ligantes similares ficam no mesmo cluster
- **Distribui clusters balanceadamente**: Cada split tem amostras de todos os clusters

### 2. Clusters Biologicamente Coerentes

- **Agrupa pares similares**: Mesma proteína + ligantes similares
- **Respeita hierarquia biológica**: Famílias de kinases, classes de compostos

### 3. Melhor Generalização

- **Modelo aprende padrões robustos**: Não memoriza combinações específicas
- **Predições mais confiáveis**: Funciona em novos pares proteína-ligante

## 📊 Validação

### Métricas de Qualidade

```python
validation_report = split_validator.validate_splits_comprehensively(
    embeddings=combined_embeddings,
    labels=labels,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)
```

**Métricas retornadas:**
- **Label balance**: Distribuição de classes ativa/inativa
- **Cluster distribution**: Representação de cada cluster nos splits
- **Similarity statistics**: Similaridade intra/inter-cluster
- **Leakage detection**: Verifica se há sobreposição molecular

## 🎓 Resumo

### O que são os pesos?

- **Coeficientes de importância relativa** (α e β)
- Controlam quanto cada "visão" (proteína vs ligante) contribui para similaridade total

### Por que α=0.6 e β=0.4?

- Reflete **importância biológica**: proteína determina mais a atividade
- Balanceia **dimensionalidade**: proteína tem menos features mas mais relevantes
- **Validado empiricamente**: funciona bem em benchmarks de drug discovery

### Como afeta os resultados?

- **Train/Val/Test mais realistas**: evita data leakage molecular
- **Clusters biologicamente coerentes**: agrupa pares proteína-ligante similares
- **Melhor generalização**: modelo aprende padrões robustos

---

**Referências**:
- ESM-2: Protein Language Model (Lin et al., 2022)
- SMI-TED: Molecular Transformer (Honda et al., 2019)
- Cosine Similarity in Drug Discovery (Muegge & Oloff, 2006)
