# Level 3 (Cross-Attention): Technical FAQ

## 📋 Visão Geral

**Level 3** implementa uma arquitetura **Transformer + Cross-Attention Bidirecional** para predição de afinidade proteína-ligante.

### Pipeline Completo

```
Level 1: Fingerprints + KNN/MLP (baseline clássico)
Level 2: Embeddings (mean-pooled) + KNN/MLP
Level 3: Transformer + Cross-Attention ← VOCÊ ESTÁ AQUI
Level 6: Optimized Transformer (HPO com Optuna)
```

### Como executar?

```bash
# Pipeline completo (Levels 1, 2 e 3)
python semantic_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 1 2 3 \
    --epochs 50 \
    --batch_size 32
```

## 🔧 Arquitetura Detalhada

### Fluxo Completo

```
Entrada:
├─ Proteína: matriz ESM-2 [seq_len, 320/640/1280]
└─ Ligante: matriz MoLFormer [mol_len, 768]

Processamento:
├─ Protein Encoder (Linear de 2 camadas)
│  └─ [seq_len, protein_dim] → [seq_len, 512]
│
├─ Ligand Encoder (Linear de 2 camadas)
│  └─ [mol_len, 768] → [mol_len, 512]
│
├─ Cross-Attention Bidirecional (2 camadas, 8 heads)
│  ├─ Protein → Ligand: Q=prot, K=lig, V=lig → prot_updated
│  └─ Ligand → Protein: Q=lig, K=prot, V=prot → lig_updated
│
├─ Attention Pooling
│  ├─ Protein: [seq_len, 512] → [512]
│  └─ Ligand: [mol_len, 512] → [512]
│
└─ Classifier
   └─ Concat[prot, lig] → [1024] → Dropout(0.2) → [1] (logit)

Saída: Probabilidade de atividade (IC50 ≤ 1000 nM)
```

### Componentes Detalhados

#### 1. Encoders Lineares (2 camadas cada)

```python
# Protein Encoder
self.protein_encoder = nn.Sequential(
    nn.Linear(protein_dim, hidden_dim),  # e.g., [320 → 512]
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),    # [512 → 512]
    nn.LayerNorm(hidden_dim),
)

# Ligand Encoder (idêntico, mas input sempre 768)
self.ligand_encoder = nn.Sequential(
    nn.Linear(768, hidden_dim),           # [768 → 512]
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),    # [512 → 512]
    nn.LayerNorm(hidden_dim),
)
```

**Função**: Projetar embeddings de diferentes dimensões para espaço comum (512D).

#### 2. Cross-Attention Bidirecional

```python
# Criação das camadas
self.cross_attn_prot_to_lig = nn.ModuleList([
    nn.MultiheadAttention(
        embed_dim=512,
        num_heads=8,
        dropout=dropout,
        batch_first=True
    )
    for _ in range(num_cross_attn_layers)  # 2 camadas
])

self.cross_attn_lig_to_prot = nn.ModuleList([
    nn.MultiheadAttention(
        embed_dim=512,
        num_heads=8,
        dropout=dropout,
        batch_first=True
    )
    for _ in range(num_cross_attn_layers)  # 2 camadas
])

# Forward pass
for layer_p2l, layer_l2p in zip(self.cross_attn_prot_to_lig, self.cross_attn_lig_to_prot):
    # Protein → Ligand
    prot_updated, _ = layer_p2l(
        query=prot,        # O que queremos atualizar
        key=lig,          # Onde olhar
        value=lig,        # O que copiar
        key_padding_mask=~lig_mask
    )
    
    # Ligand → Protein
    lig_updated, _ = layer_l2p(
        query=lig,
        key=prot,
        value=prot,
        key_padding_mask=~prot_mask
    )
    
    prot = prot + prot_updated  # Residual connection
    lig = lig + lig_updated
```

**Função**: Permitir que cada resíduo da proteína "veja" os tokens do ligante e vice-versa.

#### 3. Attention Pooling

```python
self.protein_pooling = nn.Linear(hidden_dim, 1)
self.ligand_pooling = nn.Linear(hidden_dim, 1)

# Forward
prot_weights = F.softmax(
    self.protein_pooling(prot).squeeze(-1).masked_fill(~prot_mask, -1e9),
    dim=1
)  # [batch, seq_len]

prot_pooled = torch.sum(prot * prot_weights.unsqueeze(-1), dim=1)  # [batch, 512]
```

**Função**: Aprender quais posições são mais importantes para a predição final.

#### 4. Classifier

```python
self.classifier = nn.Sequential(
    nn.Dropout(classifier_dropout),       # 0.2
    nn.Linear(hidden_dim * 2, 1)          # [1024 → 1]
)

# Forward
combined = torch.cat([prot_pooled, lig_pooled], dim=-1)  # [batch, 1024]
logits = self.classifier(combined)  # [batch, 1]
```

## 🧠 Como funcionam as 8 cabeças de atenção?

### Conceito

Cada cabeça aprende um "tipo" diferente de interação:

```
Cabeça 1: Ligações de hidrogênio
Cabeça 2: Interações hidrofóbicas
Cabeça 3: Empilhamento π-π
Cabeça 4: Grupos polares
Cabeça 5: Flexibilidade conformacional
Cabeça 6: Tamanho do bolso
Cabeça 7: Distribuição de carga
Cabeça 8: Complementaridade de forma
```

### Implementação Matemática

```python
hidden_dim = 512
num_heads = 8
head_dim = 512 / 8 = 64  # Cada cabeça processa 64 dimensões

# Dentro do MultiheadAttention:
for h in range(num_heads):
    # Projeções lineares independentes para cada cabeça
    Q_h = W_Q_h @ query    # [batch, seq, 64]
    K_h = W_K_h @ key      # [batch, seq, 64]
    V_h = W_V_h @ value    # [batch, seq, 64]
    
    # Atenção scaled dot-product
    scores_h = (Q_h @ K_h.T) / sqrt(64)        # [batch, seq_q, seq_k]
    attn_h = softmax(scores_h)                 # Pesos de atenção
    output_h = attn_h @ V_h                    # [batch, seq_q, 64]

# Concatenar todas as cabeças
output = concat(output_1, ..., output_8)       # [batch, seq_q, 512]
output = W_O @ output                          # Projeção final
```

### Por que 8 cabeças?

1. **Múltiplas perspectivas**: Cada cabeça captura um padrão diferente
2. **Paralelização eficiente**: Processamento simultâneo em GPU
3. **Redundância robusta**: Se uma cabeça falha, outras compensam
4. **Padrão estabelecido**: Literatura usa 8-16 heads

## 📊 Parâmetros do Modelo

**Total**: ~15.5M parâmetros

### Distribuição

```
Protein Encoder:           ~820K
  ├─ Linear1 [320→512]:    163,840
  ├─ LayerNorm:            1,024
  ├─ Linear2 [512→512]:    262,144
  └─ LayerNorm:            1,024

Ligand Encoder:            ~820K
  ├─ Linear1 [768→512]:    393,216
  ├─ LayerNorm:            1,024
  ├─ Linear2 [512→512]:    262,144
  └─ LayerNorm:            1,024

Cross-Attention (2×2):     ~13M
  ├─ Q projection:         2,097,152
  ├─ K projection:         2,097,152
  ├─ V projection:         2,097,152
  ├─ Output projection:    2,097,152
  └─ (× 4 módulos)

Attention Pooling:         ~1K
  ├─ Protein:              512
  └─ Ligand:               512

Classifier:                ~525K
  └─ Linear [1024→1]:      524,288
```

## 🎯 Hiperparâmetros

### Padrão (Level 3)

```python
hidden_dim = 512              # Dimensão latente
num_cross_attn_layers = 2     # Camadas de cross-attention
num_heads = 8                 # Cabeças por camada
dropout = 0.1                 # Dropout nos encoders
classifier_dropout = 0.2      # Dropout no classifier
learning_rate = 1e-4          # Adam optimizer
batch_size = 32               # Batch size
patience = 5                  # Early stopping
```

### Como ajustar?

```bash
# Learning rate menor (mais estável)
python semantic_screening_models.py \
    --levels 3 \
    --learning_rate 5e-5

# Batch size menor (economiza memória)
python semantic_screening_models.py \
    --levels 3 \
    --batch_size 16

# Mais epochs (convergência completa)
python semantic_screening_models.py \
    --levels 3 \
    --epochs 100
```

## 📈 Resultados Esperados

### Human 8M (375K pares)

| Epoch | Val MCC | Val AUC | Val Acc |
|-------|---------|---------|---------|
| 1     | 0.418   | 0.789   | 0.713   |
| 2     | 0.423   | 0.792   | 0.717   |
| 3     | 0.499   | 0.831   | 0.754   |
| ...   | ...     | ...     | ...     |
| Best  | ~0.55   | ~0.85   | ~0.78   |

**Baseline (Level 1 FP+MLP)**: MCC=0.428

**Meta**: Superar baseline consistentemente (MCC > 0.45)

### Convergência

- **Épocas típicas**: 10-15 até early stopping
- **Tempo por época** (8M, batch=32, GPU V100): ~2-3 min
- **Tempo total** (5 seeds): ~2-3 horas

## 🚀 Comandos Úteis

### Level 3 isolado

```bash
python semantic_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 3 \
    --epochs 50 \
    --batch_size 32 \
    --seeds 42 123 456
```

### Pipeline completo

```bash
python semantic_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 1 2 3 \
    --epochs 50
```

### Forçar re-treino

```bash
python semantic_screening_models.py \
    --levels 3 \
    --force
```

### Embedding maior (650M)

```bash
python semantic_screening_models.py \
    --embedding 650M \
    --levels 3 \
    --batch_size 16  # GPU memory!
```

## 📁 Estrutura de Saída

```
results/benchmark_human_8M/level3_crossatt_8M/
├── seed_42/
│   ├── best_model.pt              # Checkpoint do melhor modelo
│   ├── training_log.json          # Métricas por época
│   └── final_metrics.json         # Resultados finais
├── seed_123/
│   └── ...
├── aggregated_metrics.json        # Média ± std sobre seeds
└── training_curves.png            # Gráficos de convergência
```

## 🔍 Troubleshooting

### "FileNotFoundError: protein_matrices/"

**Causa**: Embeddings ESM-2 não foram gerados.

**Solução**:
```bash
python scripts/run_complete_pipeline.py \
    --input tests/datasets/kinase_human_compounds.tsv \
    --output results/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D \
    --protein-model esm2_t6_8M_UR50D
```

### "CUDA out of memory"

**Causa**: Batch size muito grande.

**Solução**:
```bash
python semantic_screening_models.py \
    --levels 3 \
    --batch_size 16  # ou 8, 4
```

### MCC baixo (< 0.4)

**Causas possíveis**:
1. Learning rate alto → `--learning_rate 5e-5`
2. Poucos epochs → `--epochs 100`
3. Dados ruins → Verificar embeddings

## 📚 Referências

- Vaswani et al. "Attention Is All You Need" (2017)
- Lee et al. "Set Transformer" (2019)
- Nguyen et al. "GraphDTA" (2021)
- PyTorch: `torch.nn.MultiheadAttention`

## 🔗 Arquivos Relacionados

- `semantic_screening_models_beta.py` - Script principal
- `src/models/level5_lite.py` - Modelo Level 3
- `crossattention_split_analysis/` - Módulo de experimentos
- `LEVEL-5-LITE.md` - Documentação completa
