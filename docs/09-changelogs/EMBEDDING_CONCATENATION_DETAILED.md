# 🧬 Fluxo de Embeddings no DockTKinase - Boltz vs ESM

## BOLTZ-2 (Seu Caso)

```
INPUT (TSV)
    ↓
    ├─ Coluna: SMILES ──────────────────────────────────────────┐
    │                                                            │
    └─ Coluna: Protein_Sequence ─────────┐                      │
                                         │                      │
                                         ↓                      ↓
                                    [Boltz-2]              [FM4M SMI-TED]
                                   Model Load              Model Load
                                    Inference              Inference
                                         │                      │
                                         ↓                      ↓
                                   384-dim Emb             768-dim Emb
                                   (Protein)               (Ligand)
                                         │                      │
                                         └──────────┬───────────┘
                                                    ↓
                                         np.concatenate()
                                                    ↓
                                      ┌─────────────────────┐
                                      │  1152-dim Vector    │
                                      ├─ Boltz: 0:384      │
                                      ├─ FM4M:  384:1152   │
                                      └─────────────────────┘
                                                    ↓
                         ┌──────────────────────────────────────────┐
                         │ EMBEDDING MATRIX (n_samples × 1152)     │
                         │                                          │
                         │ Sample 1: [Boltz features|FM4M features]│
                         │ Sample 2: [Boltz features|FM4M features]│
                         │ ...                                      │
                         │ Sample N: [Boltz features|FM4M features]│
                         └──────────────────────────────────────────┘
                                         ↓
                    ┌────────────────────┬─────────────────┐
                    ↓                    ↓                 ↓
                Classifier            Regression        Validation
               (Multi-model)         (Multi-model)      (Metrics)
```

---

## Comparação: BOLTZ vs ESM-2

### COM BOLTZ-2 (384-dim)
```
                Input
                  │
        ┌─────────┴──────────┐
        │                    │
    Boltz (384)          FM4M (768)
        │                    │
        └─────────┬──────────┘
                  │
          Concatenate
                  │
        ┌─────────────────┐
        │  1152-dim Total │ ← COMPACTO
        └─────────────────┘
                  │
            Classification/
             Regression
```

### COM ESM-2 t33 (1280-dim)
```
                Input
                  │
        ┌─────────┴──────────┐
        │                    │
   ESM-2 (1280)          FM4M (768)
        │                    │
        └─────────┬──────────┘
                  │
          Concatenate
                  │
        ┌─────────────────┐
        │  2048-dim Total │ ← MAIOR (1.78x)
        └─────────────────┘
                  │
            Classification/
             Regression
```

---

## ❌ O Que NÃO Acontece com Boltz

```
INCORRECT ASSUMPTION (NÃO ACONTECE):
    
    ┌─────────────────────────────────────┐
    │  ESM + Boltz + FM4M = X-dim Total   │
    └─────────────────────────────────────┘
    
    ✗ ESM não está ativo quando você escolhe Boltz
    ✗ Não há concatenação triple
    ✗ Boltz OU ESM, nunca ambos no mesmo embedding
```

---

## Mudança de Modelo = Mudança Completa

### Cenário 1: Usando ESM-2 t33
```bash
--protein-model esm2_t33_650M_UR50D
```
Pipeline gera:
- ESM embeddings: 1280-dim cada
- FM4M embeddings: 768-dim cada (reutilizável)
- **Concatenação:** [1280, 768] = **2048-dim**

### Cenário 2: Usando Boltz-2
```bash
--protein-model boltz2
```
Pipeline gera:
- Boltz embeddings: 384-dim cada
- FM4M embeddings: 768-dim cada (MESMOS arquivos!)
- **Concatenação:** [384, 768] = **1152-dim**

### Cenário 3: Usando ESM-C 600M
```bash
--protein-model esmc-600m-2024-12
```
Pipeline gera:
- ESMC embeddings: 1152-dim cada
- FM4M embeddings: 768-dim cada (MESMOS arquivos!)
- **Concatenação:** [1152, 768] = **1920-dim**

---

## 🧵 Reutilização de Embeddings

### Ligante (FM4M) - COMPARTILHÁVEL
```
├─ ligand_embeddings/
│   ├─ 10001_ligand.npy (768-dim)
│   ├─ 10002_ligand.npy (768-dim)
│   └─ ... (todos com 768-dim)
```

**Pode ser compartilhado entre:**
- Experimento com Boltz
- Experimento com ESM-2
- Experimento com ESM-C

Porque todos usam 768-dim!

### Proteína - ESPECÍFICO DO MODELO
```
├─ protein_embeddings/ (Boltz)
│   ├─ P12345_protein_embedding.npy (384-dim)
│   └─ ...
│
├─ protein_embeddings_esm/ (ESM-2)
│   ├─ P12345_protein_embedding.npy (1280-dim)
│   └─ ...
```

**NÃO podem ser compartilhados** entre modelos diferentes!

---

## 📊 Matriz Final - Estrutura Interna

```
EMBEDDING MATRIX (Boltz):
Shape: (1024, 1152) para 1024 amostras

┌────────────────┬──────────────────┐
│ Boltz (0:384)  │ FM4M (384:1152)  │
├────────────────┼──────────────────┤
│ ┌────────────┐ │ ┌──────────────┐ │
│ │ 0.234      │ │ │ -0.156       │ │
│ │ 0.512      │ │ │  0.789       │ │
│ │ ...        │ │ │ ...          │ │
│ │ (384 cols) │ │ │ (768 cols)   │ │
│ └────────────┘ │ └──────────────┘ │
├────────────────┼──────────────────┤
│ (Sample 2)     │ (Sample 2)       │
├────────────────┼──────────────────┤
│ ...            │ ...              │
└────────────────┴──────────────────┘
```

**Como acessar em código:**
```python
import numpy as np

matrix = np.load('embedding_matrix.npy')  # Shape: (1024, 1152)

# Primeiros 384 features são do Boltz
boltz_features = matrix[:, :384]         # Shape: (1024, 384)

# Últimos 768 features são do FM4M
fm4m_features = matrix[:, 384:]          # Shape: (1024, 768)

# Amostra específica (completa)
sample_1 = matrix[0]                     # Shape: (1152,)
```

---

## 🎯 Decisão de Modelo - Árvore de Decisão

```
Preciso escolher um modelo de proteína?

├─ Prioridade: VELOCIDADE
│  └─ Escolha: Boltz-2 (384-dim)
│     ├─ Matriz: 1152-dim
│     ├─ Tempo: ~0.1s/proteína
│     └─ Memória: ↓↓
│
├─ Prioridade: QUALIDADE
│  ├─ ESM-2 t33 (1280-dim)
│  │  ├─ Matriz: 2048-dim
│  │  ├─ Tempo: ~0.5s/proteína
│  │  └─ Acurácia: Muito alta
│  │
│  └─ ESM-2 t48 (5120-dim)
│     ├─ Matriz: 5888-dim
│     ├─ Tempo: ~3s/proteína (GPU)
│     └─ Acurácia: Excelente (melhor)
│
├─ Prioridade: EQUILÍBRIO
│  └─ ESM-2 t30 (640-dim)
│     ├─ Matriz: 1408-dim
│     ├─ Tempo: ~0.3s/proteína
│     └─ Acurácia: Bom
│
└─ Precisa API remota?
   └─ ESM-C 6B (3072-dim) [requer ESM_API_KEY]
      ├─ Matriz: 3840-dim
      ├─ Acurácia: Excelente
      └─ Custo: $ por uso

Resposta: A escolha anterior (Boltz) é ótima para velocidade!
```

---

## ✅ Verificação - Como Confirmar

```bash
# 1. Ver dimensão da matriz final
python -c "
import numpy as np
m = np.load('results/boltz2_pipeline_test/build/embedding_matrix.npy')
print(f'✓ Matriz: {m.shape}')
print(f'✓ Esperado: (n_samples, 1152)')
print(f'✓ Match: {m.shape[1] == 1152}')
"

# 2. Ver dimensão dos embeddings individuais
python -c "
import numpy as np
from pathlib import Path

# Pegar um arquivo de proteína (Boltz)
protein_files = list(Path('results/boltz2_pipeline_test/build/embeddings/protein_embeddings/').glob('*.npy'))
if protein_files:
    p = np.load(protein_files[0])
    print(f'✓ Proteína: {p.shape} (deve ser (384,))')

# Pegar um arquivo de ligante (FM4M)
ligand_files = list(Path('results/boltz2_pipeline_test/build/embeddings/ligand_embeddings/').glob('*.npy'))
if ligand_files:
    l = np.load(ligand_files[0])
    print(f'✓ Ligante: {l.shape} (deve ser (768,))')
"
```

---

## 🎓 TL;DR

**Sua Pergunta:** "Ao utilizar Boltz, concatena ESM+Boltz+FM4M ou apenas Boltz+FM4M?"

**Resposta:** **Apenas Boltz + FM4M (1152-dim)**

- ❌ ESM não é usado quando você escolhe Boltz
- ✅ Boltz (384-dim) + FM4M (768-dim) = 1152-dim
- 🚀 Isso é ótimo! Mais rápido e mais leve.

