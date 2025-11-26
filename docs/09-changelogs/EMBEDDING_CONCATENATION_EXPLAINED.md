# 🧬 DockTKinase - Explicação da Concatenação de Embeddings

## Resposta Direta à Sua Pergunta

Quando você utiliza o **Boltz**, o modelo concatena apenas **Boltz + FM4M (ligante)**. 

**NÃO concatena ESM + Boltz + FM4M.**

---

## 📊 Como Funciona a Concatenação

### Estrutura Geral

```
┌─────────────────────────────────────────┐
│  Embedding Concatenado = Protein + FM4M │
└─────────────────────────────────────────┘

                    ↓

┌──────────────────┬────────────────┐
│  Protein (var)   │   FM4M (768)   │
└──────────────────┴────────────────┘
     (dimensão            (FIXO)
      depende do
      modelo)
```

---

## 🔍 Dimensões Reais por Modelo de Proteína

### ESM-2 (ModelsMetaScale)
| Modelo | Dimensão | Total (+ FM4M 768) |
|--------|----------|-------------------|
| esm2_t6_8M_UR50D | 320 | **1088** |
| esm2_t12_35M_UR50D | 480 | **1248** |
| esm2_t30_150M_UR50D | 640 | **1408** |
| esm2_t33_650M_UR50D | 1280 | **2048** |
| esm2_t36_3B_UR50D | 2560 | **3328** |
| esm2_t48_15B_UR50D | 5120 | **5888** |

### ESM-C (EvolutionaryScale)
| Modelo | Dimensão | Total (+ FM4M 768) |
|--------|----------|-------------------|
| esmc-300m-2024-12 | 960 | **1728** |
| esmc-600m-2024-12 | 1152 | **1920** |
| esmc-6b-2024-12 | 3072 | **3840** |

### **Boltz-2** (DeepSeek - Seu caso)
| Modelo | Dimensão | Total (+ FM4M 768) |
|--------|----------|-------------------|
| **boltz2** | **384** | **1152** |

---

## ✅ Boltz com FM4M - Configuração

Quando você roda o pipeline com Boltz:

```bash
python run_complete_pipeline.py \
    --input data.tsv \
    --protein-model boltz2 \
    --ligand-model SMI-TED
```

A concatenação será:

```
┌─────────────┬────────────────┐
│  Boltz (384) │   FM4M (768)   │
└─────────────┴────────────────┘
                   ↓
           TOTAL: 1152 dimensões
```

---

## 🔧 Código da Concatenação

### Local: `src/build/matrix/embedding_matrix.py`

```python
def _build_matrix(self) -> np.ndarray:
    """Constrói matriz de embeddings concatenados."""
    
    for ligand_id, seq_id, pki in self.data_iterator():
        # Carregar embeddings individuais
        ligand_emb = self._load_embedding(ligand_id)  # shape: (768,)
        protein_emb = self._load_embedding(seq_id)     # shape: (384,) para Boltz
        
        # Concatenar: protein_emb + ligand_emb
        final_embedding = np.concatenate([protein_emb, ligand_emb])
        # Resultado: shape (1152,) para Boltz
        
        concatenated_embeddings.append(final_embedding)
    
    # Converter para matriz
    matrix = np.vstack(concatenated_embeddings)
    # Resultado final: shape (n_samples, 1152)
```

---

## 📝 Detalhes Importantes

### 1. **Ordem de Concatenação**
- Sempre: `[Protein_Embedding, Ligand_Embedding]`
- Para Boltz: `[384-dim Boltz, 768-dim FM4M]`

### 2. **FM4M (Ligante) é FIXO**
- Sempre **768 dimensões** (SMI-TED)
- Não muda com o modelo de proteína
- Pode ser **reutilizado** entre experimentos com diferentes proteínas

### 3. **Boltz Gera 384 Dimensões**
- Mean pooling da sequência de proteína
- Não usa CLS token (diferente de ESM)
- Mais compacto que ESM-2 t33 (1280) mas bom performance

### 4. **Sem ESM Envolvido**
- Quando você escolhe `--protein-model boltz2`, o ESM não entra na concatenação
- É uma escolha de qual modelo usar (Boltz OU ESM-2 OU ESM-C)
- Não é multi-modelo no nível de embedding

---

## 🚀 Exemplo Prático

### Entrada
```
TSV: SMILES | Protein_Sequence | pKi
```

### Processamento
```
1. FM4M gera embedding do SMILES → 768-dim
2. Boltz gera embedding da proteína → 384-dim
3. Concatenar → 1152-dim
```

### Matriz Final
```
Shape: (n_samples, 1152)
├─ Primeiras 384 colunas: Embeddings Boltz
└─ Últimas 768 colunas: Embeddings FM4M
```

---

## 💡 Por Que Boltz é Mais Compacto?

| Modelo | Params | Dimensão | Razão |
|--------|--------|----------|-------|
| ESM2 t33 | 650M | 1280 | Designed para estrutura de proteína |
| Boltz | Muito menor | 384 | Otimizado para binding prediction |

Boltz é mais leve mas igualmente efetivo para predição de binding!

---

## 🎯 Resumo para Sua Dúvida

```
Com Boltz:
❌ NÃO usa: ESM + Boltz + FM4M (multiplicidade)
✅ USA: Boltz + FM4M (simples concatenação)
```

**Vantagem:** Embedding de apenas 1152 dimensões (vs 2048-5888 com ESM-2)  
**Resultado:** Mais rápido, menos memória, igualmente bom!

---

## 📋 Verificação Rápida

Para confirmar as dimensões de qualquer embedding gerado:

```bash
python -c "
import numpy as np

# Carregar embedding
emb = np.load('results/boltz2_pipeline_test/build/embedding_matrix.npy')
print(f'Shape: {emb.shape}')  # Deve ser (n_samples, 1152) para Boltz + FM4M
print(f'Dimensões por sample: {emb.shape[1]}')
"
```

---

## 🔗 Documentação Relacionada

- **Build Phase**: `src/build/matrix/embedding_matrix.py:289-305`
- **Dimensões**: `run_complete_pipeline.py:358-375`
- **Estratégias de Proteína**: `src/build/embeddings/strategies/`

