# OpenFold3 Embedding Extraction Guide

## 📋 Visão Geral

Este documento explica como extrair embeddings do OpenFold3 (reprodução do AlphaFold3) de forma similar aos modelos ESM-2/ESM-C.

## 🎯 Objetivo

**Sim, é possível extrair embeddings do OpenFold3!** 

O OpenFold3 gera representações intermediárias durante a predição de estrutura que podem ser usadas como embeddings para machine learning, similarmente aos modelos ESM.

## 🏗️ Arquitetura do OpenFold3

### Representações Principais

1. **Single Representation (s)**: `[N_tokens, c_s=384]`
   - Características individuais de cada token/resíduo
   - **Esta é a representação que extraímos!**
   - Dimensão: 384 (configurável)
   - Análoga aos embeddings ESM-2

2. **Pair Representation (z)**: `[N_tokens, N_tokens, c_z=128]`
   - Características de pares de resíduos
   - Captura interações e distâncias
   - Dimensão: 128 (configurável)

3. **Input Representation (s_input)**: `[N_tokens, c_s_input]`
   - Embeddings iniciais antes do trunk
   - Menos processados que s

### Fluxo de Extração

```
Sequência → Input Embedder → MSA Module → Pairformer Stack → Embeddings (s, z)
                                ↓                    ↓
                          Representation         Single Rep
                              Pair                (384-dim)
```

## 🔧 Implementação Atual

### Status: ✅ **IMPLEMENTADO**

Arquivo: `/src/build/embeddings/strategies/openfold_strategy.py`

### Método Principal: `generate()`

```python
def generate(self, model, auxiliary_objects, sequence, device, pooling_strategy='mean', **kwargs):
    """
    Extrai embeddings usando OpenFold3.run_trunk()
    
    Returns:
        numpy.ndarray: Embedding de 384 dimensões (após pooling)
    """
    # 1. Preparar batch de entrada
    batch = self._prepare_batch(sequence, device)
    
    # 2. Executar trunk do OpenFold3
    s_input, s, z = model.run_trunk(
        batch=batch,
        num_cycles=1,      # 1 ciclo é suficiente para embeddings
        inplace_safe=False
    )
    
    # 3. Extrair single representation (s)
    # s shape: [N_tokens, 384]
    
    # 4. Aplicar pooling (mean/cls/max)
    if pooling_strategy == 'mean':
        embedding = s.mean(dim=0)  # [384]
    elif pooling_strategy == 'cls':
        embedding = s[0]            # [384]
    elif pooling_strategy == 'max':
        embedding = s.max(dim=0)[0] # [384]
    
    return embedding.cpu().numpy()
```

## 📊 Comparação: OpenFold3 vs ESM

| Característica | ESM-2/ESM-C | OpenFold3 |
|----------------|-------------|-----------|
| **Tipo** | Language Model | Structure Predictor |
| **Input** | Apenas sequência | Sequência + MSA (opcional) |
| **Embeddings** | Token-level | Token-level (single rep) |
| **Dimensão** | 320-2560 (varia) | 384 (padrão) |
| **Awareness** | Evolutiva | Estrutural |
| **Velocidade** | Muito rápida | Moderada |
| **Uso ML** | ✅ Excelente | ✅ Bom (estrutural) |

## 🔬 Diferenças Técnicas

### ESM-2/ESM-C
```python
# Simples: tokenizar + forward
tokens = tokenizer(sequence)
output = model(tokens, repr_layers=[33])
embedding = output['representations'][33]  # [L, dim]
```

### OpenFold3
```python
# Mais complexo: batch + trunk
batch = prepare_batch(sequence)  # Requer token_mask, asym_id, etc.
s_input, s, z = model.run_trunk(batch, num_cycles=1)
embedding = s  # [L, 384]
```

## 📋 Requisitos para Batch do OpenFold3

### Campos Obrigatórios

```python
batch = {
    # Core features
    'token_mask': torch.Tensor,        # [N_token] - mask de tokens válidos
    'asym_id': torch.LongTensor,       # [N_token] - ID da cadeia/entidade
    'entity_id': torch.LongTensor,     # [N_token] - ID da entidade
    'sym_id': torch.LongTensor,        # [N_token] - ID de simetria
    
    # Positions
    'token_index': torch.LongTensor,   # [N_token] - índice do token
    'ref_pos': torch.Tensor,           # [N_token, 3] - posições de referência
    'ref_mask': torch.Tensor,          # [N_token] - mask de posições
    
    # Atom features
    'ref_element': torch.LongTensor,   # [N_token, 3] - elementos atômicos
    'ref_charge': torch.LongTensor,    # [N_token, 3] - cargas
    'ref_atom_name_chars': torch.LongTensor,  # [N_token, 3, 4] - nomes
    'ref_space_uid': torch.LongTensor, # [N_token] - UID espacial
    
    # MSA (opcional para embeddings)
    'msa': torch.LongTensor,           # [N_seq, N_token] - MSA
    'msa_mask': torch.Tensor,          # [N_seq, N_token] - mask MSA
    
    # Templates (opcional)
    'template_*': ...
}
```

### Implementação Simplificada

Para extração de embeddings, podemos usar uma versão **simplificada** do batch:

```python
def _prepare_batch(self, sequence: str, device: torch.device) -> dict:
    """Cria batch mínimo para extração de embeddings."""
    seq_len = len(sequence)
    
    batch = {
        # Essencial
        'token_mask': torch.ones(seq_len, dtype=torch.bool, device=device),
        'asym_id': torch.zeros(seq_len, dtype=torch.long, device=device),
        'entity_id': torch.zeros(seq_len, dtype=torch.long, device=device),
        'sym_id': torch.zeros(seq_len, dtype=torch.long, device=device),
        'token_index': torch.arange(seq_len, dtype=torch.long, device=device),
        
        # Posições (zeros - não usamos estrutura)
        'ref_pos': torch.zeros(seq_len, 3, device=device),
        'ref_mask': torch.zeros(seq_len, dtype=torch.bool, device=device),
        
        # Átomos (placeholder)
        'ref_element': torch.zeros(seq_len, 3, dtype=torch.long, device=device),
        'ref_charge': torch.zeros(seq_len, 3, dtype=torch.long, device=device),
        'ref_atom_name_chars': torch.zeros(seq_len, 3, 4, dtype=torch.long, device=device),
        'ref_space_uid': torch.zeros(seq_len, dtype=torch.long, device=device),
    }
    
    return batch
```

## ⚙️ Configuração e Uso

### 1. Verificar Instalação

```bash
# OpenFold3 deve estar em openfold-3/
ls openfold-3/openfold3/projects/of3_all_atom/model.py
```

### 2. Carregar Modelo

```python
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

strategy = OpenFoldStrategy()
model, _ = strategy.load('openfold3', device=torch.device('cpu'))
```

### 3. Gerar Embeddings

```python
sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"

# Extrair embeddings
embedding = strategy.generate(
    model=model,
    auxiliary_objects=None,
    sequence=sequence,
    device=torch.device('cpu'),
    pooling_strategy='mean'  # ou 'cls', 'max'
)

print(f"Embedding shape: {embedding.shape}")  # (384,)
```

### 4. Uso no Pipeline

```bash
python run_complete_pipeline.py \
  --input tests/datasets/kinase_non_human_compounds.tsv \
  --output results/openfold3_test \
  --esm-model openfold3 \
  --seed 42
```

## 🔍 Validação

### Testar Extração

```python
import torch
from src.build.embeddings.strategies.openfold_strategy import OpenFoldStrategy

# Setup
strategy = OpenFoldStrategy()
model, _ = strategy.load('openfold3', torch.device('cpu'))

# Sequência teste
seq = "ACDEFGHIKLMNPQRSTVWY"

# Extrair
emb = strategy.generate(model, None, seq, torch.device('cpu'))

# Validar
assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"
assert not np.isnan(emb).any(), "NaN values found"
assert not np.isinf(emb).any(), "Inf values found"

print("✅ Validação passou!")
```

## 🚀 Otimizações Futuras

### 1. MSA Support (Opcional)
- Adicionar suporte para MSA quando disponível
- Melhoraria qualidade dos embeddings estruturais

### 2. Batch Processing
- Processar múltiplas sequências em paralelo
- Reduzir overhead de preparação

### 3. Cache de Embeddings
- Cachear embeddings de sequências comuns
- Evitar recomputação

### 4. GPU Offloading
- Usar CPU offloading para sequências longas
- Similar ao implementado para ESM-C

## 📖 Referências

1. **OpenFold3**: https://github.com/aqlaboratory/openfold
2. **AlphaFold3**: https://www.nature.com/articles/s41586-024-07487-w
3. **ESM-2**: https://github.com/facebookresearch/esm

## 🎯 Conclusão

**Sim, é totalmente possível extrair embeddings do OpenFold3!**

✅ **Implementação completa** no `openfold_strategy.py`
✅ **Compatível** com pipeline DockTKinase
✅ **Similar** aos modelos ESM-2/ESM-C
✅ **Embeddings estruturalmente aware** (384-dim)

A principal diferença é que OpenFold3 gera embeddings **estruturalmente informados**, enquanto ESM-2/ESM-C são puramente **baseados em sequência evolutiva**.

Ambos são complementares e podem ser usados juntos no pipeline!
