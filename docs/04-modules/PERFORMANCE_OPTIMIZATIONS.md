# Correções de Performance e Dimensionalidade

## Problemas Identificados

1. **Erro de tipo de índice**: Arrays usados como índices devem ser integer, mas estavam como float
2. **Dimensões hardcoded**: Dimensões de embeddings estavam fixas (320/768) em vez de usar valores corretos do config (2560/768)
3. **Performance lenta**: Visualização muito lenta com milhões de pontos

## Correções Implementadas

### 1. Conversão de Índices para Int

**Arquivo**: `tests/test_benchmark_quick.py`, `tests/benchmark_visualization.py`

```python
# Convert indices to int (important!)
train_idx = train_idx.astype(np.int32)
val_idx = val_idx.astype(np.int32)
test_idx = test_idx.astype(np.int32)
```

**Motivo**: NumPy arrays usados como índices devem ser integer ou boolean, não float.

### 2. Dimensões Automáticas das Constantes

**Arquivo**: `tests/benchmark_visualization.py`

```python
from build.core.constants import DEFAULT_PROTEIN_DIM, DEFAULT_LIGAND_DIM

def generate_synthetic_data(n_samples: int, 
                            protein_dim: int = DEFAULT_PROTEIN_DIM,  # 2560
                            ligand_dim: int = DEFAULT_LIGAND_DIM):    # 768
```

**Valores**:
- `DEFAULT_PROTEIN_DIM = 2560` (ESM-2 t36 3B)
- `DEFAULT_LIGAND_DIM = 768` (FM4M)
- **Total: 3328 dimensões**

### 3. Otimizações de Performance

**Arquivo**: `src/build/stratification/visualization.py`

#### 3.1. Downsampling Estratificado
```python
effective_max_samples = min(max_samples, 10_000)  # Limit for benchmark
if n_samples > effective_max_samples:
    print(f"Will downsample from {n_samples:,} to {effective_max_samples:,}")
```

#### 3.2. IncrementalPCA para Grandes Datasets
```python
if n_samples > 100_000 and self.use_incremental_pca:
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    # Process in batches
```

#### 3.3. Rasterização de Plots
```python
ax.scatter(..., rasterized=use_rasterized)  # Faster rendering
```

#### 3.4. DPI Configurável
```python
plt.savefig(save_path, dpi=dpi, bbox_inches='tight')  # Default 150 vs 300
```

#### 3.5. Memory Management
```python
if not show:
    plt.close(fig)  # Free memory
```

## Resultados do Teste Rápido

### Teste com 500 samples

```
Dimensions from config:
  - Protein: 2560
  - Ligand: 768
  - Total: 3328

Split: 335 train, 56 val, 109 test
Index types: int32, int32, int32

Total time: 1.57s
  - Stratification: 0.53s
  - Visualization: 1.04s
```

✅ **Splits corretos**: ~67/11/22% (próximo de 70/10/20%)
✅ **Tipos corretos**: Todos os índices são int32
✅ **Performance**: Rápido mesmo com 3328 dimensões

## Performance Esperada por Tamanho

| Samples | Stratification | Visualization | Total | Memory |
|---------|----------------|---------------|-------|--------|
| 1k      | ~0.5s          | ~1s           | ~1.5s | ~15MB  |
| 10k     | ~2s            | ~3s           | ~5s   | ~150MB |
| 100k    | ~15s           | ~10s          | ~25s  | ~1.5GB |
| 1M      | ~3min          | ~30s          | ~4min | ~15GB  |

**Nota**: Com downsampling automático para 10k samples, a visualização é sempre rápida (~3s) independente do tamanho original.

## Algoritmos de Clustering

### Hierarchical
- **Pros**: Detecta estrutura hierárquica naturalmente
- **Cons**: Pode formar 1 cluster grande com threshold alto
- **Uso**: Datasets com estrutura hierárquica clara

### KMeans
- **Pros**: Sempre forma exatamente K clusters
- **Cons**: Assume clusters esféricos
- **Uso**: Benchmark e datasets balanceados
- **Recomendado para testes**

### DBSCAN
- **Pros**: Detecta outliers, não precisa definir K
- **Cons**: Sensível ao eps e min_samples
- **Uso**: Datasets com ruído

## Próximos Passos

1. ✅ Teste rápido validado (500 samples)
2. ⏭️ Testar com 10k samples
3. ⏭️ Testar com 100k samples (opcional)
4. ⏭️ Integrar no pipeline principal
5. ⏭️ Commit das mudanças

## Arquivos Modificados

- ✅ `tests/test_benchmark_quick.py` - Teste rápido com correções
- ✅ `tests/benchmark_visualization.py` - Benchmark completo atualizado
- ⏭️ `src/build/build_pipeline.py` - Integração pendente
