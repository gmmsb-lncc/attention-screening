## 📊 ETAPA 1: ANÁLISE DE I/O E DATA LOADING

**Data**: 26 de Novembro de 2025  
**Escopo**: Diagnóstico de gargalos em I/O, carregamento de dados e processamento em batch

---

## 🔍 ACHADOS PRINCIPAIS

### ✅ BOM: Configuração de DataLoader (Classifier)
**Localização**: `src/classifier/classifier.py` linha 360

```python
common_kwargs = dict(
    batch_size=self.batch_size,
    pin_memory=pin,           # ✅ Habilitado para GPU
    num_workers=self.num_workers,
    persistent_workers=False,  # ✅ Correto (evita fd leak)
)
```

**Positivos**:
- ✅ `pin_memory` habilitado quando GPU disponível
- ✅ `persistent_workers=False` para evitar vazamento de file descriptors
- ✅ Usa `Subset` para splits eficientes

---

### ⚠️ ACHADO #1: num_workers=0 no Classifier
**Localização**: `src/classifier/config/mlp_config.py` linha 33

```python
num_workers: int = 0  # ⚠️ GARGALO - desabilita multi-threading
```

**Impacto**: 
- ❌ Sem paralelização de data loading
- ❌ CPU+GPU ociosos enquanto dados são carregados
- ❌ Reduz throughput especialmente em datasets grandes

**Recomendação**: `num_workers = 4 ou min(4, cpu_count()-1)`

---

### ⚠️ ACHADO #2: Falta de prefetch_factor
**Localização**: `src/classifier/classifier.py` linha 360

```python
common_kwargs = dict(
    batch_size=self.batch_size,
    pin_memory=pin,
    num_workers=self.num_workers,
    # ❌ FALTANDO: prefetch_factor
    persistent_workers=False,
)
```

**Impacto**:
- ❌ Sem buffering de próximos batches
- ❌ GPU esperando dados após processar batch atual

**Recomendação**: Adicionar `prefetch_factor=2` se `num_workers > 0`

---

### ✅ BOM: Regression usa num_workers=4
**Localização**: `src/regression/config.py` linha 92

```python
num_workers: int = 4  # ✅ Bom
```

---

### ⚠️ ACHADO #3: NPY files com allow_pickle=True
**Localização**: `src/build/labels/binary_labels.py` linha 94

```python
interaction_data = np.load(self.interaction_labels_path, allow_pickle=True)
```

**Impacto**:
- ⚠️ `allow_pickle=True` é mais lento que carregamento direto
- ⚠️ Segurança: possibilidade de exploração
- ⚠️ Usado quando desnecessário

**Recomendação**: Verificar se realmente precisa de pickle, senão remover

---

### ✅ BOAS PRÁTICAS ENCONTRADAS

1. **Uso de dtype eficiente**: `torch.FloatTensor`, `torch.LongTensor`
2. **Lazy loading**: Carregamento sob demanda
3. **Context managers**: Cleanup automático em base_builder.py
4. **Validação de splits**: Sem sobreposição entre treino/val/test

---

## 📈 ESTIMATIVA DE GANHO DE PERFORMANCE

| Otimização | Impacto Estimado | Esforço |
|------------|------------------|--------|
| Aumentar `num_workers` de 0→4 | **+200% throughput** | Baixo |
| Adicionar `prefetch_factor=2` | **+30-50% speedup** | Baixo |
| Remover `allow_pickle=True` | **+10-15% I/O** | Baixo |
| Usar memory mapping para NPY grandes | **+40-60%** (arquivos >1GB) | Médio |

---

## 🎯 RECOMENDAÇÕES ETAPA 1

### 🔴 CRÍTICO (ganho >50%)
1. Aumentar `num_workers` no classifier: 0 → 4

### 🟡 IMPORTANTE (ganho 20-50%)
2. Adicionar `prefetch_factor=2`
3. Remover `allow_pickle=True` desnecessário

### 🟢 OPCIONAL (ganho <20%)
4. Implementar memory mapping para arquivos grandes

---

## ✅ PRÓXIMO PASSO
Etapa 2: Análise GPU/CUDA (verificar mixed precision, memory management)

