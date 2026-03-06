# Gerenciamento Inteligente de Memória GPU

## 📋 Visão Geral

O pipeline implementa um **sistema adaptativo de batch processing** que gerencia automaticamente a memória GPU durante a geração de embeddings ESM-2, eliminando erros de **Out Of Memory (OOM)**.

## ✨ Features Implementadas

### 1. **Ajuste Dinâmico de Batch Size**
```python
# Inicia com batch_size definido (ex: 8)
# Se OOM: reduz pela metade (8 → 4 → 2 → 1)
# Se sucesso: aumenta gradualmente (1 → 2 → 3 → 4...)
```

**Comportamento**:
- ✅ Detecta OOM automaticamente
- ✅ Reduz batch size pela metade a cada erro
- ✅ Aumenta gradualmente em caso de sucesso
- ✅ Mínimo: 1 sequência por batch

### 2. **Limpeza Agressiva de Memória**
```python
# Após cada batch:
del batch_tokens, results, token_embeddings, batch_embeddings
torch.cuda.empty_cache()
gc.collect()
```

**Benefícios**:
- Libera memória GPU imediatamente
- Remove fragmentação
- Garante memória disponível para próximo batch

### 3. **Checkpoint Automático**
```python
# Salva progresso a cada 100 batches em:
tmp/embedding_checkpoint.npz
```

**Vantagens**:
- ✅ Retoma processamento em caso de falha
- ✅ Não perde trabalho já realizado
- ✅ Permite interrupção e retomada manual

### 4. **Monitoramento em Tempo Real**
```
      5,000/50,000 (10.0%) | batch=4 | GPU: 15.2GB usado, 17.8GB reservado
```

Mostra:
- Progresso (sequências processadas)
- Batch size atual
- Uso de memória GPU em tempo real

### 5. **PYTORCH_ALLOC_CONF Automático**

Se batch=1 ainda falhar, o sistema:
1. Define `PYTORCH_ALLOC_CONF=expandable_segments:True`
2. Recarrega o modelo
3. Tenta novamente

Isso reduz fragmentação de memória.

## 🚀 Uso

### Comando Básico
```bash
python scripts/run_complete_pipeline.py \
    --dataset all \
    --model esm2_t36_3B_UR50D \
    --device cuda
```

O batch size será ajustado **automaticamente** baseado na memória disponível.

### Monitoramento
```bash
# Terminal 1: Executar pipeline
nohup python scripts/run_complete_pipeline.py --dataset all --model esm2_t36_3B_UR50D --device cuda > logs/production.log 2>&1 &

# Terminal 2: Monitorar logs
tail -f logs/production.log

# Terminal 3: Monitorar GPU
watch -n 1 nvidia-smi
```

## 📊 Exemplo de Saída

### Processamento Normal
```
🧬 ETAPA 2: Gerando Embeddings ESM-2
------------------------------------------------------------
   Carregando modelo esm2_t36_3B_UR50D...
   ✅ Modelo carregado em 45.23s
   📊 Gerando embeddings para 150,000 sequências...
      50/150,000 (0.0%) | batch=8 | GPU: 12.3GB usado, 15.1GB reservado
      100/150,000 (0.1%) | batch=8 | GPU: 12.4GB usado, 15.2GB reservado
      ...
   ✅ Embeddings gerados!
   📊 Shape: (150000, 2560)
   ⏱️  Tempo: 3245.67s (0.022s/seq)
```

### Processamento com Ajuste Automático
```
🧬 ETAPA 2: Gerando Embeddings ESM-2
------------------------------------------------------------
   Carregando modelo esm2_t36_3B_UR50D...
   ✅ Modelo carregado em 45.23s
   📊 Gerando embeddings para 150,000 sequências...
      50/150,000 (0.0%) | batch=8 | GPU: 18.2GB usado, 21.3GB reservado
   ⚠️  OOM Error! Reduzindo batch size de 8...
      50/150,000 (0.0%) | batch=4 | GPU: 15.1GB usado, 17.8GB reservado
      100/150,000 (0.1%) | batch=4 | GPU: 15.2GB usado, 17.9GB reservado
      ...
   ✅ Embeddings gerados!
   📊 Shape: (150000, 2560)
   ⏱️  Tempo: 3567.89s (0.024s/seq)
   ⚠️  OOM Errors: 1 (batch ajustado automaticamente)
```

### Retomada de Checkpoint
```
🧬 ETAPA 2: Gerando Embeddings ESM-2
------------------------------------------------------------
   Carregando modelo esm2_t36_3B_UR50D...
   ✅ Modelo carregado em 45.23s
   📊 Gerando embeddings para 150,000 sequências...
   🔄 Checkpoint encontrado! Retomando do índice 45600
      45,650/150,000 (30.4%) | batch=4 | GPU: 15.1GB usado, 17.8GB reservado
      ...
```

## 🔧 Configuração Avançada

### Modificar Batch Size Inicial
```python
# Em run_complete_pipeline.py, linha ~1097
X = self.generate_embeddings(df, batch_size=16)  # Padrão: 8
```

### Modificar Intervalo de Checkpoint
```python
# Em run_complete_pipeline.py, linha ~257
checkpoint_interval = 50  # Padrão: 100
```

### Modificar Número de Retries
```python
# Em run_complete_pipeline.py, linha ~255
max_retries = 5  # Padrão: 3
```

## ⚠️ Troubleshooting

### Problema: Ainda recebo OOM mesmo com batch=1

**Solução 1**: Usar modelo menor
```bash
python scripts/run_complete_pipeline.py \
    --model esm2_t33_650M_UR50D \  # Menor (650M parâmetros)
    --device cuda
```

**Solução 2**: Processar em chunks
```bash
# Processar 10k sequências por vez
python scripts/run_complete_pipeline.py --max-samples 10000 --output-dir results/chunk1
python scripts/run_complete_pipeline.py --max-samples 10000 --output-dir results/chunk2
# ... combinar resultados
```

**Solução 3**: Usar CPU (muito mais lento)
```bash
python scripts/run_complete_pipeline.py --device cpu
```

### Problema: Checkpoint não está sendo removido

**Causa**: Processo interrompido antes de finalizar

**Solução**: Remover manualmente
```bash
rm -f tmp/embedding_checkpoint.npz
```

### Problema: Memória fragmentada

**Solução**: Definir variável de ambiente **antes** de executar
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_complete_pipeline.py --dataset all --model esm2_t36_3B_UR50D --device cuda
```

## 📈 Performance

### GPU com 24GB VRAM (RTX 3090/4090, A5000)

| Modelo | Batch Size Típico | Tempo/Seq | Dataset Completo (~150k) |
|--------|-------------------|-----------|-------------------------|
| `esm2_t36_3B_UR50D` | 2-4 | ~0.025s | ~60-90 min |
| `esm2_t33_650M_UR50D` | 8-16 | ~0.015s | ~40-60 min |
| `esm2_t30_150M_UR50D` | 32-64 | ~0.008s | ~20-30 min |

### GPU com 12GB VRAM (RTX 3060, GTX 1080 Ti)

| Modelo | Batch Size Típico | Tempo/Seq | Dataset Completo (~150k) |
|--------|-------------------|-----------|-------------------------|
| `esm2_t36_3B_UR50D` | 1 | ~0.030s | ~75-120 min |
| `esm2_t33_650M_UR50D` | 4-8 | ~0.018s | ~45-75 min |
| `esm2_t30_150M_UR50D` | 16-32 | ~0.010s | ~25-40 min |

### GPU com 8GB VRAM (RTX 3070, GTX 1080)

| Modelo | Batch Size Típico | Tempo/Seq | Observação |
|--------|-------------------|-----------|------------|
| `esm2_t36_3B_UR50D` | ❌ Pode falhar | - | Usar modelo menor |
| `esm2_t33_650M_UR50D` | 2-4 | ~0.020s | Recomendado |
| `esm2_t30_150M_UR50D` | 8-16 | ~0.012s | Melhor opção |

## 🎯 Recomendações

### Para Produção (Dataset Completo)

1. **GPU ≥ 24GB**: Use `esm2_t36_3B_UR50D` (melhor qualidade)
2. **GPU 12-16GB**: Use `esm2_t33_650M_UR50D` (ótimo custo-benefício)
3. **GPU 8GB**: Use `esm2_t30_150M_UR50D` (menor, mais rápido)

### Para Desenvolvimento/Testes

```bash
# Testar com 1000 amostras primeiro
python scripts/run_complete_pipeline.py \
    --max-samples 1000 \
    --model esm2_t36_3B_UR50D \
    --device cuda
```

### Para Máxima Velocidade

```bash
# Modelo menor + GPU potente
python scripts/run_complete_pipeline.py \
    --model esm2_t30_150M_UR50D \
    --device cuda
```

## 🔬 Detalhes Técnicos

### Algoritmo de Ajuste de Batch

```python
current_batch_size = initial_batch_size  # Ex: 8

for cada batch:
    try:
        processar(batch_size=current_batch_size)
        
        # Sucesso: aumentar gradualmente
        if sem_erros_oom and current_batch_size < 2 * initial:
            current_batch_size += 1
            
    except OOMError:
        # Falha: reduzir pela metade
        current_batch_size = max(1, current_batch_size // 2)
        
        if current_batch_size < 1:
            aplicar_pytorch_cuda_alloc_conf()
            recarregar_modelo()
            current_batch_size = 1
```

### Estrutura do Checkpoint

```python
np.savez_compressed(
    'tmp/embedding_checkpoint.npz',
    embeddings=np.vstack(embeddings),  # Embeddings já processados
    last_idx=batch_end                  # Último índice processado
)
```

## 📚 Referências

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [ESM-2 Model Documentation](https://github.com/facebookresearch/esm)
- [CUDA Out of Memory Best Practices](https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory)

## 🆘 Suporte

Se ainda enfrentar problemas de memória:

1. Verifique uso de GPU: `nvidia-smi`
2. Verifique logs: `tail -f logs/production.log`
3. Tente modelo menor primeiro
4. Reporte issue com detalhes da GPU e erro exato
