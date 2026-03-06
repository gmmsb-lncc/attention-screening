# 🚀 Pipeline Completo DockTKinase

Pipeline end-to-end para classificação de compostos kinase usando embeddings ESM-2.

## 📋 Funcionalidades

O pipeline executa automaticamente:

1. **📂 Carregamento de Dados** - Lê datasets TSV de kinases
2. **🧬 Geração de Embeddings** - Usa ESM-2 para gerar embeddings de sequências
3. **🔀 Estratificação Robusta** - Divisão train/test com validação estatística
4. **🤖 Treinamento** - Random Forest Classifier
5. **📊 Avaliação** - Métricas completas (Acc, F1, ROC-AUC, etc.)
6. **💾 Salvamento** - Resultados e estatísticas em JSON

## 🎯 Uso Básico

### Teste Rápido (100 amostras)

```bash
python scripts/run_complete_pipeline.py \
    --dataset human \
    --model esm2_t6_8M_UR50D \
    --max-samples 100 \
    --output-dir tests/quick_test
```

### Dataset Completo Humano

```bash
python scripts/run_complete_pipeline.py \
    --dataset human \
    --model esm2_t33_650M_UR50D \
    --output-dir results_human
```

### Dataset Não-Humano

```bash
python scripts/run_complete_pipeline.py \
    --dataset non_human \
    --model esm2_t6_8M_UR50D \
    --output-dir results_non_human
```

### Dataset Completo (All)

```bash
python scripts/run_complete_pipeline.py \
    --dataset all \
    --model esm2_t36_3B_UR50D \
    --device cuda \
    --output-dir results_all
```

## 📊 Datasets Disponíveis

| Dataset | Arquivo | Tamanho | Descrição |
|---------|---------|---------|-----------|
| `human` | `kinase_human_compounds.tsv` | 404 MB | Kinases humanas |
| `non_human` | `kinase_non_human_compounds.tsv` | 11 MB | Kinases não-humanas |
| `all` | `kinase_all_compounds.tsv` | 415 MB | Todas as kinases |

## 🧬 Modelos ESM-2 Disponíveis

| Modelo | Params | Dimensão | Velocidade | Uso Recomendado |
|--------|--------|----------|------------|-----------------|
| `esm2_t6_8M_UR50D` | 8M | 320 | ⚡⚡⚡ Rápido | Testes, desenvolvimento |
| `esm2_t12_35M_UR50D` | 35M | 480 | ⚡⚡ Médio | Validação |
| `esm2_t30_150M_UR50D` | 150M | 640 | ⚡ Médio-Lento | Produção (GPU) |
| `esm2_t33_650M_UR50D` | 650M | 1280 | 🐢 Lento | Produção (GPU) |
| `esm2_t36_3B_UR50D` ⭐ | 3B | 2560 | 🐢🐢 Muito Lento | Produção (GPU forte) |
| `esm2_t48_15B_UR50D` | 15B | 5120 | 🐢🐢🐢 Extremo | Pesquisa (GPU multi) |

## ⚙️ Argumentos Completos

```bash
python scripts/run_complete_pipeline.py [OPÇÕES]

Argumentos:
  --dataset {human,non_human,all}
                        Dataset a usar (default: human)
  
  --model MODEL         Modelo ESM-2 a usar (default: esm2_t6_8M_UR50D)
  
  --test-size FLOAT     Proporção do conjunto de teste (default: 0.2)
  
  --max-samples INT     Limite de amostras para teste rápido
                        (default: todas)
  
  --device {cpu,cuda,auto}
                        Device a usar (default: auto)
  
  --output-dir DIR      Diretório para salvar resultados
                        (default: pipeline_output)
  
  --seed INT            Random seed para reprodutibilidade
                        (default: 42)
  
  --quiet               Modo silencioso (sem logs detalhados)
```

## 📈 Saídas do Pipeline

O pipeline cria um diretório de saída com:

### `pipeline_stats.json`

Contém todas as estatísticas do pipeline:

```json
{
  "start_time": "2025-10-22T...",
  "end_time": "2025-10-22T...",
  "dataset": "human",
  "model": "esm2_t6_8M_UR50D",
  "device": "cpu",
  "total_samples": 100,
  "load_time": 2.94,
  "embedding_time": 214.12,
  "embedding_shape": [100, 320],
  "split_time": 0.12,
  "train_size": 80,
  "test_size": 20,
  "max_proportion_diff": 0.0,
  "chi2_p_value_train": 1.0,
  "chi2_p_value_test": 1.0,
  "train_time": 0.16,
  "eval_time": 0.04,
  "metrics": {
    "accuracy": 0.5000,
    "precision": 0.4890,
    "recall": 0.5000,
    "f1": 0.4896,
    "roc_auc": 0.4343,
    "avg_precision": 0.4537
  }
}
```

## 🎯 Exemplos de Uso

### Exemplo 1: Desenvolvimento Rápido

```bash
# Teste com 500 amostras, modelo pequeno
python scripts/run_complete_pipeline.py \
    --dataset human \
    --model esm2_t6_8M_UR50D \
    --max-samples 500 \
    --output-dir tests/dev_test
```

**Tempo estimado**: ~20 minutos (CPU)

### Exemplo 2: Validação com Modelo Médio

```bash
# Dataset não-humano completo, modelo médio
python scripts/run_complete_pipeline.py \
    --dataset non_human \
    --model esm2_t33_650M_UR50D \
    --device cuda \
    --output-dir tests/validation_results
```

**Tempo estimado**: ~2 horas (GPU)

### Exemplo 3: Produção com Melhor Modelo

```bash
# Dataset humano completo, melhor modelo
python scripts/run_complete_pipeline.py \
    --dataset human \
    --model esm2_t36_3B_UR50D \
    --device cuda \
    --test-size 0.15 \
    --output-dir tests/production_human
```

**Tempo estimado**: ~8 horas (GPU RTX 4090)

### Exemplo 4: Dataset Completo

```bash
# Todos os dados, modelo grande
python scripts/run_complete_pipeline.py \
    --dataset all \
    --model esm2_t36_3B_UR50D \
    --device cuda \
    --output-dir tests/production_all
```

**Tempo estimado**: ~10-12 horas (GPU RTX 4090)

## 📊 Interpretando Resultados

### Métricas de Classificação

- **Accuracy**: Proporção de predições corretas
- **Precision**: Proporção de positivos preditos que são verdadeiros
- **Recall**: Proporção de positivos verdadeiros encontrados
- **F1-Score**: Média harmônica de precision e recall
- **ROC AUC**: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)
- **Avg Precision**: Média ponderada de precision em diferentes thresholds

### Validação Estatística do Split

O pipeline valida automaticamente que o split está balanceado:

- ✅ **Chi-quadrado p-value > 0.05**: Distribuições similares
- ✅ **Diferença de proporções < 5%**: Balanceamento adequado

## ⚡ Otimizações de Performance

### CPU

```bash
# Usar modelo pequeno
python scripts/run_complete_pipeline.py --model esm2_t6_8M_UR50D
```

### GPU

```bash
# Usar modelo grande com CUDA
python scripts/run_complete_pipeline.py \
    --model esm2_t36_3B_UR50D \
    --device cuda
```

### Teste Rápido

```bash
# Limitar amostras
python scripts/run_complete_pipeline.py --max-samples 1000
```

## 🔧 Requisitos de Sistema

### Mínimo (Teste - t6 8M)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **GPU**: Não necessária
- **Tempo**: ~0.1s/sequência

### Recomendado (Produção - t33 650M)
- **CPU**: 8 cores
- **RAM**: 16 GB
- **GPU**: 8 GB VRAM (RTX 3070, 4060)
- **Tempo**: ~0.5s/sequência

### Ideal (Produção - t36 3B)
- **CPU**: 16 cores
- **RAM**: 32 GB
- **GPU**: 12+ GB VRAM (RTX 4090, 3090)
- **Tempo**: ~1s/sequência

## 🐛 Troubleshooting

### Out of Memory (GPU)

```bash
# Usar modelo menor
python scripts/run_complete_pipeline.py --model esm2_t6_8M_UR50D

# Ou usar CPU
python scripts/run_complete_pipeline.py --device cpu
```

### Dataset muito grande

```bash
# Limitar amostras
python scripts/run_complete_pipeline.py --max-samples 10000
```

### Pipeline lento

```bash
# Usar modelo menor e GPU
python scripts/run_complete_pipeline.py \
    --model esm2_t6_8M_UR50D \
    --device cuda
```

## 📝 Labels dos Dados

O pipeline cria labels automaticamente baseado em `pchembl_value`:

- **Ativo (label=1)**: pchembl_value >= 6.0
- **Inativo (label=0)**: pchembl_value < 6.0

Se o dataset já tiver uma coluna `label`, ela será usada diretamente.

## 🎓 Referências

- **ESM-2**: Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure"
- **Random Forest**: Breiman (2001) - "Random Forests"
- **Stratified Split**: Validação com teste chi-quadrado

## 📞 Suporte

Para problemas ou dúvidas:
1. Verifique a documentação em `docs/`
2. Execute testes com `--max-samples 100`
3. Verifique logs detalhados (sem `--quiet`)

---

**Status**: ✅ Pronto para produção  
**Última atualização**: 22 de outubro de 2025
