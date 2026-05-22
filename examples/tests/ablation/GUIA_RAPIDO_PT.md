# Guia Rápido: Estudo de Ablação com Dois Datasets

## 📋 Resumo das Alterações

Todos os scripts agora suportam executar experimentos em **dois datasets diferentes**:
1. **Quinases não-humanas** (`results_non_human`) - já completo
2. **Quinases humanas** (`results_human`) - pronto para executar

Os resultados ficam completamente **isolados** em diretórios separados.

## 🚀 Como Usar

### Opção 1: Script Orquestrador (Recomendado)

```bash
cd ${PROJECT_ROOT}/ablation

# Dataset não-humano (já executado)
python run_ablation_study.py --dataset non_human

# Dataset humano (novo)
python run_ablation_study.py --dataset human

# Ambos os datasets
python run_ablation_study.py --dataset both
```

### Opção 2: Execução Manual

#### Classificação - Humanos
```bash
cd classification/scripts

# 1. Extrair dados
python 01_extract_data.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human

# 2. Gerar Morgan fingerprints
python 02_generate_morgan_fingerprints.py --results-suffix results_human

# 3. Gerar encoding One-Hot
python 03_generate_aac_dpc_encoding.py --results-suffix results_human

# 4. Criar combinações C1-C4
python 04_create_combinations.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2

# 5. Executar experimentos (KNN + MLP)
python 05_run_classification.py --results-suffix results_human

# 6. Gerar visualizações
python 06_visualize_results.py --results-suffix results_human
```

#### Regressão - Humanos
```bash
cd regression/scripts

# 1. Extrair dados
python 01_extract_data_regression.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human

# 2. Executar experimentos (demora 4-6 horas)
nohup python -u 02_run_regression.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2 \
    > ../results_human/regression.log 2>&1 &

# 3. Monitorar progresso
tail -f ../results_human/regression.log

# 4. Após conclusão, consolidar checkpoints
python consolidate_checkpoints.py --results-suffix results_human

# 5. Gerar visualizações
python 03_visualize_regression_results.py --results-suffix results_human
```

## 📁 Estrutura de Diretórios

```
ablation/
├── classification/
│   ├── data/
│   │   ├── results_non_human/      # Dados não-humanos
│   │   └── results_human/          # Dados humanos
│   ├── results_non_human/          # Resultados não-humanos
│   │   ├── classification_results.json
│   │   ├── classification_summary.csv
│   │   └── figures/
│   └── results_human/              # Resultados humanos
│       ├── classification_results.json
│       ├── classification_summary.csv
│       └── figures/
│
└── regression/
    ├── data/
    │   ├── results_non_human/
    │   └── results_human/
    ├── results_non_human/
    │   └── figures/
    └── results_human/
        └── figures/
```

## 🔧 Configuração dos Datasets

### Não-Humanos (Atual)
- **TSV**: `${PROJECT_ROOT}/tests/datasets/kinase_non_human_compounds.tsv`
- **Embeddings**: `${PROJECT_ROOT}/results/protein_model_benchmark_non_human_v2/`
- **Sufixo**: `results_non_human`
- **Status**: ✅ Classificação completa, Regressão em andamento

### Humanos (Novo)
- **TSV**: `/data/docktkinase/datasets/kinase_human_compounds.tsv`
- **Embeddings**: `/data/docktkinase/results/protein_model_benchmark_human_v2/`
- **Sufixo**: `results_human`
- **Status**: ⏳ Pronto para executar

## 📊 Comparar Resultados

```python
import pandas as pd

# Carregar resultados de classificação
df_nh = pd.read_csv('classification/results_non_human/classification_summary.csv')
df_h = pd.read_csv('classification/results_human/classification_summary.csv')

# Comparar ROC-AUC
print("Não-Humanos - ROC-AUC médio:", df_nh['test_auc'].mean())
print("Humanos - ROC-AUC médio:", df_h['test_auc'].mean())

# Carregar resultados de regressão
df_reg_nh = pd.read_csv('regression/results_non_human/regression_summary.csv')
df_reg_h = pd.read_csv('regression/results_human/regression_summary.csv')

# Comparar R²
print("Não-Humanos - R² médio:", df_reg_nh['test_r2'].mean())
print("Humanos - R² médio:", df_reg_h['test_r2'].mean())
```

## ⏱️ Tempo Estimado

| Tarefa | Não-Humanos | Humanos | Observações |
|--------|-------------|---------|-------------|
| **Classificação Completa** | ~40 min | ~60 min | 100 experimentos |
| **Regressão Completa** | ~4 horas | ~6 horas | 30 experimentos, treinamento MLP |
| **Total por Dataset** | ~4.5 horas | ~7 horas | Pipeline completo |

## 🎯 Argumentos dos Scripts

Todos os scripts aceitam:
- `--tsv-path`: Caminho do arquivo TSV de entrada
- `--results-suffix`: Sufixo do diretório de resultados
- `--embeddings-dir`: Diretório dos embeddings ESM-2/SMI-TED

**Padrão**: Se não especificado, usa `results_non_human`

## ✅ Verificação

### Conferir se os resultados existem
```bash
# Não-humanos
ls classification/results_non_human/classification_summary.csv
ls regression/results_non_human/regression_summary.csv

# Humanos
ls classification/results_human/classification_summary.csv
ls regression/results_human/regression_summary.csv
```

### Conferir figuras geradas
```bash
# Não-humanos
ls classification/results_non_human/figures/
ls regression/results_non_human/figures/

# Humanos
ls classification/results_human/figures/
ls regression/results_human/figures/
```

## 🐛 Solução de Problemas

### "TSV file not found"
Verificar se o caminho existe:
```bash
ls -lh /data/docktkinase/datasets/kinase_human_compounds.tsv
```

### "Embeddings directory not found"
Verificar se os embeddings ESM-2 existem:
```bash
ls /data/docktkinase/results/protein_model_benchmark_human_v2/
```

### "No checkpoint files found"
Garantir que os experimentos de regressão foram concluídos:
```bash
ls regression/results_human/regression_summary_*.csv
```

## 📚 Documentação Completa

- **README.md**: Documentação principal em inglês
- **DUAL_DATASET_GUIDE.md**: Guia detalhado de uso
- **MIGRATION_SUMMARY.md**: Resumo técnico das mudanças
- **QUICK_REFERENCE.md**: Referência rápida em inglês

## 🔬 Perguntas de Pesquisa

Com os dois datasets, podemos investigar:

1. **Generalização**: Representações aprendidas (ESM-2) generalizam melhor para humanos?
2. **Robustez**: Features handcrafted são mais robustas entre espécies?
3. **Performance**: Qual abordagem tem menor queda de performance?
4. **Tamanho do modelo**: ESM-2 8M, 150M ou 3B é melhor para transferência cross-species?

## 💡 Próximos Passos

1. ✅ Scripts adaptados para ambos datasets
2. ⏳ Executar pipeline completo para dataset humano
3. ⏳ Comparar resultados humanos vs não-humanos
4. ⏳ Análise de transferência cross-species
5. ⏳ Visualizações comparativas

---

**Última Atualização**: 17 de Janeiro de 2026  
**Status**: Implementação completa ✅
