# Quick Start

**Last Updated**: October 28, 2025  
**Section**: Chapter 01 - Getting Started  
**Audience**: New Users

---

Quick guide to run classification and regression pipelines in 4 simple steps.

## Table of Contents

1. [4 Simple Steps](#4-simple-steps)
2. [Complete Execution Order](#complete-execution-order)
3. [Available Pipelines](#available-pipelines)
4. [Available Models](#available-models)
5. [Quick Configuration](#quick-configuration)
6. [Quick Troubleshooting](#quick-troubleshooting)

---

## 4 Simple Steps

### Step 1: Initial Setup (one-time only)

```bash
# Clonar repositório
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Criar ambiente virtual
python3 -m venv env
source env/bin/activate  # No Windows: env\Scripts\activate

# Instalar dependências automaticamente
python setup.py
```

### **2️⃣ Configurar Dados**

- Coloque seu arquivo TSV em `src/database/`
- **Formato esperado**: colunas `Ligand_SMILES`, `Target_Seq`, `Y` (label binário), `Ki`/`Kd`/`IC50` (valores nM)

### **3️⃣ Pipeline de Classificação** (Binário: Ativo/Inativo)

```bash
# Pipeline completo: embeddings + build + classificação
python run_complete_pipeline.py

# Ou apenas classificação (se embeddings já existem)
python -m src.classifier.train
```

### **4️⃣ Pipeline de Regressão** (Quantitativo: Ki, Kd, IC50)

```bash
# Pipeline completo: preparação + treinamento + avaliação
python run_regression_pipeline.py

# Ou modular
from src.regression import RegressionTrainer
trainer = RegressionTrainer(config_path='config.json')
trainer.train_all_models()
```

---

## 📋 **Ordem de Execução Completa**

```bash
# 1. Setup (primeira vez)
python setup.py

# 2. Ativar ambiente
source env/bin/activate

# 3. Pipeline COMPLETO (classificação + regressão)
python run_complete_pipeline.py      # Classificação
python run_regression_pipeline.py    # Regressão

# Resultados salvos em:
# - results/<dataset_name>/embeddings/
# - results/<dataset_name>/matrix/
# - results/<dataset_name>/labels/
# - results/<dataset_name>/models/
# - results/<dataset_name>/regression/
```

---

## 🔬 **Pipelines Disponíveis**

### **Pipeline de Classificação** (Binário)
```python
from src.build import BuildPipeline

pipeline = BuildPipeline(
    input_tsv='src/database/kinase_data.tsv',
    output_dir='results/my_experiment'
)
pipeline.run()
```

**Saída**: Modelos classificadores (RF, XGBoost, etc.) com métricas de performance

---

### **Pipeline de Regressão** (Quantitativo)
```python
from src.regression import RegressionTrainer

trainer = RegressionTrainer(
    data_path='results/my_experiment/matrix/embedding_matrix.npz',
    output_dir='results/my_experiment/regression',
    activity_type='Ki'  # ou 'Kd', 'IC50'
)
trainer.train_all_models()  # 11 modelos de regressão
```

**Saída**: Modelos de regressão com métricas (R², MAE, RMSE) e visualizações

**Prioridade de Atividades**: Ki > Kd > IC50 (ordem científica)

---

## 🎯 **Modelos Disponíveis**

### Classificação (Binário)
- Random Forest
- XGBoost
- Gradient Boosting
- SVM
- KNN
- MLP Neural Network

### Regressão (Quantitativo) - **11 Modelos**
- **Linear**: LinearRegression, Ridge, Lasso, ElasticNet
- **Tree-based**: RandomForest, GradientBoosting, XGBoost, DecisionTree
- **Outros**: SVR, KNN, MLP

---

## � **Configuração Rápida**

Crie um arquivo `config.json`:

```json
{
  "data_path": "results/my_experiment/matrix/embedding_matrix.npz",
  "output_dir": "results/my_experiment/regression",
  "activity_type": "Ki",
  "test_size": 0.2,
  "random_state": 42,
  "n_jobs": -1,
  "models_to_train": [
    "LinearRegression",
    "RandomForest",
    "XGBoost"
  ]
}
```

Execute:
```bash
python run_regression_pipeline.py --config config.json
```

---

## 📚 **Documentação Completa**

- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Guia detalhado com todas as opções
- **[USER_GUIDE.md](USER_GUIDE.md)** - Manual completo do usuário
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Instalação detalhada
- **[../src/README.md](../src/README.md)** - Arquitetura modular
- **[../src/regression/README.md](../src/regression/README.md)** - Módulo de regressão
- **[../src/utils/README.md](../src/utils/README.md)** - Utilitários
- **[../README.md](../README.md)** - README principal do projeto

---

## 🔧 **Troubleshooting Rápido**

### Erro de importação ESM
```bash
pip install fair-esm transformers
```

### CUDA out of memory
```python
# Reduzir batch_size no config
config = {'batch_size': 16}  # Padrão: 32
```

### Ambiente não ativa
```bash
# Recriar ambiente
rm -rf env
python3 -m venv env
source env/bin/activate
python setup.py
```

---

## ✅ **Checklist Rápido**

- [ ] Repositório clonado
- [ ] Ambiente virtual criado e ativado
- [ ] `python setup.py` executado sem erros
- [ ] Arquivo TSV configurado em `src/database/`
- [ ] Pipeline de classificação executado
- [ ] Pipeline de regressão executado (opcional)
- [ ] Resultados gerados em `results/`

---

**✨ Pronto! Em 4 passos você terá embeddings moleculares, modelos de classificação E regressão treinados!**

**Última atualização**: 28 de outubro de 2025 | **Branch**: regression
