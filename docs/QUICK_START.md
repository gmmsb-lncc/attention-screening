# ⚡ Quick Start - DockTKinase

**Guia rápido para executar os pipelines de classificação e regressão.**

---

## 🚀 **4 Passos Simples**

### **1️⃣ Setup Inicial (apenas uma vez)**

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

**Duas opções disponíveis:**

**A. Pipeline Tradicional** (reutiliza embeddings do classificador):
```bash
python run_regression_pipeline.py
```

**B. Pipeline Modular** ⭐ **NOVO** (standalone, interface simplificada):
```bash
# Básico
python src/regression/modular_regression.py embeddings.npy targets.npy

# Com opções
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --models RandomForest XGBoost KNN \
    --output results/my_experiment
```

**OU via Python:**
```python
# Pipeline tradicional
from src.regression import RegressionTrainer
trainer = RegressionTrainer(config_path='config.json')
trainer.train_all_models()

# Pipeline modular ⭐ NOVO
from regression.modular_pipeline import RegressionPipeline
pipeline = RegressionPipeline('embeddings.npy', 'targets.npy')
results = pipeline.run()
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

**Duas interfaces disponíveis:**

**A. Pipeline Tradicional**:
```python
from src.regression import RegressionTrainer

trainer = RegressionTrainer(
    data_path='results/my_experiment/matrix/embedding_matrix.npz',
    output_dir='results/my_experiment/regression',
    activity_type='Ki'  # ou 'Kd', 'IC50'
)
trainer.train_all_models()  # 11 modelos de regressão
```

**B. Pipeline Modular** ⭐ **NOVO**:
```python
from regression.modular_pipeline import RegressionPipeline

pipeline = RegressionPipeline(
    embeddings_path='embeddings.npy',
    targets_path='targets.npy',
    output_dir='results/regression',
    models_to_train=['RandomForest', 'XGBoost', 'KNN']
)
results = pipeline.run()
```

**Saída**: Modelos de regressão com 15+ métricas (R², MAE, RMSE, percentiles) e visualizações

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
- **Linear**: Ridge, Lasso, ElasticNet (regularização)
- **Tree-based**: RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost
- **Other**: SVR, KNN, MLP

**⭐ NOVO**: Interface modular standalone disponível!
```bash
python src/regression/modular_regression.py embeddings.npy targets.npy \
    --models RandomForest XGBoost KNN
```

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
- **[../src/regression/README_MODULAR.md](../src/regression/README_MODULAR.md)** ⭐ **NOVO** - Arquitetura modular de regressão
- **[../src/regression/README_IMPROVEMENTS.md](../src/regression/README_IMPROVEMENTS.md)** - Melhorias do módulo de regressão
- **[../docs/REGRESSION_MODULAR_REPORT.md](REGRESSION_MODULAR_REPORT.md)** ⭐ **NOVO** - Relatório completo de modularização
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

**Última atualização**: 7 de novembro de 2025 | **Branch**: regression
