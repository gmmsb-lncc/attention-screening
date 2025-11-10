# 🚀 Quick Start - Integrated Pipeline

## ⚡ TL;DR - Execute em 3 Comandos

```bash
# 1. Clone e setup
git clone <repo> && cd docktkinase
pip install -r requirements.txt

# 2. Execute o workflow completo
python -m src.integrated_pipeline --input data/kinase_data.tsv --output results/

# 3. Veja os resultados
cat results/integrated_results.json
```

**Pronto!** Em ~5-10 minutos você terá:
- ✅ Embeddings gerados (ligand + protein)
- ✅ Classificação binária (ROC-AUC ~0.85)
- ✅ Regressão quantitativa (MAE, R², RMSE)

---

## 📊 Modos de Execução

### 1. Workflow Completo (Padrão)

```bash
python -m src.integrated_pipeline \
    --input data/kinase_data.tsv \
    --output results/complete
```

**Output**:
- Build: `results/complete/build/`
- Classification: `results/complete/classifier/`
- Regression: `results/complete/regression/`
- Summary: `results/complete/integrated_results.json`

---

### 2. Apenas Embeddings (Build)

```bash
python -m src.integrated_pipeline \
    --input data/kinase_data.tsv \
    --output results/build_only \
    --no-classification \
    --no-regression
```

**Output**:
- Protein embeddings: `results/build_only/build/embeddings/protein_embeddings.npy`
- Ligand embeddings: `results/build_only/build/embeddings/ligand_embeddings.npy`
- Matrix: `results/build_only/build/matrix/embedding_matrix.npy`

---

### 3. Build + Classification

```bash
python -m src.integrated_pipeline \
    --input data/kinase_data.tsv \
    --output results/classification \
    --no-regression
```

**Output**:
- Build outputs
- Trained MLP model: `results/classification/classifier/mlp_model.pth`
- Metrics: `results/classification/classifier/metrics.json`

---

### 4. Build + Regression

```bash
python -m src.integrated_pipeline \
    --input data/kinase_data.tsv \
    --output results/regression \
    --no-classification
```

**Output**:
- Build outputs
- Trained models: `results/regression/regression/models/`
- Metrics: `results/regression/regression/metrics.json`

---

## 🎯 Customização

### Escolher Modelos de Regressão

```bash
python -m src.integrated_pipeline \
    --input data.tsv \
    --output results/ \
    --regression-models Ridge Lasso XGBoost
```

**Modelos disponíveis**:
- `Ridge`, `Lasso`, `ElasticNet`
- `RandomForest`, `GradientBoosting`
- `XGBoost`, `LightGBM`, `CatBoost`
- `SVR`, `KNN`, `MLP`

---

### Escolher Modelo ESM

```bash
python -m src.integrated_pipeline \
    --input data.tsv \
    --output results/ \
    --esm-model esm2_t33_650M_UR50D
```

**Modelos disponíveis**:
- `esm2_t6_8M_UR50D` (pequeno, rápido)
- `esm2_t12_35M_UR50D` (médio)
- `esm2_t30_150M_UR50D` (grande)
- `esm2_t33_650M_UR50D` (muito grande, melhor qualidade)

---

### Usar GPU

```bash
python -m src.integrated_pipeline \
    --input data.tsv \
    --output results/ \
    --device cuda  # ou 'mps' para Apple Silicon
```

---

### Custom Random Seed

```bash
python -m src.integrated_pipeline \
    --input data.tsv \
    --output results/ \
    --random-state 123
```

---

## 🐍 Python API

### Exemplo Básico

```python
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

# Configure
config = IntegratedConfig(
    input_tsv="data/kinase_data.tsv",
    output_dir="results/",
    run_classification=True,
    run_regression=True
)

# Run
pipeline = IntegratedPipeline(config)
results = pipeline.run()

# Check status
print(f"Status: {results['status']}")
print(f"Total time: {results['total_time_seconds']:.2f}s")
```

---

### Exemplo Avançado

```python
from src.integrated_pipeline import IntegratedPipeline, IntegratedConfig

config = IntegratedConfig(
    # Input/Output
    input_tsv="data/kinase_data.tsv",
    output_dir="results/custom",
    
    # Build configuration
    esm_model="esm2_t12_35M_UR50D",
    batch_size=16,
    device="cuda",
    
    # Data split
    test_size=0.25,
    val_size=0.15,
    random_state=42,
    
    # Classification
    run_classification=True,
    classifier_epochs=100,
    classifier_cv_folds=10,
    
    # Regression
    run_regression=True,
    regression_models=['Ridge', 'XGBoost', 'LightGBM'],
    regression_cv_folds=5,
    
    verbose=True
)

pipeline = IntegratedPipeline(config)
results = pipeline.run()

# Access results
if results['status'] == 'completed':
    # Classification metrics
    clf = results['classifier']
    print(f"\n🧠 Classification:")
    print(f"  ROC-AUC: {clf['test_metrics']['roc_auc']:.4f}")
    print(f"  Accuracy: {clf['test_metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {clf['test_metrics']['f1']:.4f}")
    
    # Regression metrics
    reg = results['regression']
    print(f"\n📈 Regression:")
    print(f"  Best Model: {reg['best_model']}")
    print(f"  Best MAE: {reg['best_mae']:.3f}")
    print(f"  Best R²: {reg['best_r2']:.4f}")
    
    # Individual model results
    print(f"\n📊 Individual Models:")
    for model_name, metrics in reg['individual_results'].items():
        print(f"  {model_name}:")
        print(f"    MAE: {metrics['mae']:.3f}")
        print(f"    R²: {metrics['r2']:.4f}")
```

---

### Exemplo com Dict

```python
from src.integrated_pipeline import IntegratedPipeline

# Configuração como dict (útil para carregar de JSON/YAML)
config_dict = {
    'input_tsv': 'data.tsv',
    'output_dir': 'results/',
    'esm_model': 'esm2_t6_8M_UR50D',
    'run_classification': True,
    'run_regression': True,
    'regression_models': ['Ridge', 'XGBoost'],
    'verbose': True
}

pipeline = IntegratedPipeline(config_dict)
results = pipeline.run()
```

---

## 📊 Acessar Resultados

### JSON File

```bash
# Ver resultados completos
cat results/integrated_results.json | jq .

# Ver apenas status
cat results/integrated_results.json | jq .status

# Ver métricas de classificação
cat results/integrated_results.json | jq .classifier.test_metrics

# Ver melhor modelo de regressão
cat results/integrated_results.json | jq '.regression | {best_model, best_mae, best_r2}'
```

### Python

```python
import json

# Load results
with open("results/integrated_results.json") as f:
    results = json.load(f)

# Check status
if results['status'] == 'completed':
    print("✅ Pipeline completed successfully")
    
    # Classification metrics
    roc_auc = results['classifier']['test_metrics']['roc_auc']
    print(f"Classification ROC-AUC: {roc_auc:.4f}")
    
    # Regression metrics
    best_model = results['regression']['best_model']
    best_mae = results['regression']['best_mae']
    print(f"Best Regression Model: {best_model} (MAE: {best_mae:.3f})")
else:
    print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
```

---

## 🧪 Exemplos Prontos

```bash
# Ver exemplos interativos
python examples/integrated_pipeline_examples.py

# Executar exemplo específico
python examples/integrated_pipeline_examples.py 1  # Workflow completo
python examples/integrated_pipeline_examples.py 2  # Build only
python examples/integrated_pipeline_examples.py 3  # Build + Classification
python examples/integrated_pipeline_examples.py 4  # Build + Regression
```

---

## ⚙️ Opções CLI Completas

```bash
python -m src.integrated_pipeline --help

Options:
  --input INPUT              Input TSV file (required)
  --output OUTPUT            Output directory (default: results/integrated)
  --esm-model MODEL          ESM model name (default: esm2_t6_8M_UR50D)
  --device DEVICE            Device: cpu, cuda, mps (default: cpu)
  --no-classification        Skip classification phase
  --no-regression            Skip regression phase
  --regression-models MODEL  Regression models to train (default: Ridge Lasso ElasticNet RandomForest XGBoost)
  --random-state SEED        Random seed (default: 42)
  --quiet                    Suppress verbose output
  -h, --help                 Show help message
```

---

## 🚦 Status Codes

```python
# Status values in results['status']
'initialized'  # Pipeline created, not run yet
'completed'    # All phases completed successfully
'failed'       # Pipeline failed (check results['error'])
```

---

## 📁 Output Structure

```
results/
└── integrated/
    ├── integrated_results.json      # Consolidated results
    ├── build/
    │   ├── embeddings/
    │   │   ├── protein_embeddings.npy
    │   │   ├── ligand_embeddings.npy
    │   │   └── concatenated_embeddings.npy
    │   ├── matrix/
    │   │   └── embedding_matrix.npy
    │   ├── labels/
    │   │   ├── binary_labels.npy
    │   │   └── regression_targets.npy
    │   └── splits/
    │       ├── train_indices.npy
    │       ├── val_indices.npy
    │       └── test_indices.npy
    ├── classifier/
    │   ├── mlp_model.pth
    │   └── metrics.json
    └── regression/
        ├── models/
        │   ├── ridge_model.pkl
        │   ├── xgboost_model.pkl
        │   └── ...
        └── metrics.json
```

---

## ❓ Troubleshooting

### Issue: FileNotFoundError

```bash
FileNotFoundError: data/kinase_data.tsv not found
```

**Solution**: Verificar path do arquivo
```python
from pathlib import Path
input_path = Path("data/kinase_data.tsv").resolve()
print(f"Looking for: {input_path}")
print(f"Exists: {input_path.exists()}")
```

---

### Issue: CUDA out of memory

```bash
RuntimeError: CUDA out of memory
```

**Solution**: Usar modelo menor ou batch menor
```bash
python -m src.integrated_pipeline \
    --input data.tsv \
    --esm-model esm2_t6_8M_UR50D \  # Modelo menor
    --device cpu  # ou reduzir batch_size no código
```

---

### Issue: Module not found

```bash
ModuleNotFoundError: No module named 'src'
```

**Solution**: Executar do diretório raiz
```bash
cd /path/to/docktkinase  # Ir para raiz do projeto
python -m src.integrated_pipeline --input data.tsv --output results/
```

---

## 📚 Documentação Completa

- **Architecture**: `docs/03-architecture/integrated-pipeline.md`
- **Examples**: `examples/integrated_pipeline_examples.py`
- **Tests**: `tests/integration/test_integrated_pipeline.py`
- **Main README**: `README.md`

---

## 🎯 Next Steps

1. ✅ **Executar exemplo básico**
   ```bash
   python -m src.integrated_pipeline --input data/small_sample.tsv --output results/test
   ```

2. ✅ **Verificar resultados**
   ```bash
   cat results/test/integrated_results.json | jq .
   ```

3. ✅ **Explorar exemplos**
   ```bash
   python examples/integrated_pipeline_examples.py
   ```

4. ✅ **Customizar para seu caso**
   - Ajustar modelos de regressão
   - Escolher modelo ESM apropriado
   - Configurar splits e CV folds

---

**Status**: 🚀 **READY TO USE** ✅
