# Quick Reference: Dual-Dataset Ablation Study

## 🚀 Quick Start

### Option 1: Use Orchestrator (Recommended)
```bash
cd /media/leon/ssd2tb/docktkinase/ablation

# Non-human dataset (already complete)
python run_ablation_study.py --dataset non_human

# Human dataset (new)
python run_ablation_study.py --dataset human

# Both datasets
python run_ablation_study.py --dataset both
```

### Option 2: Manual Execution

#### Non-Human Classification
```bash
cd classification/scripts
python 01_extract_data.py --results-suffix results_non_human
python 02_generate_morgan_fingerprints.py --results-suffix results_non_human
python 03_generate_aac_dpc_encoding.py --results-suffix results_non_human
python 04_create_combinations.py --results-suffix results_non_human \
    --embeddings-dir /media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2
python 05_run_classification.py --results-suffix results_non_human
python 06_visualize_results.py --results-suffix results_non_human
```

#### Human Classification
```bash
cd classification/scripts
python 01_extract_data.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human
python 02_generate_morgan_fingerprints.py --results-suffix results_human
python 03_generate_aac_dpc_encoding.py --results-suffix results_human
python 04_create_combinations.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2
python 05_run_classification.py --results-suffix results_human
python 06_visualize_results.py --results-suffix results_human
```

#### Non-Human Regression
```bash
cd regression/scripts
python 01_extract_data_regression.py --results-suffix results_non_human
nohup python -u 02_run_regression.py --results-suffix results_non_human \
    --embeddings-dir /media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2 \
    > ../results_non_human/regression.log 2>&1 &
# Wait for completion, then:
python consolidate_checkpoints.py --results-suffix results_non_human
python 03_visualize_regression_results.py --results-suffix results_non_human
```

#### Human Regression
```bash
cd regression/scripts
python 01_extract_data_regression.py \
    --tsv-path /data/docktkinase/datasets/kinase_human_compounds.tsv \
    --results-suffix results_human
nohup python -u 02_run_regression.py --results-suffix results_human \
    --embeddings-dir /data/docktkinase/results/protein_model_benchmark_human_v2 \
    > ../results_human/regression.log 2>&1 &
# Wait for completion, then:
python consolidate_checkpoints.py --results-suffix results_human
python 03_visualize_regression_results.py --results-suffix results_human
```

## 📊 Dataset Paths

| Dataset | TSV Path | Embeddings Path | Results Suffix |
|---------|----------|-----------------|----------------|
| **Non-Human** | `/media/leon/ssd2tb/docktkinase/tests/datasets/kinase_non_human_compounds.tsv` | `/media/leon/ssd2tb/docktkinase/results/protein_model_benchmark_non_human_v2/` | `results_non_human` |
| **Human** | `/data/docktkinase/datasets/kinase_human_compounds.tsv` | `/data/docktkinase/results/protein_model_benchmark_human_v2/` | `results_human` |

## 📁 Output Locations

### Classification Results
- **Non-Human**: `classification/results_non_human/`
  - `classification_results.json` (full details)
  - `classification_summary.csv` (aggregated metrics)
  - `figures/` (visualizations)

- **Human**: `classification/results_human/`
  - Same structure

### Regression Results
- **Non-Human**: `regression/results_non_human/`
  - `regression_results.json` (full details)
  - `regression_summary.csv` (aggregated metrics)
  - `regression_summary_*_seed*.csv` (checkpoints)
  - `figures/` (visualizations)

- **Human**: `regression/results_human/`
  - Same structure

## 🔍 Monitoring Progress

### Check Regression Status
```bash
# Non-human
tail -f regression/results_non_human/regression.log

# Human
tail -f regression/results_human/regression.log
```

### Check Process
```bash
ps aux | grep "python.*regression"
```

## 📈 Result Comparison

### Load and Compare (Python)
```python
import pandas as pd

# Classification
df_nh = pd.read_csv('classification/results_non_human/classification_summary.csv')
df_h = pd.read_csv('classification/results_human/classification_summary.csv')

print("Non-Human ROC-AUC:", df_nh['test_auc'].mean())
print("Human ROC-AUC:", df_h['test_auc'].mean())

# Regression
df_reg_nh = pd.read_csv('regression/results_non_human/regression_summary.csv')
df_reg_h = pd.read_csv('regression/results_human/regression_summary.csv')

print("Non-Human R²:", df_reg_nh['test_r2'].mean())
print("Human R²:", df_reg_h['test_r2'].mean())
```

### Quick Stats (Bash)
```bash
# Non-human classification summary
csvstat classification/results_non_human/classification_summary.csv --mean

# Human regression summary
csvstat regression/results_human/regression_summary.csv --mean
```

## ⏱️ Estimated Time

| Task | Non-Human | Human | Notes |
|------|-----------|-------|-------|
| **Classification** |
| Data extraction | 1 min | ~1 min | Depends on size |
| Morgan FP | 2 min | ~2 min | RDKit computation |
| One-Hot | 1 min | ~1 min | Simple encoding |
| Combinations | 5 min | ~5 min | 10 combinations |
| Experiments | 30 min | ~45 min | 100 experiments (5 seeds × 10 combos × 2 classifiers) |
| Visualization | 2 min | ~2 min | Generate plots |
| **Total Classification** | ~40 min | ~60 min | |
| **Regression** |
| Data extraction | <1 min | <1 min | Reuses classification |
| Experiments | 3-4 hours | ~4-6 hours | 30 experiments (5 seeds × 3 models × 2 regressors), MLP training |
| Consolidation | <1 min | <1 min | Merge CSVs |
| Visualization | 2 min | ~2 min | Generate plots |
| **Total Regression** | ~4 hours | ~6 hours | |
| **GRAND TOTAL** | ~4.5 hours | ~7 hours | Per dataset |

## 🎯 Command Arguments

All scripts accept:
- `--tsv-path`: Input TSV file path
- `--results-suffix`: Results directory suffix (`results_non_human` or `results_human`)
- `--embeddings-dir`: ESM-2/SMI-TED embeddings directory

Defaults to non-human if not specified.

## ✅ Verification

### Check Results Exist
```bash
# Non-human
ls classification/results_non_human/classification_summary.csv
ls regression/results_non_human/regression_summary.csv

# Human
ls classification/results_human/classification_summary.csv
ls regression/results_human/regression_summary.csv
```

### Check Figures
```bash
# Non-human
ls classification/results_non_human/figures/
ls regression/results_non_human/figures/

# Human
ls classification/results_human/figures/
ls regression/results_human/figures/
```

## 🐛 Troubleshooting

### Issue: "TSV file not found"
**Solution**: Verify path exists
```bash
ls -lh /data/docktkinase/datasets/kinase_human_compounds.tsv
```

### Issue: "Embeddings directory not found"
**Solution**: Check ESM-2 embeddings exist
```bash
ls /data/docktkinase/results/protein_model_benchmark_human_v2/
```

### Issue: "No checkpoint files found"
**Solution**: Ensure regression experiments completed
```bash
ls regression/results_human/regression_summary_*.csv
```

## 📚 Documentation

- **Main README**: [README.md](README.md) - Complete documentation
- **Dual-Dataset Guide**: [DUAL_DATASET_GUIDE.md](DUAL_DATASET_GUIDE.md) - Detailed usage
- **Migration Summary**: [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) - Technical changes

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 17, 2026
