# 📊 Visualization System - DockTKinase (Non-Human Kinase Dataset)

## Overview

This modular visualization system is designed for analyzing and comparing **non-human kinase binding prediction models**, following **SOLID**, **KISS**, and **Clean Code** principles.

**Dataset:** Non-human kinase-ligand interactions for drug discovery and selectivity studies.

---

## 📁 Project Structure

```
scripts/visualization/
├── __init__.py              # Package initialization
├── data_loader.py           # JSON data loading
├── metrics_extractor.py     # Metrics extraction
├── basic_plots.py           # Basic visualizations
├── advanced_plots.py        # Advanced visualizations
└── report_generator.py      # Markdown report generation
```

---

## 📦 Modules

### 1. `data_loader.py`

**Responsibility**: Load and validate JSON result files for non-human kinase models.

```python
from visualization.data_loader import load_results_from_files

results = load_results_from_files(['file1.json', 'file2.json'])
```

**Classes**:
- `ResultsLoader`: Manages loading multiple files

**Functions**:
- `load_results_from_files()`: Simplified utility function

---

### 2. `metrics_extractor.py`

**Responsibility**: Extract and process classification and regression metrics.

```python
from visualization.metrics_extractor import MetricsExtractor, calculate_overall_score

extractor = MetricsExtractor(results)
classification_data = extractor.extract_classification_metrics()
regression_data = extractor.extract_regression_metrics()
```

**Classes**:
- `MetricsExtractor`: Extracts structured metrics from results

**Functions**:
- `calculate_overall_score()`: Calculates overall score combining metrics

**Supported Metrics**:
- Classification: Accuracy, F1, ROC_AUC, MCC, Precision, Recall
- Regression: Pearson_R, Pearson_P, RMSE, R2, MAE

---

### 3. `basic_plots.py`

**Responsibility**: Create basic comparison visualizations.

```python
from visualization.basic_plots import BasicPlotter

plotter = BasicPlotter(output_dir)
plotter.plot_classification_comparison(classification_data)
plotter.plot_regression_comparison(regression_data)
```

**Classes**:
- `BasicPlotter`: Generates standard comparison plots

**Methods**:
- `plot_classification_comparison()`: Compares classification metrics
- `plot_regression_comparison()`: Compares regression metrics (Pearson focus)
- `plot_embedding_dimensions()`: Visualizes embedding dimensions
- `plot_overall_ranking()`: Overall model ranking

---

### 4. `advanced_plots.py`

**Responsibility**: Create advanced scientific visualizations.

```python
from visualization.advanced_plots import AdvancedPlotter

plotter = AdvancedPlotter(output_dir)
plotter.create_radar_chart(classification_data)
plotter.create_heatmap(classification_data, regression_data)
```

**Classes**:
- `AdvancedPlotter`: Generates sophisticated visualizations

**Methods**:
- `create_radar_chart()`: Multi-dimensional radar chart
- `create_heatmap()`: Normalized performance heatmap
- `create_tradeoff_scatter()`: Trade-off analysis
- `create_pareto_chart()`: Efficiency vs performance (2 subplots: ranking + scatter)
- `create_distribution_boxplot()`: Algorithm distributions
- `create_regression_correlation_heatmap()`: Correlation between regression metrics

---

### 5. `report_generator.py`

**Responsibility**: Generate markdown reports and save CSVs.

```python
from visualization.report_generator import ReportGenerator, save_dataframes_to_csv

generator = ReportGenerator(output_dir)
generator.generate_summary(class_data, reg_data, emb_data, overall_data)
```

**Classes**:
- `ReportGenerator`: Creates structured markdown reports

**Functions**:
- `save_dataframes_to_csv()`: Saves data in CSV format

---

## 🚀 Main Scripts

### 1. **Classical ML Models Comparison** (Non-Human Kinase Dataset)

Compares 12 classical algorithms (XGBoost, Random Forest, LightGBM, etc.) across 6 protein models on **non-human kinase-ligand interactions**.

```bash
# Basic comparison
python scripts/compare_models_v2.py \
  --files results/integrated_results_*.json \
  --output results/protein_model_comparison

# Advanced analysis
python scripts/advanced_analysis.py \
  --files results/integrated_results_*.json \
  --output results/protein_model_advanced
```

**Outputs**:
- `classification_comparison.png` - 4 metrics (Accuracy, F1, ROC_AUC, MCC)
- `regression_comparison.png` - 4 metrics (Pearson_R, Pearson_P, RMSE, R2)
- `distribution_comparison.png` - Distribution by algorithm
- `overall_ranking.png` - Overall ranking
- `radar_classification.png` - Classification radar chart
- `heatmap_performance.png` - Performance heatmap
- `pareto_efficiency.png` - Pareto analysis (2 subplots: ranking + scatter)
- `scatter_matrix.png` - Scatter matrix
- `ridge_plot.png` - Ridge plot distributions
- `heatmap_regression_correlation.png` - Regression metrics correlation
- `full_report.md` - Complete report
- 3 CSV files with raw data

**Results Obtained (Non-Human Dataset)**:
- ✅ **Best model:** esm2_t30_150M (Score: 64.91)
- ✅ **Best classification:** esm2_t33_650M + RandomForest (F1: 0.927)
- ✅ **Best regression:** esm2_t30_150M + KNN (Pearson: 0.766)
- ✅ 6 protein models analyzed, 12 algorithms tested

---

### 2. **CNN+Cross-Attention Visualization** (Non-Human Kinase Dataset)

Analyzes CNN+CrossAttention architecture results on **non-human kinase binding predictions**.

```bash
# Single model analysis
python scripts/visualize_cnn_attention.py \
  --files results/results_esm2_t36_3B.json \
  --output results/cnn_single_model

# All models analysis
python scripts/visualize_cnn_attention.py \
  --files results/results_*.json \
  --output results/cnn_all_models
```

**Outputs**:
- `classification_comparison.png` - 2×2 grid (Accuracy, F1, ROC_AUC, MCC)
- `regression_comparison.png` - 2×2 grid (Pearson_R, Pearson_P, RMSE, R2)
- `embedding_dimensions.png` - Embedding dimensions
- `overall_ranking.png` - Overall ranking
- `radar_classification.png` - Radar chart
- `heatmap_performance.png` - Performance heatmap
- `pareto_efficiency.png` - Pareto efficiency
- `heatmap_regression_correlation.png` - Regression correlation
- `training_curves.png` - Training loss curves
- 3 CSV files with raw data

**Results Obtained (Non-Human Dataset, 7 models)**:
- ✅ **Best overall model:** esm2_t30_150M (Score: 88.80)
- ✅ **Best classification:** esm2_t30_150M (F1: 0.925, Acc: 0.958, ROC-AUC: 0.985)
- ✅ **Best regression:** esm2_t30_150M (Pearson: 0.851, R²: 0.487)
- ✅ Performance classified as "Excellent" for top model

---

### 3. **Classical ML vs CNN+Attention Comparison** (Non-Human Kinase Dataset)

Direct comparison between classical ML and CNN+CrossAttention on **non-human kinase binding predictions**.

```bash
python scripts/compare_classical_vs_cnn.py \
  --classical results/integrated_results_*.json \
  --cnn results/results_*.json \
  --output results/classical_vs_cnn_comparison
```

**Outputs**:
- `classification_classical_vs_cnn.png` - Side-by-side classification comparison
- `regression_classical_vs_cnn.png` - Side-by-side regression comparison
- `aggregated_comparison.png` - Average performance by type
- `best_algorithms_distribution.png` - Classical algorithms distribution
- `comparison_summary.md` - Comparison report
- 2 CSV files with comparative data

**Results Obtained (Non-Human Dataset)**:
- ✅ **Classification:** Classical ML slightly superior
  - Classical F1: 0.770 vs CNN F1: 0.735 (diff: 0.036)
  - Classical Acc: 0.833 vs CNN Acc: 0.827
- ✅ **Regression:** CNN+Attention **much superior**
  - CNN Pearson: 0.582 vs Classical: 0.307 (diff: **0.275** 🔥)
  - CNN R²: 0.139 vs Classical: -0.154 (diff: **0.292** 🔥)
- ✅ **Most used algorithms:**
  - Classification: RandomForest (3×), ExtraTrees (2×)
  - Regression: RandomForest (2×), KNN, Ridge, GradientBoosting

### **Visual Results & Discussion**

#### **Figure 1: Average Performance Comparison**
![Average Performance](../../docs/images/aggregated_comparison.png)

This aggregated view shows the **mean performance across all protein models**:

**Classification Metrics:**
- **Accuracy:** Both approaches achieve high accuracy (~0.83), indicating reliable binding/non-binding predictions
- **F1-Score:** Classical ML shows slight advantage (0.770 vs 0.735), suggesting better balance between precision and recall
- **ROC-AUC:** Classical ML performs better (0.894 vs 0.872), indicating superior discriminative capacity
- **MCC:** Classical ML achieves higher correlation (0.686 vs 0.582), showing stronger true positive/negative balance

**Regression Metrics (Binding Affinity Prediction):**
- **Pearson R:** CNN+Attention shows **dramatically superior** correlation (0.582 vs 0.307)
  - This 89% improvement indicates CNN captures binding affinity patterns much better
  - Critical for drug discovery where accurate affinity prediction is essential
- **R²:** CNN achieves positive explained variance (0.139) while Classical shows negative (-0.154)
  - Negative R² in Classical ML means predictions are **worse than simply using the mean**
  - CNN successfully learns meaningful relationships between structure and binding strength

**Key Insight:** Classical ML excels at discrete classification (binding/non-binding), while deep learning excels at continuous regression (binding affinity). This suggests **hybrid approaches** could leverage both strengths.

---

#### **Figure 2: Per-Model Classification Comparison**
![Classification Comparison](../../docs/images/classification_classical_vs_cnn.png)

This detailed comparison reveals **model-specific patterns**:

**Model Performance Analysis:**
- **boltz2:** Shows zero performance in Classical ML (data quality issue or model incompatibility)
- **esm2_t12_35M, esm2_t30_150M, esm2_t33_650M:** Classical ML consistently outperforms
  - F1 scores: 0.925-0.927 (Classical) vs 0.773-0.925 (CNN)
  - These ESM-2 models have strong sequence representations that Classical algorithms effectively exploit
- **esm2_t6_8M:** Smallest model shows comparable performance
- **esmc-300m, esmc-600m:** Performance parity between approaches

**Algorithmic Insights:**
- RandomForest and ExtraTrees dominate Classical ML selection
- These tree-based methods effectively capture non-linear patterns in protein embeddings
- CNN+CrossAttention shows more consistent performance across models (less variance)

**ROC-AUC Analysis:**
- All models achieve >0.9 ROC-AUC with Classical ML
- Exceptional discriminative power for kinase binding site identification
- Important for virtual screening where ranking compounds is critical

---

#### **Figure 3: Per-Model Regression Comparison**
![Regression Comparison](../../docs/images/regression_classical_vs_cnn.png)

This comparison highlights the **regression performance gap**:

**Critical Observations:**

1. **Pearson R (Binding Affinity Correlation):**
   - **CNN Strengths:** esm2_t12_35M (0.731), esm2_t30_150M (0.851), esm2_t33_650M (0.761)
   - **Classical Failures:** Multiple models show **negative correlations** or near-zero values
   - esm2_t33_650M with Ridge regression shows -0.342 (inverse correlation!)
   - This catastrophic failure suggests Classical ML cannot model the complex binding affinity landscape

2. **R² (Explained Variance):**
   - **CNN Success:** esm2_t30_150M achieves 0.487 (explains ~49% of variance)
   - **Classical Disaster:** esm2_t33_650M shows -1.946 (predictions amplify error by 3×!)
   - Negative R² indicates the model learned **anti-patterns** instead of true relationships

3. **RMSE/MAE (Prediction Error):**
   - Classical ML shows high variability (0.771 to 2.201 RMSE)
   - CNN maintains more consistent errors (0.940 to 1.354 RMSE)
   - Lower error consistency is crucial for reliable lead compound prioritization

**Root Cause Analysis:**
- **Classical ML Limitation:** Tree-based and linear models struggle with the **high-dimensional, non-linear** binding affinity landscape
- **CNN Advantage:** CrossAttention mechanism captures **long-range dependencies** between protein and ligand features
- **Embedding Quality:** Deep learning models learn task-specific representations during training, while Classical ML relies on fixed embeddings

**Practical Implications:**
- For **virtual screening** (classification): Use Classical ML for speed and interpretability
- For **lead optimization** (regression): Use CNN+Attention for accurate affinity prediction
- For **production systems**: Deploy hybrid pipeline with Classical ML filtering + CNN refinement

---

## 📈 Metrics Analyzed (Non-Human Kinase Binding Prediction)

### **Classification**
| Metric | Description | Range |
|---------|-----------|-------|
| **Accuracy** | % of correct predictions | 0-1 (higher better) |
| **F1-Score** | Harmonic mean Precision/Recall | 0-1 (higher better) |
| **ROC-AUC** | Area under ROC curve | 0-1 (higher better) |
| **MCC** | Matthews Correlation Coefficient | -1 to 1 (higher better) |
| **Precision** | True positive rate | 0-1 (higher better) |
| **Recall** | Positive detection rate | 0-1 (higher better) |

### **Regression** (Binding Affinity Prediction)
| Metric | Description | Range | Interpretation |
|---------|-----------|-------|---------------|
| **Pearson R** | Linear correlation | -1 to 1 | >0.7: strong, 0.3-0.7: moderate, <0.3: weak |
| **Pearson P** | Statistical significance | 0-1 | <0.05: significant |
| **Spearman R** | Monotonic correlation | -1 to 1 | Similar to Pearson |
| **R²** | Explained variance | -∞ to 1 | >0.7: excellent, 0.4-0.7: good, <0: worse than baseline |
| **RMSE** | Root mean squared error | 0-∞ | Lower better |
| **MAE** | Mean absolute error | 0-∞ | Lower better |

---

## 🎨 Visual Features

### **Color Palette**
- Classical ML: `#3498db` (blue)
- CNN+Attention: `#e74c3c` (red)
- Best model: `gold` with `darkorange` border

### **Layouts**
- Comparison plots: 2×2 grid (16×12 inches)
- Spacing: `hspace=0.35, wspace=0.3`
- DPI: 300 (high resolution)
- Fonts: Titles 13pt, labels 11pt, values 10pt

### **Features**
- ✅ Values on bars
- ✅ Reference lines (0.9 "Excellent", 0.7 "Good")
- ✅ Grid for readability
- ✅ Best model highlight (gold)
- ✅ Informative legend

---

## 🔧 Requirements

```bash
# Main dependencies
pip install matplotlib seaborn pandas numpy tabulate

# Already included in project's requirements.txt
```

---

## 📊 Complete Usage Example (Non-Human Kinase Dataset)

```bash
# 1. Complete classical models analysis
python scripts/compare_models_v2.py \
  --files results/integrated_results_*.json \
  --output results/ml_comparison

python scripts/advanced_analysis.py \
  --files results/integrated_results_*.json \
  --output results/ml_advanced

# 2. Complete CNN+Attention analysis
python scripts/visualize_cnn_attention.py \
  --files results/results_*.json \
  --output results/cnn_analysis

# 3. Direct comparison
python scripts/compare_classical_vs_cnn.py \
  --classical results/integrated_results_*.json \
  --cnn results/results_*.json \
  --output results/final_comparison
```

---

## 📝 Input Formats

### **Classical Models** (`integrated_results_*.json`) - Non-Human Kinase Dataset
```json
{
  "config": {...},
  "classifier": {
    "success": true,
    "best_model": "RandomForest",
    "best_metrics": {
      "Accuracy": 0.916,
      "F1": 0.927,
      "ROC_AUC": 0.973,
      "MCC": 0.829
    },
    "individual_results": {...}
  },
  "regression": {
    "success": true,
    "best_model": "KNN",
    "test_results": {
      "KNN": {
        "Pearson_R": 0.766,
        "R2": 0.558,
        "RMSE": 0.853,
        "MAE": 0.564
      }
    }
  }
}
```

### **CNN+Attention** (`results_*.json`) - Non-Human Kinase Dataset
```json
{
  "config": {...},
  "metrics": {
    "classification": {
      "accuracy": 0.958,
      "f1": 0.925,
      "roc_auc": 0.985,
      "mcc": 0.841
    },
    "regression": {
      "pearson_r": 0.851,
      "r2": 0.487,
      "rmse": 0.940,
      "mae": 0.671
    }
  },
  "training": {
    "train_loss": [...],
    "val_loss": [...]
  }
}
```

---

## 🐛 Troubleshooting

### **Error: "Missing optional dependency 'tabulate'"**
```bash
pip install tabulate
```

### **Error: matplotlib units registry**
- ✅ **Already fixed** - Code automatically clears registry
- See: `basic_plots.py` lines with `units.registry.clear()`

### **Error: 'None' values in metrics**
- ✅ **Already fixed** - Handling with `safe_float()` in `compare_classical_vs_cnn.py`

### **Distorted plots**
- ✅ **Already fixed** - Layouts adjusted to 16×12 with proper spacing

---

## 💡 Usage Examples

### Basic Usage

```python
from pathlib import Path
from visualization.data_loader import load_results_from_files
from visualization.metrics_extractor import MetricsExtractor
from visualization.basic_plots import BasicPlotter

# Load data (non-human kinase results)
results = load_results_from_files(['model1.json', 'model2.json'])

# Extract metrics
extractor = MetricsExtractor(results)
class_data = extractor.extract_classification_metrics()

# Plot
plotter = BasicPlotter(Path('output'))
plotter.plot_classification_comparison(class_data)
```

### Advanced Usage

```python
from visualization.advanced_plots import AdvancedPlotter
from visualization.report_generator import ReportGenerator

# Advanced visualizations
adv_plotter = AdvancedPlotter(Path('output'))
adv_plotter.create_radar_chart(class_data)
adv_plotter.create_heatmap(class_data, reg_data)

# Generate report
generator = ReportGenerator(Path('output'))
generator.generate_summary(class_data, reg_data, emb_data, overall_data)
```

---

## 🧬 Dataset Information

- **Dataset:** Non-human kinase-ligand interactions
- **Purpose:** Drug discovery, selectivity studies, cross-species analysis
- **Applications:** 
  - Kinase inhibitor binding prediction
  - Species-specific selectivity analysis
  - Cross-species model generalization
  - Lead compound optimization

---

## 🔧 Extensibility

### Add New Metric

1. Add to list in `metrics_extractor.py`:
```python
REGRESSION_METRICS = ['Pearson_R', 'Pearson_P', 'RMSE', 'R2', 'MAE', 'New_Metric']
```

2. Update extraction method if needed

### Add New Visualization

1. Create method in appropriate class (`BasicPlotter` or `AdvancedPlotter`)
2. Follow existing pattern: accept data, return Path
3. Use helper methods for code reuse

---

## 📊 Prioritized Metrics (Non-Human Kinase Binding)

### Classification (Binding/Non-binding)
- F1 Score (balance)
- ROC-AUC (discriminative capacity)
- MCC (correlation coefficient)

### Regression (Binding Affinity)
1. **Pearson R**: Linear correlation strength
2. **Pearson P-value**: Statistical significance
3. **RMSE**: Prediction error magnitude
4. R²: Explained variance (reference)

---

## 📝 Code Conventions

- **Imports**: grouped and ordered (stdlib, third-party, local)
- **Docstrings**: Google style for all functions/classes
- **Type hints**: used whenever possible
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Private methods**: `_` prefix for internal methods

---

## 🎓 Benefits of Modular Architecture

✅ **Maintainability**: Easy to locate and fix bugs
✅ **Testability**: Each module can be tested in isolation
✅ **Reusability**: Modules can be used in other projects
✅ **Scalability**: Easy to add new features
✅ **Readability**: Organized, self-documenting code
✅ **Collaboration**: Team can work on different modules simultaneously

---

## 📚 References

- **Design Principles:** SOLID, KISS, Clean Code
- **Plotting Library:** Matplotlib + Seaborn
- **Data Analysis:** Pandas + NumPy
- **Metrics:** scikit-learn compatible
- **Dataset:** Non-human kinase-ligand interactions

---

## 📧 Support

For questions or issues:
1. Check this README
2. Consult `CRITICAL_FIXES_APPLIED.md` for metric-related questions
3. Check generated `.md` reports for automatic insights

---

**Last Updated:** 2025-12-10  
**Version:** 2.0 (With critical metric corrections)  
**Dataset:** Non-Human Kinase-Ligand Interactions
