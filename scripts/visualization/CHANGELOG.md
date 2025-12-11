# Changelog - Modular Visualization System

## [2024-12-10] - Bug Fixes and Enhancements

### Fixed
1. **Distribution Boxplot Data Structure Bug**
   - **Issue**: Distribution plots were recording protein model names (e.g., "boltz2") ~125 times instead of showing algorithm names
   - **Root Cause**: Data collection loop was appending `'Model': model_name` instead of `'Algorithm': algo_name`
   - **Solution**: Changed data structure to include both:
     - `'Protein_Model'`: The protein embedding model (esm2_t30_150M, boltz2, etc.)
     - `'Algorithm'`: The ML algorithm (RandomForest, SVM, GradientBoosting, etc.)
   - **Impact**: Distribution plots now correctly show algorithm performance comparison across protein models

2. **Classification Distribution Plot Not Generated**
   - **Issue**: Classification distribution plot was returning None
   - **Root Cause**: JSON files use key `'classifier'` but code was looking for `'classification'`
   - **Solution**: Added backward compatibility support for both keys
   - **Impact**: Both classification and regression distribution plots now generate successfully

3. **Matplotlib Units Registry Pollution**
   - **Issue**: TypeError with ufunc 'add' due to incompatible dtypes (object vs float64)
   - **Root Cause**: StrCategoryConverter persisting in matplotlib units.registry between plots
   - **Solution**: Clear units.registry before AND after each plot using string labels
   - **Impact**: All plots generate without dtype conflicts

4. **JSON Data Type Issues**
   - **Issue**: Some models had string 'None' instead of numeric values
   - **Root Cause**: JSON serialization of None values
   - **Solution**: Added filtering and conversion to float(0.0) in metrics_extractor.py
   - **Impact**: Regression plots handle missing values gracefully

### Architecture
- **Modular Design**: 5 separate modules following SOLID principles
  - `data_loader.py`: JSON loading and validation
  - `metrics_extractor.py`: Metrics extraction and processing
  - `basic_plots.py`: 4 basic comparison plots
  - `advanced_plots.py`: 6 advanced scientific visualizations
  - `report_generator.py`: Markdown reports and CSV exports

### Outputs Generated
**Basic Comparison** (4 plots + 3 CSVs + 1 report):
- classification_comparison.png
- regression_comparison.png
- embedding_dimensions.png
- overall_ranking.png
- classification_metrics.csv
- regression_metrics.csv
- overall_scores.csv
- SUMMARY.md

**Advanced Analysis** (6 plots):
- radar_classification.png
- heatmap_performance.png
- tradeoff_analysis.png
- pareto_efficiency.png
- distribution_classification.png (✓ Fixed)
- distribution_regression.png (✓ Fixed)

### Data Structure
Distribution plots now correctly structure data as:
```python
{
    'Protein_Model': 'esm2_t30_150M_UR50D',  # Protein embedding model
    'Algorithm': 'RandomForest',              # ML algorithm
    'Metric': 'F1',                          # Performance metric
    'Value': 0.922                           # Metric value
}
```

This allows comparison of algorithm performance (x-axis) across different protein models (shown as distribution).

### Validation
- Tested with 6 protein models: esm2_t30_150M, esm2_t33_650M, esm2_t36_3B, boltz2, esmc-300m, esmc-600m
- Total entries per task: 6 models × 12 algorithms × 3 metrics = 216 data points
- All 10 visualizations (4 basic + 6 advanced) generate successfully
- Best performing model: esm2_t30_150M_UR50D (Score: 64.91)

### Technical Notes
- Python 3.12 with matplotlib 3.10+
- Uses Agg backend to avoid GUI issues
- Supports both 'classifier' and 'classification' keys (backward compatibility)
- Proper error handling for None/'None'/missing values
- Consistent plot proportions across all visualizations
