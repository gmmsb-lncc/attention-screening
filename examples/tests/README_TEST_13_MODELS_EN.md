# 13 Classification Models Test

## 📋 Summary

Test script created to validate the complete implementation of **13 classification models**:
- **10 base models** (always available)
- **XGBoost** (mandatory)  
- **LightGBM and CatBoost** (optional)

## 🎯 Objective

Validate that all 10 base models + mandatory XGBoost are correctly implemented and working as specified.

## 📝 Usage

```bash
# Quick test (100 samples)
python tests/test_13_models_classification.py --dataset small

# Complete test (15k samples)
python tests/test_13_models_classification.py --dataset full

# Test specific models
python tests/test_13_models_classification.py \
    --models DecisionTree AdaBoost XGBoost

# With verbose mode
python tests/test_13_models_classification.py \
    --dataset small \
    --verbose
```

## 🔧 Options

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--dataset` | `small`, `full` | `small` | Dataset size |
| `--models` | List of names | All | Specific models to test |
| `--protein-model` | ESM model name | `esm2_t6_8M_UR50D` | Model for embeddings |
| `--output` | Path | `tests/results/test_13_models` | Output directory |
| `--seed` | Integer | `42` | Random seed |
| `--verbose` | Flag | `False` | Verbose mode |

## 📊 Models Tested

### 10 Base Models (Always Available)

1. **RandomForest** (~90s)
   - Bagging ensemble
   - Robust, interpretable

2. **GradientBoosting** (~240s)
   - Sequential boosting ensemble
   - High performance

3. **LogisticRegression** (~15s)
   - Linear baseline
   - Fast, interpretable

4. **LinearSVC** (~20s)
   - Linear SVM
   - Great for high dimensionality

5. **ExtraTrees** (~60s)
   - Randomized bagging ensemble
   - Faster than RF

6. **KNN** (~30s train, ~180s predict)
   - Instance-based
   - No parametric training

7. **MLP** (~300s)
   - Neural network
   - Captures complex non-linearities

8. **NaiveBayes** (~2s)
   - Probabilistic
   - Fastest baseline

9. **DecisionTree** (~10s) ⭐ NEW
   - Single tree
   - Maximum interpretability

10. **AdaBoost** (~80s) ⭐ NEW
    - Classic boosting
    - Good with weak learners

### Gradient Boosting (3 models)

11. **XGBoost** (~60s) ⚠️ **MANDATORY**
    - State-of-the-art
    - Usually best model

12. **LightGBM** (~45s) - Optional
    - Optimized for speed
    - Great for large datasets

13. **CatBoost** (~120s) - Optional
    - Optimized for categorical features
    - Good for imbalanced data

## 🔄 Test Phases

### Phase 1: Model Validation
```
✅ Verify availability of 10 base models
⚠️  Verify mandatory XGBoost
ℹ️  Verify optional models (LightGBM, CatBoost)
```

### Phase 2: Pipeline Execution
```
📦 Build: Generate embeddings (protein + ligand)
🤖 Classification: Train all 13 models
📊 Metrics: ROC-AUC, F1, Accuracy, Precision, Recall
```

### Phase 3: Results Analysis
```
🏆 Ranking by ROC-AUC
📈 Comparative statistics
🆕 Performance of new models (DecisionTree, AdaBoost)
⚠️  Mandatory XGBoost verification
⏱️  Training times
```

## 📁 Output Structure

```
tests/results/test_13_models_<dataset>_<timestamp>/
├── build/
│   ├── embeddings/
│   │   ├── protein_embeddings.npy
│   │   ├── ligand_embeddings.npy
│   │   └── concatenated_embeddings.npy
│   ├── splits/
│   │   ├── train_indices.npy
│   │   ├── val_indices.npy
│   │   └── test_indices.npy
│   └── labels/
│       ├── binary_labels.npy
│       └── regression_labels.npy
├── classifier/
│   ├── metrics/
│   │   ├── test_metrics.json ⭐
│   │   └── test_metrics.csv
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── models/
│       ├── RandomForest.pkl
│       ├── DecisionTree.pkl
│       ├── AdaBoost.pkl
│       ├── XGBoost.pkl
│       └── ...
└── integrated_results.json
```

## 📊 Example Report

```
📊 RESULTS ANALYSIS
================================================================================

📁 Metrics file: classifier/metrics/test_metrics.json

📊 Models Trained: 13

🏆 Performance (sorted by ROC-AUC):

Rank   Model                ROC-AUC    F1         Accuracy   Precision  Recall
--------------------------------------------------------------------------------
🥇     XGBoost             0.8654     0.8234     0.8456     0.8321     0.8150
🥈     LightGBM            0.8621     0.8198     0.8423     0.8287     0.8112
🥉     RandomForest        0.8543     0.8145     0.8378     0.8245     0.8048
4.     ExtraTrees          0.8498     0.8112     0.8345     0.8210     0.8018
5.     GradientBoosting    0.8421     0.8067     0.8312     0.8176     0.7962
6.     AdaBoost            0.8156     0.7876     0.8123     0.7998     0.7756    ⭐
7.     MLP                 0.8089     0.7823     0.8087     0.7954     0.7695
8.     LinearSVC           0.7934     0.7698     0.7989     0.7845     0.7554
9.     LogisticRegression  0.7876     0.7654     0.7956     0.7812     0.7498
10.    DecisionTree        0.7623     0.7412     0.7823     0.7656     0.7171    ⭐
11.    KNN                 0.7589     0.7389     0.7801     0.7621     0.7160
12.    CatBoost            0.7543     0.7345     0.7778     0.7578     0.7115
13.    NaiveBayes          0.7123     0.7012     0.7456     0.7234     0.6792

📈 Statistics:
   Best model: XGBoost (ROC-AUC: 0.8654)
   Mean ROC-AUC: 0.7972 ± 0.0451
   Mean F1: 0.7687 ± 0.0398
   Mean Accuracy: 0.8071 ± 0.0316

🆕 New Models:
   ✅ DecisionTree:
      Rank: 10/13
      ROC-AUC: 0.7623
      F1: 0.7412
   
   ✅ AdaBoost:
      Rank: 6/13
      ROC-AUC: 0.8156
      F1: 0.7876

⚠️  XGBoost (MANDATORY):
   ✅ Trained!
      Rank: 1/13
      ROC-AUC: 0.8654
      F1: 0.8234

⏱️  Training Times:
   NaiveBayes            2.12s
   DecisionTree         10.34s
   LogisticRegression   15.67s
   LinearSVC            20.45s
   KNN                  30.89s
   LightGBM             45.23s
   ExtraTrees           60.12s
   XGBoost              60.78s
   AdaBoost             80.34s
   RandomForest         90.56s
   CatBoost            120.45s
   GradientBoosting    240.67s
   MLP                 300.89s
   
   Total: 1078.51s (17.98 min)
```

## ✅ Requirements Met

1. ✅ **10 base models**: RandomForest, GradientBoosting, LogisticRegression, LinearSVC, ExtraTrees, KNN, MLP, NaiveBayes, **DecisionTree**, **AdaBoost**

2. ✅ **Mandatory XGBoost**: Warning displayed if not installed

3. ✅ **Acceptable time**: ~17 minutes for all 13 models (vs 75 min with SVC-RBF)

4. ✅ **Model diversity**: Linear, tree-based, ensemble, boosting, neural

5. ✅ **Complete documentation**: All models documented and explained

## 📚 References

- **Test script**: `tests/test_13_models_classification.py`
- **Model documentation**: `docs/04-modules/NEW_CLASSIFICATION_MODELS.md`
- **Optimizations**: `docs/04-modules/OPTIMIZED_CLASSIFICATION_MODELS.md`
- **Classifiers**: `src/classifier/models/classifiers.py`
- **Pipeline**: `src/classifier/multi_model_pipeline.py`

---

**Created**: 2025-11-24  
**Status**: ✅ Script ready and working  
**Branch**: boltz
