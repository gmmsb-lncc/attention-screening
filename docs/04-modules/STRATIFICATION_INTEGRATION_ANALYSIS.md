# Stratification Integration - Current Architecture Analysis

## 🔍 Current Architecture Overview

### BuildPipeline (Main Orchestrator)
**Location**: `src/build/pipeline/build_pipeline.py`

**Current Flow**:
```
1. run_embedding_generation()
   ↓
2. run_matrix_construction()
   ↓
3. run_label_generation()
   ↓
4. run_stratification() [OPTIONAL - currently standalone]
   ↓
5. run_validation()
```

**Key Observations**:
- ✅ Already has `Stratifier` component initialized
- ✅ Already has `run_stratification()` method
- ⚠️ Stratification results NOT shared with classification/regression
- ⚠️ Classification and regression pipelines called OUTSIDE BuildPipeline
- ⚠️ No mechanism to pass split indices to downstream pipelines

### ClassificationPipeline
**Location**: `src/classifier/modular_pipeline.py`

**Current Split Logic**:
```python
class MLPEmbeddingPipeline:
    def load_data(self):
        # Current: Creates OWN random splits
        train_test = self.data_manager.load_and_split_data(...)
        # Uses StratifiedKFold internally
        # NO support for external split indices
```

**Key Observations**:
- ❌ No parameter to receive external split indices
- ❌ Always creates its own splits using StratifiedKFold
- ❌ No way to use stratification from BuildPipeline
- ⚠️ Uses Spark for metrics (complex dependency)

### RegressionPipeline
**Location**: `src/regression/modular_pipeline.py`

**Current Split Logic**:
```python
class RegressionPipeline:
    def load_data(self):
        # Current: Creates OWN random splits
        X_train, X_val, X_test = self.data_manager.load_and_split(...)
        # Uses train_test_split from sklearn
        # NO support for external split indices
```

**Key Observations**:
- ❌ No parameter to receive external split indices
- ❌ Always creates its own splits (different from classification!)
- ❌ No way to use stratification from BuildPipeline
- ⚠️ Splits are DIFFERENT from classification (DATA LEAKAGE!)

## 🚨 Critical Problem Identified

### The Data Leakage Issue

**Current Reality**:
```
BuildPipeline:
├── Generates embeddings (100 samples)
├── Can stratify: train=[0,1,2...], val=[50,51...], test=[80,81...]
│
ClassificationPipeline:
├── Loads same embeddings
└── Creates NEW random splits: train=[5,10,15...], val=[30,35...], test=[70,75...]
    ❌ DIFFERENT INDICES!
│
RegressionPipeline:
├── Loads same embeddings
└── Creates ANOTHER set of random splits: train=[2,8,12...], val=[40,45...], test=[85,90...]
    ❌ DIFFERENT INDICES AGAIN!
```

**Consequence**: 
- Sample #5 might be in TRAIN for classification but TEST for regression
- Sample #80 might be in TEST for classification but TRAIN for regression
- **This is DATA LEAKAGE and makes results INVALID for comparison!**

## 🎯 Required Solution Architecture

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BuildPipeline                             │
│                                                                   │
│  1. Generate embeddings (protein + ligand)                       │
│  2. Generate labels (classification + regression)                │
│  3. STRATIFY ONCE ← NEW: Single stratification point             │
│     ├── multi_view_stratified_split()                            │
│     ├── Returns: train_idx, val_idx, test_idx                    │
│     └── Save: stratified_splits.npz                              │
│  4. Pass splits to BOTH pipelines ← NEW: Shared splits           │
│     ├── ClassificationPipeline(split_indices=splits) ← NEW       │
│     └── RegressionPipeline(split_indices=splits) ← NEW           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │ Classification    │       │ Regression        │
        │                   │       │                   │
        │ Uses SAME indices │       │ Uses SAME indices │
        │ train_idx         │       │ train_idx         │
        │ val_idx           │       │ val_idx           │
        │ test_idx          │       │ test_idx          │
        └───────────────────┘       └───────────────────┘
```

## 📋 Detailed Implementation Requirements

### 1. Create SplitIndices Data Class

**Purpose**: Type-safe, immutable container for split indices

**File**: `src/build/pipeline/split_indices.py` (NEW)

**Requirements**:
- ✅ Immutable dataclass
- ✅ Stores train_idx, val_idx, test_idx as np.ndarray (int32)
- ✅ Validates no overlap between splits
- ✅ Validates all indices are unique
- ✅ Validates indices cover full dataset
- ✅ Can save to .npz file
- ✅ Can load from .npz file
- ✅ Can convert to/from dict
- ✅ Stores metadata (timestamp, config, sizes, etc.)

**Design Principles Applied**:
- **SRP**: Single responsibility = hold split indices
- **Immutable**: Created once, never modified
- **Type-safe**: np.ndarray with explicit dtype=int32

### 2. Create StratificationManager

**Purpose**: Encapsulate all stratification logic

**File**: `src/build/pipeline/stratification_manager.py` (NEW)

**Requirements**:
- ✅ Wraps existing `Stratifier` from stratification module
- ✅ Caches splits (stratify once, use many times)
- ✅ Can save splits to disk
- ✅ Can load splits from disk
- ✅ Returns `SplitIndices` object
- ✅ Fallback to random splitting if stratification fails
- ✅ Configurable via BuildConfig

**Design Principles Applied**:
- **SRP**: Single responsibility = manage stratification
- **OCP**: Can extend with new splitting strategies
- **DIP**: Returns abstract SplitIndices, not implementation details

**Interface**:
```python
class StratificationManager:
    def __init__(self, config: BuildConfig)
    
    def stratify(
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> SplitIndices
    
    def get_splits(self) -> SplitIndices
    def save_splits(self, filepath: str) -> None
    def load_splits(self, filepath: str) -> SplitIndices
```

### 3. Update BuildPipeline

**Purpose**: Integrate stratification and pass splits to pipelines

**File**: `src/build/pipeline/build_pipeline.py` (MODIFY)

**Required Changes**:

**Change 1**: Add StratificationManager
```python
def _initialize_components(self):
    self.components = {
        # ... existing components ...
        'stratification_manager': StratificationManager(self.config)  # NEW
    }
```

**Change 2**: Modify `run_complete_pipeline()` to stratify and pass splits
```python
def run_complete_pipeline(self, ...):
    # 1. Generate embeddings
    self.run_embedding_generation(...)
    
    # 2. Build matrix
    self.run_matrix_construction(...)
    
    # 3. Generate labels
    self.run_label_generation(...)
    
    # 4. STRATIFY ONCE (NEW)
    split_indices = self._perform_stratification(
        protein_embeddings_path=...,
        ligand_embeddings_path=...,
        labels_path=...
    )
    
    # 5. Save splits for reproducibility (NEW)
    self._save_splits(split_indices, output_dir)
    
    # 6. Run classification WITH splits (MODIFIED)
    if run_classification:
        self._run_classification_pipeline(
            embeddings_path=...,
            labels_path=...,
            split_indices=split_indices  # NEW PARAMETER
        )
    
    # 7. Run regression WITH SAME splits (MODIFIED)
    if run_regression:
        self._run_regression_pipeline(
            embeddings_path=...,
            targets_path=...,
            split_indices=split_indices  # NEW PARAMETER (SAME!)
        )
```

**Change 3**: Add helper methods
```python
def _perform_stratification(self, ...) -> SplitIndices:
    """Perform stratification using StratificationManager."""
    # Load embeddings and labels
    # Call stratification_manager.stratify()
    # Return SplitIndices
    
def _save_splits(self, split_indices: SplitIndices, output_dir: Path):
    """Save split indices to disk."""
    
def _run_classification_pipeline(self, ..., split_indices: SplitIndices):
    """Run classification pipeline with provided splits."""
    # Import ClassificationPipeline
    # Initialize with split_indices
    # Run pipeline
    
def _run_regression_pipeline(self, ..., split_indices: SplitIndices):
    """Run regression pipeline with provided splits."""
    # Import RegressionPipeline
    # Initialize with split_indices
    # Run pipeline
```

**Design Principles Applied**:
- **SRP**: BuildPipeline orchestrates, doesn't implement splitting
- **OCP**: Easy to add new pipeline types
- **DIP**: Depends on SplitIndices interface, not concrete implementation

### 4. Update ClassificationPipeline

**Purpose**: Accept and use external split indices

**File**: `src/classifier/modular_pipeline.py` (MODIFY)

**Required Changes**:

**Change 1**: Add optional parameter to `__init__`
```python
def __init__(
    self,
    embeddings_path: str,
    labels_path: str,
    split_indices: Optional[SplitIndices] = None,  # NEW PARAMETER
    # ... existing parameters ...
):
    self.split_indices = split_indices  # NEW
    # ... existing code ...
```

**Change 2**: Modify `load_data()` to use provided splits
```python
def load_data(self):
    # Load embeddings and labels
    X, y = self.data_manager.load_data()
    
    # Use provided splits if available (NEW)
    if self.split_indices is not None:
        train_idx = self.split_indices.train_idx
        val_idx = self.split_indices.val_idx
        test_idx = self.split_indices.test_idx
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
    else:
        # Fallback to existing split logic (BACKWARD COMPATIBLE)
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.data_manager.load_and_split_data(...)
    
    # Create data loaders
    self.train_loader = self._create_loader(X_train, y_train)
    self.val_loader = self._create_loader(X_val, y_val)
    self.test_loader = self._create_loader(X_test, y_test)
```

**Design Principles Applied**:
- **SRP**: Pipeline focuses on training, not splitting
- **OCP**: Accepts splits from any source
- **Backward Compatible**: Optional parameter maintains existing behavior

### 5. Update RegressionPipeline

**Purpose**: Accept and use external split indices (SAME as classification!)

**File**: `src/regression/modular_pipeline.py` (MODIFY)

**Required Changes**:

**Change 1**: Add optional parameter to `__init__`
```python
def __init__(
    self,
    embeddings_path: str,
    targets_path: str,
    split_indices: Optional[SplitIndices] = None,  # NEW PARAMETER
    # ... existing parameters ...
):
    self.split_indices = split_indices  # NEW
    # ... existing code ...
```

**Change 2**: Modify `load_data()` to use provided splits
```python
def load_data(self):
    # Load embeddings and targets
    X, y = self.data_manager.load_data()
    
    # Use provided splits if available (NEW)
    if self.split_indices is not None:
        train_idx = self.split_indices.train_idx
        val_idx = self.split_indices.val_idx
        test_idx = self.split_indices.test_idx
        
        self.X_train, self.y_train = X[train_idx], y[train_idx]
        self.X_val, self.y_val = X[val_idx], y[val_idx]
        self.X_test, self.y_test = X[test_idx], y[test_idx]
    else:
        # Fallback to existing split logic (BACKWARD COMPATIBLE)
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = \
            self.data_manager.load_and_split(...)
```

**Design Principles Applied**:
- **SRP**: Pipeline focuses on training, not splitting
- **OCP**: Accepts splits from any source
- **DIP**: Depends on SplitIndices interface
- **Backward Compatible**: Optional parameter maintains existing behavior

## 🧪 Testing Strategy

### Test Hierarchy

```
Unit Tests (test components in isolation)
├── test_split_indices.py
│   ├── test_validation (no overlap, all covered)
│   ├── test_save_load
│   └── test_immutability
├── test_stratification_manager.py
│   ├── test_stratify_once
│   ├── test_cache_splits
│   └── test_fallback_random
│
Integration Tests (test components together)
├── test_stratification_integration.py
│   ├── test_buildpipeline_stratifies_once
│   ├── test_classification_uses_splits
│   ├── test_regression_uses_same_splits
│   └── test_splits_consistency
│
End-to-End Tests (test full workflow)
└── test_complete_pipeline_with_stratification.py
    ├── test_full_pipeline_flow
    ├── test_results_consistency
    └── test_reproducibility
```

### Critical Tests

**Test 1: Split Consistency** (MOST IMPORTANT!)
```python
def test_classification_and_regression_use_same_splits():
    """
    Verify that classification and regression use IDENTICAL splits.
    This is the MAIN GOAL of the entire integration.
    """
    # Run BuildPipeline with stratification
    pipeline = BuildPipeline(config)
    pipeline.run_complete_pipeline(...)
    
    # Get splits from classification
    clf_splits = classification_pipeline.split_indices
    
    # Get splits from regression
    reg_splits = regression_pipeline.split_indices
    
    # Assert they are IDENTICAL
    assert np.array_equal(clf_splits.train_idx, reg_splits.train_idx)
    assert np.array_equal(clf_splits.val_idx, reg_splits.val_idx)
    assert np.array_equal(clf_splits.test_idx, reg_splits.test_idx)
```

**Test 2: No Data Leakage**
```python
def test_no_overlap_between_splits():
    """Verify train/val/test have no overlap."""
    splits = stratification_manager.stratify(...)
    
    train_set = set(splits.train_idx)
    val_set = set(splits.val_idx)
    test_set = set(splits.test_idx)
    
    assert len(train_set & val_set) == 0  # No overlap train-val
    assert len(train_set & test_set) == 0  # No overlap train-test
    assert len(val_set & test_set) == 0  # No overlap val-test
```

**Test 3: Backward Compatibility**
```python
def test_pipelines_work_without_splits():
    """Verify pipelines still work without external splits."""
    # Classification without splits
    clf = MLPEmbeddingPipeline(
        embeddings_path='...',
        labels_path='...'
        # NO split_indices parameter
    )
    clf.load_data()  # Should create its own splits
    assert clf.train_loader is not None
    
    # Regression without splits
    reg = RegressionPipeline(
        embeddings_path='...',
        targets_path='...'
        # NO split_indices parameter
    )
    reg.load_data()  # Should create its own splits
    assert reg.X_train is not None
```

## 📊 Configuration Management

### BuildConfig Extensions

**File**: `src/build/core/config.py` (MODIFY)

**Add stratification configuration**:
```python
# Stratification settings
STRATIFICATION_ENABLED: bool = True
STRATIFICATION_PARAMS: Dict[str, Any] = {
    'clustering_algorithm': 'kmeans',
    'similarity_threshold': 0.7,
    'test_size': 0.2,
    'val_size': 0.1,
    'protein_weight': 0.6,
    'ligand_weight': 0.4,
    'random_state': 42,
    'save_splits': True,
    'splits_filename': 'stratified_splits.npz'
}

# Pipeline integration settings
RUN_CLASSIFICATION: bool = True
RUN_REGRESSION: bool = True
SHARE_SPLITS: bool = True  # NEW: Use same splits for both
```

### Example Configuration File

**File**: `stratification_config.json` (NEW)

```json
{
  "build": {
    "stratification_enabled": true,
    "run_classification": true,
    "run_regression": true,
    "share_splits": true
  },
  "stratification": {
    "clustering_algorithm": "kmeans",
    "similarity_threshold": 0.7,
    "test_size": 0.2,
    "val_size": 0.1,
    "protein_weight": 0.6,
    "ligand_weight": 0.4,
    "random_state": 42,
    "save_splits": true,
    "splits_filename": "stratified_splits.npz"
  },
  "classification": {
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 50,
    "early_stopping_patience": 5
  },
  "regression": {
    "models_to_train": ["catboost", "xgboost", "random_forest"],
    "cv_folds": 5
  }
}
```

## 🚀 Implementation Sequence

### Phase 1: Foundation (Day 1)
- [ ] Create `SplitIndices` data class
- [ ] Write unit tests for `SplitIndices`
- [ ] Ensure all tests pass

### Phase 2: Manager (Day 2)
- [ ] Create `StratificationManager`
- [ ] Write unit tests for `StratificationManager`
- [ ] Test integration with existing `Stratifier`

### Phase 3: BuildPipeline Integration (Day 3)
- [ ] Modify `BuildPipeline` to use `StratificationManager`
- [ ] Add methods to pass splits to pipelines
- [ ] Write integration tests

### Phase 4: ClassificationPipeline Update (Day 4 Morning)
- [ ] Add `split_indices` parameter
- [ ] Modify `load_data()` logic
- [ ] Test backward compatibility

### Phase 5: RegressionPipeline Update (Day 4 Afternoon)
- [ ] Add `split_indices` parameter
- [ ] Modify `load_data()` logic
- [ ] Test backward compatibility

### Phase 6: Integration Testing (Day 5)
- [ ] Write end-to-end integration tests
- [ ] Verify splits are shared correctly
- [ ] Test reproducibility
- [ ] Performance benchmarking

### Phase 7: Documentation (Day 5 Afternoon)
- [ ] Update BuildPipeline documentation
- [ ] Update ClassificationPipeline documentation
- [ ] Update RegressionPipeline documentation
- [ ] Create usage examples
- [ ] Update main README

## ✅ Success Criteria

1. **Single Stratification**: ✅ Stratification happens exactly once
2. **Shared Splits**: ✅ Classification and regression use IDENTICAL indices
3. **No Data Leakage**: ✅ No overlap between train/val/test
4. **Reproducibility**: ✅ Can save and reload exact splits
5. **Backward Compatible**: ✅ Existing code continues to work
6. **Performance**: ✅ No significant slowdown
7. **Clean Code**: ✅ SOLID principles followed
8. **Well Tested**: ✅ All tests passing (unit + integration + e2e)
9. **Documented**: ✅ Clear documentation in English

## 📝 Key Decisions Record

### Decision 1: Where to perform stratification?
**Chosen**: BuildPipeline  
**Rationale**: Has access to all embeddings and labels, can coordinate both pipelines

### Decision 2: How to pass splits?
**Chosen**: SplitIndices object as optional parameter  
**Rationale**: Explicit, type-safe, maintains backward compatibility

### Decision 3: When to stratify?
**Chosen**: After embeddings and labels are generated, before running pipelines  
**Rationale**: Need embeddings for similarity-based clustering

### Decision 4: How to maintain backward compatibility?
**Chosen**: Optional `split_indices` parameter  
**Rationale**: Simplest approach, minimal code changes

### Decision 5: What happens if stratification fails?
**Chosen**: Fallback to random splitting with warning  
**Rationale**: Pipeline should not break, but user should be notified

## ⚠️ Risks and Mitigations

### Risk 1: Breaking existing pipelines
**Likelihood**: Medium  
**Impact**: High  
**Mitigation**: 
- Make all changes optional (backward compatible)
- Extensive testing before merging
- Feature flag to enable/disable

### Risk 2: Performance degradation
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**: 
- Stratification already optimized
- Cache splits (compute once)
- Benchmark before/after

### Risk 3: Complexity increase
**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**: 
- Follow SOLID/KISS principles
- Clear documentation
- Simple, focused interfaces

### Risk 4: Integration bugs
**Likelihood**: Medium  
**Impact**: High  
**Mitigation**: 
- TDD approach (tests first)
- Integration tests at each step
- Continuous testing during development

---

**Status**: 📋 Analysis Complete  
**Next Step**: Begin Phase 1 - Create SplitIndices  
**Last Updated**: 2024-11-12
