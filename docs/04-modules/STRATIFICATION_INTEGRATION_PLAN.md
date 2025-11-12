# Stratification Integration Plan

## 📋 Overview

Integrate the stratified splitting system into the classification and regression pipelines, ensuring that **both pipelines use the same stratified splits** to maintain consistency and prevent data leakage.

## 🎯 Objectives

1. **Single Stratification**: Perform stratification once and share splits between classification and regression
2. **SOLID Principles**: Maintain modular, extensible, and testable architecture
3. **KISS Principle**: Keep implementation simple and straightforward
4. **Clean Code**: Follow best practices for readability and maintainability
5. **Backward Compatibility**: Ensure existing pipelines continue to work

## 🏗️ Architecture Analysis

### Current State

```
BuildPipeline
├── EmbeddingMatrix (generates combined embeddings)
├── ClassificationPipeline (uses embeddings)
│   └── train_test_split (random or manual)
└── RegressionPipeline (uses same embeddings)
    └── train_test_split (random or manual - DIFFERENT indices!)
```

**Problem**: Classification and regression use different random splits!

### Target State

```
BuildPipeline
├── EmbeddingMatrix (generates combined embeddings)
├── Stratifier (ONE stratification for both)
│   └── multi_view_stratified_split()
├── ClassificationPipeline (uses stratified splits)
│   └── uses train_idx, val_idx, test_idx
└── RegressionPipeline (uses SAME stratified splits)
    └── uses train_idx, val_idx, test_idx
```

**Solution**: Single stratification shared by both pipelines!

## 📊 Design Principles Application

### 1. Single Responsibility Principle (SRP)

**Before**:
- BuildPipeline: Handles embeddings + classification + regression + splitting

**After**:
- `BuildPipeline`: Orchestrates the overall process
- `StratificationManager`: Handles stratification logic
- `ClassificationPipeline`: Handles classification only
- `RegressionPipeline`: Handles regression only

### 2. Open/Closed Principle (OCP)

- New `StratificationManager` can be extended without modifying existing code
- Easy to add new splitting strategies (random, stratified, temporal, etc.)

### 3. Dependency Inversion Principle (DIP)

- Pipelines depend on abstract split indices, not concrete splitting implementation
- `StratificationManager` provides interface for getting splits

### 4. KISS Principle

- Simple interface: `get_splits()` returns `(train_idx, val_idx, test_idx)`
- No complex inheritance or coupling
- Clear separation of concerns

### 5. Clean Code

- Descriptive names: `StratificationManager`, `get_stratified_splits()`
- Small functions (<50 lines)
- Clear comments and docstrings
- Type hints for all public methods

## 🔄 Implementation Plan

### Phase 1: Create StratificationManager (SRP)

**Purpose**: Encapsulate all stratification logic in one place

**File**: `src/build/pipeline/stratification_manager.py`

**Interface**:
```python
class StratificationManager:
    """
    Manages stratified splitting for both classification and regression.
    
    Single Responsibility: Handle stratification logic
    """
    
    def __init__(
        self,
        config: BuildConfig,
        stratification_enabled: bool = True,
        clustering_algorithm: str = 'kmeans',
        similarity_threshold: float = 0.7
    )
    
    def stratify(
        self,
        protein_embeddings: np.ndarray,
        ligand_embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> SplitIndices
    
    def get_splits(self) -> SplitIndices
    
    def save_splits(self, output_dir: str) -> None
    
    def load_splits(self, output_dir: str) -> SplitIndices
```

**Key Features**:
- ✅ Caches splits (stratify once, use many times)
- ✅ Can save/load splits for reproducibility
- ✅ Fallback to random splitting if stratification fails
- ✅ Configuration via BuildConfig

### Phase 2: Create SplitIndices Data Class (Clean Code)

**Purpose**: Type-safe container for split indices

**File**: `src/build/pipeline/split_indices.py`

**Interface**:
```python
@dataclass
class SplitIndices:
    """
    Container for train/validation/test split indices.
    
    Immutable and type-safe.
    """
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate indices are integer type
        # Validate no overlap between splits
        # Validate all indices are unique
        pass
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export as dictionary for saving."""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'SplitIndices':
        """Load from dictionary."""
        pass
    
    def save(self, filepath: str) -> None:
        """Save splits to .npz file."""
        pass
    
    @classmethod
    def load(cls, filepath: str) -> 'SplitIndices':
        """Load splits from .npz file."""
        pass
```

### Phase 3: Update BuildPipeline (OCP, DIP)

**Purpose**: Integrate StratificationManager without breaking existing code

**File**: `src/build/pipeline/build_pipeline.py`

**Changes**:
```python
class BuildPipeline:
    def __init__(self, config: BuildConfig):
        self.config = config
        self.stratification_manager = StratificationManager(config)
        self.split_indices: Optional[SplitIndices] = None
    
    def run_complete_pipeline(self, ...):
        # 1. Generate embeddings (existing)
        protein_emb, ligand_emb, labels = self._generate_embeddings(...)
        
        # 2. Stratify ONCE (NEW)
        self.split_indices = self.stratification_manager.stratify(
            protein_emb, ligand_emb, labels
        )
        
        # 3. Save splits for reproducibility (NEW)
        self.stratification_manager.save_splits(output_dir)
        
        # 4. Run classification with stratified splits (UPDATED)
        classification_results = self._run_classification(
            protein_emb, ligand_emb, labels,
            split_indices=self.split_indices  # Pass splits
        )
        
        # 5. Run regression with SAME stratified splits (UPDATED)
        regression_results = self._run_regression(
            protein_emb, ligand_emb, labels,
            split_indices=self.split_indices  # Same splits!
        )
        
        return results
```

**Key Points**:
- ✅ Backward compatible (can disable stratification via config)
- ✅ Single stratification shared by both pipelines
- ✅ Splits are saved for reproducibility

### Phase 4: Update ClassificationPipeline (DIP)

**Purpose**: Use provided splits instead of creating own

**File**: `src/classifier/modular_pipeline.py`

**Changes**:
```python
class ClassificationPipeline:
    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split_indices: Optional[SplitIndices] = None,  # NEW parameter
        **kwargs
    ):
        # Use provided splits if available
        if split_indices is not None:
            train_idx = split_indices.train_idx
            val_idx = split_indices.val_idx
            test_idx = split_indices.test_idx
        else:
            # Fallback to existing random split logic
            train_idx, val_idx, test_idx = self._create_random_splits(...)
        
        # Rest of pipeline uses these indices
        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        # ... etc
```

**Key Points**:
- ✅ Optional parameter maintains backward compatibility
- ✅ Clear fallback to existing behavior
- ✅ No breaking changes to existing API

### Phase 5: Update RegressionPipeline (DIP)

**Purpose**: Use same splits as classification

**File**: `src/regression/modular_pipeline.py`

**Changes**:
```python
class RegressionPipeline:
    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        split_indices: Optional[SplitIndices] = None,  # NEW parameter
        **kwargs
    ):
        # Same logic as ClassificationPipeline
        if split_indices is not None:
            train_idx = split_indices.train_idx
            val_idx = split_indices.val_idx
            test_idx = split_indices.test_idx
        else:
            train_idx, val_idx, test_idx = self._create_random_splits(...)
        
        # Rest of pipeline uses these indices
        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        # ... etc
```

## 🧪 Testing Strategy

### Test 1: StratificationManager Unit Tests

**File**: `tests/test_stratification_manager.py`

**Tests**:
- ✅ Stratification produces valid splits
- ✅ No overlap between train/val/test
- ✅ All indices are covered
- ✅ Indices are integer type
- ✅ Can save and load splits
- ✅ Fallback to random splitting works

### Test 2: SplitIndices Unit Tests

**File**: `tests/test_split_indices.py`

**Tests**:
- ✅ Validation catches invalid indices
- ✅ Validation catches overlapping splits
- ✅ Can convert to/from dict
- ✅ Can save/load from file
- ✅ Immutable after creation

### Test 3: Integration Test

**File**: `tests/test_stratification_integration.py`

**Tests**:
- ✅ BuildPipeline performs stratification once
- ✅ Classification uses stratified splits
- ✅ Regression uses SAME stratified splits
- ✅ Splits are saved to output directory
- ✅ Can load and reuse saved splits
- ✅ Results are consistent across runs

### Test 4: Backward Compatibility Test

**File**: `tests/test_backward_compatibility.py`

**Tests**:
- ✅ Can disable stratification via config
- ✅ Old behavior preserved when disabled
- ✅ Existing code continues to work
- ✅ No breaking API changes

### Test 5: Performance Benchmark

**File**: `tests/test_stratification_performance.py`

**Tests**:
- ✅ Stratification time for different dataset sizes
- ✅ Memory usage comparison
- ✅ No significant performance degradation

## 📁 File Structure

```
src/build/
├── pipeline/
│   ├── build_pipeline.py (UPDATED)
│   ├── stratification_manager.py (NEW)
│   └── split_indices.py (NEW)
├── stratification/
│   └── (existing files, no changes needed)
│
src/classifier/
└── modular_pipeline.py (UPDATED)

src/regression/
└── modular_pipeline.py (UPDATED)

tests/
├── test_stratification_manager.py (NEW)
├── test_split_indices.py (NEW)
├── test_stratification_integration.py (NEW)
├── test_backward_compatibility.py (NEW)
└── test_stratification_performance.py (NEW)

docs/04-modules/
└── STRATIFICATION_INTEGRATION_PLAN.md (THIS FILE)
```

## 🔄 Implementation Sequence

### Step 1: Implement SplitIndices (1-2 hours)
- Create data class
- Add validation
- Add save/load methods
- Write unit tests

### Step 2: Implement StratificationManager (2-3 hours)
- Create manager class
- Add stratification logic
- Add save/load splits
- Add fallback to random
- Write unit tests

### Step 3: Update BuildPipeline (1-2 hours)
- Integrate StratificationManager
- Update pipeline flow
- Add configuration options
- Test integration

### Step 4: Update ClassificationPipeline (1 hour)
- Add split_indices parameter
- Update split logic
- Maintain backward compatibility
- Test changes

### Step 5: Update RegressionPipeline (1 hour)
- Add split_indices parameter
- Update split logic
- Maintain backward compatibility
- Test changes

### Step 6: Integration Testing (2-3 hours)
- Write integration tests
- Test end-to-end flow
- Verify splits are shared
- Test backward compatibility
- Performance benchmarking

### Step 7: Documentation (1-2 hours)
- Update BuildPipeline docs
- Update classification docs
- Update regression docs
- Add usage examples
- Update main README

**Total Estimated Time**: 9-14 hours

## 🎯 Success Criteria

1. ✅ **Single Stratification**: Both pipelines use same splits
2. ✅ **No Data Leakage**: Train/val/test have no overlap
3. ✅ **Reproducibility**: Can save/load splits
4. ✅ **Backward Compatible**: Old code still works
5. ✅ **Performance**: No significant slowdown
6. ✅ **Clean Code**: SOLID principles followed
7. ✅ **Well Tested**: All tests passing
8. ✅ **Documented**: Clear usage examples

## 🚀 Next Steps

1. **Create SplitIndices data class**
   - Start with tests (TDD approach)
   - Implement minimal viable version
   - Add validation and save/load

2. **Create StratificationManager**
   - Start with tests
   - Implement core stratification logic
   - Add caching and persistence

3. **Update pipelines incrementally**
   - One pipeline at a time
   - Test after each change
   - Maintain backward compatibility

4. **Integration testing**
   - Test end-to-end flow
   - Verify consistency
   - Benchmark performance

## 📝 Configuration Example

```python
# config.json
{
  "stratification_enabled": true,
  "stratification_params": {
    "clustering_algorithm": "kmeans",
    "similarity_threshold": 0.7,
    "test_size": 0.2,
    "val_size": 0.1,
    "protein_weight": 0.6,
    "ligand_weight": 0.4,
    "random_state": 42
  },
  "save_splits": true,
  "splits_filename": "stratified_splits.npz"
}
```

## 🔍 Key Decisions

### Decision 1: Where to perform stratification?

**Options**:
- A) In BuildPipeline ✅ **CHOSEN**
- B) In each pipeline separately
- C) As a separate preprocessing step

**Rationale**: BuildPipeline has access to embeddings and labels, and can coordinate both pipelines.

### Decision 2: How to share splits between pipelines?

**Options**:
- A) Pass SplitIndices object ✅ **CHOSEN**
- B) Save to disk and reload
- C) Use global state

**Rationale**: Explicit parameter passing is cleaner and more testable.

### Decision 3: How to maintain backward compatibility?

**Options**:
- A) Make split_indices optional parameter ✅ **CHOSEN**
- B) Create new pipeline classes
- C) Use feature flags

**Rationale**: Optional parameter is simplest and least disruptive.

## ⚠️ Risks and Mitigations

### Risk 1: Breaking existing pipelines
**Mitigation**: Make all changes optional, maintain backward compatibility, extensive testing

### Risk 2: Performance degradation
**Mitigation**: Benchmark before/after, optimize stratification, cache results

### Risk 3: Complexity increase
**Mitigation**: Follow SOLID/KISS, clear documentation, simple interfaces

### Risk 4: Hard to test
**Mitigation**: TDD approach, dependency injection, mock objects

## 📊 Expected Benefits

1. **Consistency**: Same splits for classification and regression
2. **No Data Leakage**: Proper stratification prevents information leakage
3. **Reproducibility**: Can save and reload exact splits
4. **Better Generalization**: Stratified splits better represent data distribution
5. **Maintainability**: Clean, modular architecture
6. **Extensibility**: Easy to add new splitting strategies

## 📅 Timeline

- **Day 1**: Implement SplitIndices + tests
- **Day 2**: Implement StratificationManager + tests
- **Day 3**: Update BuildPipeline + integration tests
- **Day 4**: Update Classification and Regression pipelines
- **Day 5**: Final testing, documentation, review

## 🎓 References

- SOLID Principles: https://en.wikipedia.org/wiki/SOLID
- Clean Code by Robert C. Martin
- Test-Driven Development (TDD)
- Stratification documentation: `docs/04-modules/STRATIFIER_REFACTORING.md`

---

**Status**: 📋 Planning Phase  
**Next Step**: Implement SplitIndices data class  
**Last Updated**: November 12, 2025
