# Stratification Integration - Work Plan Summary

## 📋 Executive Summary

This document provides a high-level overview of the integration plan to ensure that **classification and regression pipelines use the same stratified splits**, eliminating data leakage and ensuring consistent evaluation.

## 🎯 Main Objective

**Stratify the dataset ONCE and use the SAME splits for both classification and regression.**

## 🚨 Problem Identified

### Current State (DATA LEAKAGE!)
```
BuildPipeline generates embeddings (samples 0-99)
   ↓
ClassificationPipeline creates random splits:
   train=[5,10,15,...], val=[30,35,...], test=[70,75,...]
   ↓
RegressionPipeline creates DIFFERENT random splits:
   train=[2,8,12,...], val=[40,45,...], test=[85,90,...]
```

**Issue**: Sample #5 might be in train for classification but test for regression!
- ❌ Results are NOT comparable
- ❌ Data leakage between tasks
- ❌ Invalid scientific conclusions

### Target State (SOLUTION!)
```
BuildPipeline:
├── Generate embeddings
├── STRATIFY ONCE: train=[0,1,2...], val=[50,51...], test=[80,81...]
├── Pass SAME splits to classification
└── Pass SAME splits to regression

Both pipelines use IDENTICAL indices!
```

✅ No data leakage  
✅ Fair comparison  
✅ Valid conclusions  

## 📁 Files to Create/Modify

### New Files (3)
1. `src/build/pipeline/split_indices.py` - Data class for split indices
2. `src/build/pipeline/stratification_manager.py` - Stratification orchestrator
3. `tests/test_stratification_integration.py` - Integration tests

### Files to Modify (3)
1. `src/build/pipeline/build_pipeline.py` - Integrate stratification
2. `src/classifier/modular_pipeline.py` - Accept external splits
3. `src/regression/modular_pipeline.py` - Accept external splits

## 🏗️ Architecture Overview

```python
# 1. SplitIndices (NEW) - Type-safe container
@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    
    def save(self, filepath: str)
    def load(filepath: str) -> SplitIndices

# 2. StratificationManager (NEW) - Orchestrator
class StratificationManager:
    def stratify(...) -> SplitIndices
    def save_splits(filepath: str)
    def load_splits(filepath: str) -> SplitIndices

# 3. BuildPipeline (MODIFY) - Integration point
class BuildPipeline:
    def run_complete_pipeline(...):
        # ... generate embeddings ...
        # ... generate labels ...
        
        # NEW: Stratify once
        splits = self.stratification_manager.stratify(...)
        
        # NEW: Pass to both pipelines
        self._run_classification(split_indices=splits)
        self._run_regression(split_indices=splits)

# 4. ClassificationPipeline (MODIFY) - Accept splits
class MLPEmbeddingPipeline:
    def __init__(self, ..., split_indices=None):  # NEW parameter
        self.split_indices = split_indices
    
    def load_data(self):
        if self.split_indices:  # NEW: Use provided splits
            X_train = X[self.split_indices.train_idx]
            # ...
        else:  # OLD: Backward compatible
            # ... existing split logic ...

# 5. RegressionPipeline (MODIFY) - Accept splits
class RegressionPipeline:
    def __init__(self, ..., split_indices=None):  # NEW parameter
        self.split_indices = split_indices
    
    def load_data(self):
        if self.split_indices:  # NEW: Use provided splits
            self.X_train = X[self.split_indices.train_idx]
            # ...
        else:  # OLD: Backward compatible
            # ... existing split logic ...
```

## 🔄 Implementation Phases

### Phase 1: Foundation (Day 1) - 2-3 hours
**Goal**: Create `SplitIndices` data class

**Tasks**:
- [ ] Create `src/build/pipeline/split_indices.py`
- [ ] Implement validation (no overlap, all covered)
- [ ] Implement save/load methods
- [ ] Write unit tests (`tests/test_split_indices.py`)
- [ ] ✅ All tests passing

**Deliverables**:
- Working `SplitIndices` class
- 100% test coverage

### Phase 2: Manager (Day 2) - 3-4 hours
**Goal**: Create `StratificationManager`

**Tasks**:
- [ ] Create `src/build/pipeline/stratification_manager.py`
- [ ] Wrap existing `Stratifier` module
- [ ] Implement caching (stratify once, reuse)
- [ ] Implement save/load splits
- [ ] Write unit tests (`tests/test_stratification_manager.py`)
- [ ] ✅ All tests passing

**Deliverables**:
- Working `StratificationManager` class
- Integration with existing stratification module

### Phase 3: BuildPipeline (Day 3) - 3-4 hours
**Goal**: Integrate stratification into BuildPipeline

**Tasks**:
- [ ] Modify `src/build/pipeline/build_pipeline.py`
- [ ] Add `StratificationManager` to components
- [ ] Modify `run_complete_pipeline()` to stratify once
- [ ] Add methods to pass splits to pipelines
- [ ] Write integration tests
- [ ] ✅ All tests passing

**Deliverables**:
- BuildPipeline performs stratification
- Splits are saved for reproducibility

### Phase 4: Classification (Day 4 Morning) - 2 hours
**Goal**: Update ClassificationPipeline to accept splits

**Tasks**:
- [ ] Modify `src/classifier/modular_pipeline.py`
- [ ] Add `split_indices` parameter to `__init__`
- [ ] Modify `load_data()` to use provided splits
- [ ] Test backward compatibility (works without splits)
- [ ] ✅ All tests passing

**Deliverables**:
- ClassificationPipeline accepts external splits
- Backward compatible (still works standalone)

### Phase 5: Regression (Day 4 Afternoon) - 2 hours
**Goal**: Update RegressionPipeline to accept splits

**Tasks**:
- [ ] Modify `src/regression/modular_pipeline.py`
- [ ] Add `split_indices` parameter to `__init__`
- [ ] Modify `load_data()` to use provided splits
- [ ] Test backward compatibility (works without splits)
- [ ] ✅ All tests passing

**Deliverables**:
- RegressionPipeline accepts external splits
- Backward compatible (still works standalone)

### Phase 6: Integration Testing (Day 5 Morning) - 3 hours
**Goal**: Verify end-to-end integration

**Tasks**:
- [ ] Write `tests/test_stratification_integration.py`
- [ ] Test: Stratification happens once
- [ ] Test: Classification uses provided splits
- [ ] Test: Regression uses SAME splits
- [ ] Test: No data leakage (no overlap)
- [ ] Test: Reproducibility (save/load)
- [ ] Performance benchmarking
- [ ] ✅ All tests passing

**Critical Test**:
```python
def test_same_splits_for_both_pipelines():
    """THE MOST IMPORTANT TEST!"""
    pipeline = BuildPipeline(config)
    pipeline.run_complete_pipeline(...)
    
    clf_splits = classification_pipeline.split_indices
    reg_splits = regression_pipeline.split_indices
    
    # Must be IDENTICAL!
    assert np.array_equal(clf_splits.train_idx, reg_splits.train_idx)
    assert np.array_equal(clf_splits.val_idx, reg_splits.val_idx)
    assert np.array_equal(clf_splits.test_idx, reg_splits.test_idx)
```

### Phase 7: Documentation (Day 5 Afternoon) - 2 hours
**Goal**: Complete English documentation

**Tasks**:
- [ ] Update BuildPipeline docstrings
- [ ] Update ClassificationPipeline docstrings
- [ ] Update RegressionPipeline docstrings
- [ ] Create usage examples
- [ ] Update main README
- [ ] Document configuration options

**Deliverables**:
- Complete API documentation
- Usage examples
- Migration guide

## ✅ Success Criteria

1. ✅ **Single Stratification**: Stratification happens exactly once
2. ✅ **Shared Splits**: Classification and regression use IDENTICAL indices
3. ✅ **No Data Leakage**: No overlap between train/val/test
4. ✅ **Reproducibility**: Can save and reload exact splits
5. ✅ **Backward Compatible**: Existing code continues to work
6. ✅ **Performance**: No significant slowdown (<5% overhead)
7. ✅ **Clean Code**: SOLID principles followed throughout
8. ✅ **Well Tested**: All tests passing (unit + integration + e2e)
9. ✅ **Documented**: Clear documentation in English

## 🧪 Testing Strategy Summary

### Unit Tests
- `test_split_indices.py` - Test data class validation
- `test_stratification_manager.py` - Test manager logic

### Integration Tests
- `test_stratification_integration.py` - Test pipeline integration
- Test splits are shared correctly
- Test no data leakage
- Test backward compatibility

### End-to-End Tests
- `test_complete_pipeline_with_stratification.py`
- Test full workflow
- Test reproducibility
- Performance benchmarking

## 📊 Configuration Example

```json
{
  "stratification_enabled": true,
  "stratification_params": {
    "clustering_algorithm": "kmeans",
    "test_size": 0.2,
    "val_size": 0.1,
    "protein_weight": 0.6,
    "ligand_weight": 0.4,
    "random_state": 42,
    "save_splits": true
  },
  "run_classification": true,
  "run_regression": true,
  "share_splits": true
}
```

## 🚀 Next Steps

### Immediate Actions (START NOW!)

1. **Review Planning Documents**
   - Read `STRATIFICATION_INTEGRATION_PLAN.md`
   - Read `STRATIFICATION_INTEGRATION_ANALYSIS.md`
   - Understand architecture and requirements

2. **Set Up Testing Environment**
   - Ensure all dependencies installed
   - Verify existing tests pass
   - Create `tests/test_stratification_integration.py` skeleton

3. **Begin Phase 1**
   - Create `src/build/pipeline/split_indices.py`
   - Write tests FIRST (TDD approach)
   - Implement minimal viable version
   - Iterate until all tests pass

4. **Progress Through Phases**
   - Complete each phase before moving to next
   - Run all tests after each change
   - Commit working code frequently

## 📝 Detailed Documentation

For complete details, see:

1. **STRATIFICATION_INTEGRATION_PLAN.md**
   - Complete architectural design
   - SOLID principles application
   - Detailed implementation specifications
   - Configuration examples

2. **STRATIFICATION_INTEGRATION_ANALYSIS.md**
   - Current architecture analysis
   - Problem identification (data leakage)
   - Required changes to each file
   - Critical test specifications

## ⚠️ Important Reminders

1. **Test Before Implementing**: Write tests first (TDD)
2. **One Phase at a Time**: Don't skip ahead
3. **Backward Compatibility**: Always maintain
4. **English Documentation**: All docs in English
5. **SOLID Principles**: Follow throughout
6. **Commit Frequently**: Working code only

## 📈 Progress Tracking

```
Phase 1: Foundation        [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 2: Manager          [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 3: BuildPipeline    [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 4: Classification   [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 5: Regression       [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 6: Testing          [ ] Not Started  [ ] In Progress  [ ] Complete
Phase 7: Documentation    [ ] Not Started  [ ] In Progress  [ ] Complete
```

## 🎯 Expected Timeline

- **Day 1**: Phase 1 complete (SplitIndices)
- **Day 2**: Phase 2 complete (StratificationManager)
- **Day 3**: Phase 3 complete (BuildPipeline integration)
- **Day 4**: Phases 4-5 complete (Both pipelines updated)
- **Day 5**: Phases 6-7 complete (Testing and documentation)

**Total Estimated Time**: 14-18 hours (5 working days)

## 🏆 Final Goal

**One stratification, shared by all pipelines, no data leakage, scientifically valid results!**

---

**Status**: 📋 Planning Complete - Ready to Start Implementation  
**Next Action**: Begin Phase 1 - Create SplitIndices  
**Last Updated**: 2024-11-12
