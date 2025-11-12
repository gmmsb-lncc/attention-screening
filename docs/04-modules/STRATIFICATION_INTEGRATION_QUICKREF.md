# Stratification Integration - Quick Reference Card

## 🎯 Main Goal
**Stratify ONCE, use SAME splits for classification AND regression**

## 📁 Key Documents
1. `STRATIFICATION_INTEGRATION_SUMMARY.md` ← **START HERE!**
2. `STRATIFICATION_INTEGRATION_PLAN.md` ← Detailed design
3. `STRATIFICATION_INTEGRATION_ANALYSIS.md` ← Architecture analysis

## 🚨 Critical Problem
```
CURRENT (BAD):
ClassificationPipeline: train=[5,10,15...], test=[70,75...]
RegressionPipeline:     train=[2,8,12...], test=[85,90...]  ← DIFFERENT!
❌ Data leakage! Results invalid!

TARGET (GOOD):
Both pipelines:         train=[0,1,2...], test=[80,81...]  ← IDENTICAL!
✅ No data leakage! Valid comparison!
```

## 🏗️ Solution Architecture
```
BuildPipeline
├── Generate embeddings
├── STRATIFY ONCE ← NEW!
│   └── Returns: SplitIndices(train_idx, val_idx, test_idx)
├── Pass to ClassificationPipeline(split_indices=splits) ← NEW!
└── Pass to RegressionPipeline(split_indices=splits) ← NEW!
```

## 📝 Files to Create
1. `src/build/pipeline/split_indices.py` - Data class
2. `src/build/pipeline/stratification_manager.py` - Manager
3. `tests/test_stratification_integration.py` - Tests

## 📝 Files to Modify
1. `src/build/pipeline/build_pipeline.py` - Add stratification
2. `src/classifier/modular_pipeline.py` - Accept splits
3. `src/regression/modular_pipeline.py` - Accept splits

## 🔄 7 Implementation Phases

**Phase 1 (Day 1)**: Create SplitIndices data class
**Phase 2 (Day 2)**: Create StratificationManager
**Phase 3 (Day 3)**: Integrate into BuildPipeline
**Phase 4 (Day 4 AM)**: Update ClassificationPipeline
**Phase 5 (Day 4 PM)**: Update RegressionPipeline
**Phase 6 (Day 5 AM)**: Integration testing
**Phase 7 (Day 5 PM)**: Documentation

## 🧪 Critical Test
```python
def test_same_splits_for_both_pipelines():
    """MOST IMPORTANT TEST - Verify both pipelines use IDENTICAL splits"""
    pipeline = BuildPipeline(config)
    pipeline.run_complete_pipeline(...)
    
    clf_splits = classification_pipeline.split_indices
    reg_splits = regression_pipeline.split_indices
    
    # Must be IDENTICAL!
    assert np.array_equal(clf_splits.train_idx, reg_splits.train_idx)
    assert np.array_equal(clf_splits.val_idx, reg_splits.val_idx)
    assert np.array_equal(clf_splits.test_idx, reg_splits.test_idx)
```

## ✅ Success Criteria (9 items)
1. ✅ Single stratification
2. ✅ Shared splits
3. ✅ No data leakage
4. ✅ Reproducibility
5. ✅ Backward compatible
6. ✅ Performance (<5% overhead)
7. ✅ SOLID principles
8. ✅ All tests passing
9. ✅ English documentation

## 🚀 How to Start

### Step 1: Read Summary
```bash
cd /Users/sulfierry/docktkinase
cat docs/04-modules/STRATIFICATION_INTEGRATION_SUMMARY.md
```

### Step 2: Review Architecture
```bash
cat docs/04-modules/STRATIFICATION_INTEGRATION_ANALYSIS.md
```

### Step 3: Start Phase 1
```bash
# Create SplitIndices
touch src/build/pipeline/split_indices.py
touch tests/test_split_indices.py

# Write tests FIRST (TDD)
code tests/test_split_indices.py
```

## 💡 Key Principles

**SOLID**:
- Single Responsibility: Each component does ONE thing
- Open/Closed: Extend without modifying existing code
- Dependency Inversion: Depend on abstractions

**KISS**: Keep It Simple, Stupid
- Simple interfaces
- Clear naming
- Small functions (<50 lines)

**Clean Code**:
- Descriptive names
- Type hints
- Docstrings
- Unit tests

**TDD**: Test-Driven Development
- Write tests FIRST
- Implement to pass tests
- Refactor if needed

## ⚠️ Important Rules

1. ✅ **Test before implementing** - Always TDD
2. ✅ **One phase at a time** - Don't skip ahead
3. ✅ **Backward compatible** - Must not break existing code
4. ✅ **English only** - All documentation in English
5. ✅ **SOLID always** - Follow principles religiously
6. ✅ **Commit frequently** - Small, working commits

## 📊 Progress Tracker

```
[ ] Phase 1: SplitIndices
[ ] Phase 2: StratificationManager
[ ] Phase 3: BuildPipeline
[ ] Phase 4: ClassificationPipeline
[ ] Phase 5: RegressionPipeline
[ ] Phase 6: Integration Testing
[ ] Phase 7: Documentation
```

## 🎯 Timeline

- **Day 1**: SplitIndices complete
- **Day 2**: StratificationManager complete
- **Day 3**: BuildPipeline integration complete
- **Day 4**: Both pipelines updated
- **Day 5**: Testing and docs complete

**Total**: 14-18 hours over 5 days

## 📞 Quick Commands

```bash
# Run tests
pytest tests/test_split_indices.py -v
pytest tests/test_stratification_manager.py -v
pytest tests/test_stratification_integration.py -v

# Check code
flake8 src/build/pipeline/
mypy src/build/pipeline/

# Commit
git add .
git commit -m "feat: [phase description]"
git push origin stratifier
```

## 🏆 Final Goal

**ONE stratification → SAME splits → NO data leakage → VALID results!**

---

**Next Action**: Read STRATIFICATION_INTEGRATION_SUMMARY.md  
**Then**: Start Phase 1 - Create SplitIndices
