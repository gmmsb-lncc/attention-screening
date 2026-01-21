# Troubleshooting Guide

## NaN or Inf Values During Training

### Problem
You may see errors like:
```
ValueError: Model produced NaN or Inf values during evaluation!
ValueError: Loss became NaN or Inf during training!
ValueError: NaN or Inf detected in parameter 'xxx' after optimizer step!
```

### What This Means
**NaN (Not a Number) or Inf (Infinity) values indicate severe training instability.**

This is **NOT** a bug to hide - it means the model training has failed and the results would be completely invalid if we continued.

### Common Causes & Solutions

#### 1. Exploding Gradients ⚡
**Symptom**: Loss suddenly becomes NaN after a few epochs

**Solution**:
```python
# Increase gradient clipping (in config.py)
max_grad_norm: float = 0.5  # Default is 1.0, try lower values
```

#### 2. Learning Rate Too High 📈
**Symptom**: NaN happens in first few epochs

**Solution**:
```python
# Reduce learning rate (in config.py)
learning_rate: float = 1e-5  # Default is 1e-4, try 10x lower
```

#### 3. Data Issues 📊
**Symptom**: NaN appears inconsistently

**Check**:
- Are there NaN or Inf values in your input data?
- Are embeddings normalized properly?
- Check for extremely large or small values in matrices

**Verify**:
```python
import numpy as np
import torch

# Check your data
protein_matrix = ... # your data
print(f"Min: {protein_matrix.min()}, Max: {protein_matrix.max()}")
print(f"NaN count: {torch.isnan(protein_matrix).sum()}")
print(f"Inf count: {torch.isinf(protein_matrix).sum()}")
```

#### 4. Model Architecture Issues 🏗️
**Symptom**: NaN happens consistently with specific encoder type

**Possible issues**:
- Division by zero in pooling operations
- Missing batch normalization
- Unstable activation functions

**Check**: `encoder_compare/models/flexible_model.py`
- Ensure `.clamp(min=1)` in pooling operations
- Verify LayerNorm is applied correctly

#### 5. Numerical Precision 🔢
**Symptom**: NaN happens with very deep models or long sequences

**Solutions**:
- Use mixed precision training
- Add batch normalization
- Reduce model depth

### How We Handle NaN

**Our Approach**: **Fail Fast, Fail Clearly**

We **DO NOT** replace NaN with dummy values (like 0.5) because:
- ❌ Would give invalid results
- ❌ Would hide the real problem
- ❌ Would waste computational resources
- ❌ Would make debugging impossible

Instead, we:
- ✅ Detect NaN early (during training)
- ✅ Raise clear error with diagnostics
- ✅ Provide actionable solutions
- ✅ Preserve valid checkpoint before failure

### Quick Fix Checklist

Try these in order:

1. **Reduce Learning Rate**
   ```bash
   # Edit encoder_compare/config.py
   learning_rate: float = 1e-5  # was 1e-4
   ```

2. **Increase Gradient Clipping**
   ```bash
   # Edit encoder_compare/config.py
   max_grad_norm: float = 0.5  # was 1.0
   ```

3. **Check Your Data**
   ```bash
   # Verify no NaN in input embeddings
   python -c "import torch; data = torch.load('your_embedding.pt'); print(torch.isnan(data).any())"
   ```

4. **Try Different Encoder**
   ```bash
   # If CNN_ATTENTION fails, try CNN or LINEAR
   # They have different numerical stability characteristics
   ```

5. **Use Smaller Batch Size**
   ```bash
   # Edit encoder_compare/config.py
   batch_size: int = 16  # was 32
   ```

### Still Having Issues?

If NaN persists after trying the above:

1. Check which encoder type is failing
2. Check at which epoch it fails
3. Examine the loss values just before NaN
4. Look at gradient norms during training

Add this to `trainer.py` temporarily for debugging:
```python
# After loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Gradient norm: {grad_norm:.4f}")
```

### Remember

**NaN is a symptom, not the disease.**

Finding and fixing the root cause will lead to:
- ✅ Stable training
- ✅ Valid results
- ✅ Reproducible experiments
- ✅ Better model performance

Don't mask the problem - fix it! 🔧
