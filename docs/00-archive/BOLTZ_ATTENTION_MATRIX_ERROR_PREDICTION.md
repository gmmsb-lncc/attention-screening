# Boltz + Attention Matrix Integration: Error Prediction & Solutions

## Executive Summary

**Current Status:** Boltz embeddings work, but the attention matrix pipeline is NOT properly configured to use Boltz results. The script conditionally SKIPS matrix generation for Boltz, which will cause cascading failures.

**Predicted Error Chain:**
1. ✅ Boltz generates embeddings successfully (vectors saved as `*_embedding.npy`)
2. ❌ Attention matrix CLI runs WITHOUT `--generate-matrices` (conditional disabled it)
3. ❌ Data loader finds NO `*_matrix.npy` files → falls back to vectors
4. ❌ Vectors reshaped to `[1, dim]` instead of `[seq_len, dim]`
5. ❌ Cross-attention layer processes wrong shapes → training unstable or crashes

---

## Error #1: Matrix Files Not Found (Expected)

**When:** Data loader initializes after attention matrix training requested

**Log Message:**
```
WARNING: No matrix files found, falling back to embeddings
```

**Root Cause:**
- Script line: `if [[ $MODEL != boltz* ]]; then ATTENTION_CMD+=" --generate-matrices"; fi`
- Effect: Boltz skips matrix generation during attention_matrix CLI call
- Result: `protein_matrices/` contains NO `*_matrix.npy` files

**File:** `src/attention_matrix/data_loader.py:218-226`

**Code:**
```python
if self.embedding_mode == 'matrix':
    matrix_files = list(ligand_dir.glob('*_matrix.npy'))
    if matrix_files:
        # ... load matrix files
    else:
        logger.warning("No matrix files found, falling back to embeddings")
        # Falls back to vector embeddings reshaped to [1, dim]
```

**Impact:** MEDIUM - Not fatal, but degrades model training quality

---

## Error #2: Shape Mismatch in Cross-Attention (CRITICAL)

**When:** First training batch is processed

**Expected Error:**
```python
RuntimeError: size mismatch: (batch, 1, hidden_dim) vs (batch, N_tokens, hidden_dim) in MultiheadAttention
```

**Root Cause:**

1. **For ESM Models (works fine):**
   - Matrix shape: `[seq_len, protein_dim]` (e.g., `[250, 320]`)
   - Dataset pads to: `[256, 320]`
   - Cross-attention expects: `(batch, seq_len, hidden_dim)` ✓

2. **For Boltz (breaks):**
   - Embedding vector shape: `[768]` (mean-pooled single representation)
   - Data loader fallback reshapes to: `[1, 768]` (as 2D)
   - Dataset pads to: `[64, 768]` with zeros
   - **Result:** Most samples have only 1 real token + 63 padding tokens!

**File:** `src/attention_matrix/dataset.py:88-100`

**Code:**
```python
def _load_protein(self, seq_id: str) -> np.ndarray:
    path = self.protein_dir / f"{seq_id}.npy"
    embedding = np.load(path)  # For Boltz: shape (1, 768)
    
    # Pad to fixed length
    padded = np.zeros((self.max_protein_len, self.protein_dim), dtype=np.float32)
    padded[:len(embedding)] = embedding  # Only first token filled!
    return padded
```

**Impact:** CRITICAL - Model training will be completely broken

---

## Error #3: Attention Weights Extraction Failures

**When:** Pipeline tries to extract and save attention maps

**Expected Error:**
```
IndexError: index out of bounds: dimension 2 has size 1 but index 256 is out of bounds
```

**Root Cause:**

File: `src/attention_matrix/pipeline.py:extract_and_save_attention_maps()`

The extraction code assumes per-token attention exists:
```python
for batch_idx, sample_id in enumerate(test_loader.dataset.sample_ids):
    attention = attention_output[batch_idx]  # Shape: (num_heads, seq_len_q, seq_len_k)
    # Expects shape like (4, 256, 64) for protein-ligand attention
    # But gets (4, 1, 64) when using Boltz vectors!
```

**Impact:** HIGH - Attention analysis will crash, no visualizations generated

---

## Error #4: Empty Results Directory

**When:** Pipeline completes but no meaningful outputs generated

**Log Message:**
```
No valid samples found!
...
FileNotFoundError: No attention_analysis.json generated
```

**Root Cause:**

If training somehow completes despite shape mismatches:
- Few/no attention maps extracted (due to Error #3)
- Output directories remain empty or incomplete
- Post-processing steps fail silently

**Impact:** MEDIUM - Silent failure, wasted computation

---

## Solution #1: Enable Matrix Generation for Boltz (RECOMMENDED)

**Fix the conditional logic in shell script:**

**File:** `scripts/run_all_protein_models.sh`

**Current (BROKEN):**
```bash
if [[ $MODEL != boltz* ]]; then
    ATTENTION_CMD+=" --generate-matrices"
fi
```

**Fixed:**
```bash
# ALL models should generate matrices - ESM via attention_matrix, Boltz via own strategy
ATTENTION_CMD+=" --generate-matrices"
```

**Why this works:**
1. ESM2/ESM-C: `attention_matrix --generate-matrices` generates matrices automatically
2. Boltz: `attention_matrix --generate-matrices` triggers Boltz CLI to extract matrices
3. Result: All models produce proper `*_matrix.npy` files

**Risk:** Boltz matrix generation adds time (10-30 sec/sequence), but ensures correct shapes

**Testing:**
```bash
# After fix, check matrix files exist:
ls -lh results/kinase_non_human_multi_model/boltz*/protein_matrices/*_matrix.npy | head -5
# Should show .npy files, not .npz or missing
```

---

## Solution #2: Alternative - Skip Boltz for Attention Matrix (NOT RECOMMENDED)

**If Boltz matrix generation is too slow/memory-intensive:**

**Option A: Skip Boltz from attention matrix pipeline entirely**

**File:** `scripts/run_all_protein_models.sh`

```bash
if [[ $RUN_ATTENTION_MATRIX == true ]]; then
    # Skip Boltz models for attention matrix training
    if [[ $MODEL != boltz* ]]; then
        echo "Running attention matrix for $MODEL..."
        python src/attention_matrix/pipeline.py ...
    else
        echo "⏭️  Skipping attention matrix for Boltz (not supported yet)"
    fi
fi
```

**Pros:** Avoids performance overhead
**Cons:** No attention visualization for Boltz, less insight into predictions

---

## Solution #3: Hybrid - Use PAE as Pseudo-Attention (EXPERIMENTAL)

**Rationale:** Boltz outputs Predicted Aligned Error (PAE) matrix during inference - can use as pseudo-attention

**Implementation outline:**

1. Modify `boltz_strategy.py` to extract PAE (already computed internally)
2. Save as `*_pae.npy` files
3. Update `data_loader.py` to use PAE as fallback
4. Normalize PAE: inverse relationship (low PAE = high confidence/attention)

**Code (conceptual):**
```python
# In boltz_strategy.py _extract_embedding_matrix():
# Extract PAE and save as pseudo-attention
pae = data['predicted_aligned_error']  # [N_tokens, N_tokens]
# Normalize: 1 - (pae / max_pae) to get attention-like weights
pae_attention = 1.0 - (pae / np.max(pae))
np.save(matrix_file.replace('_matrix', '_pae'), pae_attention)
```

**Pros:** Uses Boltz's native confidence estimates
**Cons:** Requires Boltz output modification, PAE != token attention

---

## Immediate Action Plan

### Step 1: Verify Boltz Output Format ✓ (DONE)
- Confirmed: Boltz saves as `.npz` with embedded numpy arrays
- Confirmed: `BoltzStrategy.generate_matrix()` supports matrix extraction

### Step 2: Remove Problematic Conditional ✓ (DONE)
The conditional has been already removed from `scripts/run_all_protein_models.sh`.
All models now include `--generate-matrices` flag (line 358).

### Step 3: Test with Small Boltz Dataset (TODO)
```bash
# Run small test
python scripts/run_all_protein_models.sh
# Select Option 6 (attention matrix)
# Select model: boltz_seed_0

# Verify matrices generated:
ls results/kinase_non_human_multi_model/boltz_seed_0/protein_matrices/ | head -10
```

### Step 4: Monitor Training (TODO)
- Watch for shape warnings/errors
- Verify attention maps extract successfully
- Check output file sizes (should be > 0)

---

## Debugging Commands

**Check which embeddings were generated:**
```bash
# Vectors (always exist)
ls -1 results/kinase_non_human_multi_model/boltz_seed_0/embeddings/ | wc -l

# Matrices (should exist if --generate-matrices used)
ls -1 results/kinase_non_human_multi_model/boltz_seed_0/protein_matrices/ | wc -l
```

**Check matrix dimensions:**
```bash
python3 << 'EOF'
import numpy as np
import os

matrix_dir = "results/kinase_non_human_multi_model/boltz_seed_0/protein_matrices"
for f in os.listdir(matrix_dir)[:3]:
    path = os.path.join(matrix_dir, f)
    data = np.load(path)
    print(f"{f}: {data.shape}")
EOF
```

**Check data loader behavior:**
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '${HOME}/docktkinase')

from src.attention_matrix.data_loader import EmbeddingDataLoader
from pathlib import Path

loader = EmbeddingDataLoader(
    results_dir=Path("results/kinase_non_human_multi_model/boltz_seed_0"),
    embedding_mode='matrix'
)
protein_emb, ligand_emb = loader.load_embeddings_from_files()
print(f"Protein: {protein_emb.shape}, Ligand: {ligand_emb.shape}")
EOF
```

---

## Success Indicators

✅ **After fix applied:**
- ✓ Protein matrix files exist: `protein_matrices/*_matrix.npy` with shape `[seq_len, 768]`
- ✓ Data loader loads without "falling back to embeddings" warning
- ✓ Dataset returns shapes `[batch, 256, 768]` for protein, `[batch, 64, 768]` for ligand
- ✓ Cross-attention completes without shape mismatch errors
- ✓ Attention maps extract and save successfully
- ✓ Output JSON files generated with reasonable content

---

## References

- **Boltz Strategy:** `${HOME}/docktkinase/src/build/embeddings/strategies/boltz_strategy.py:461-531`
- **Data Loader:** `${HOME}/docktkinase/src/attention_matrix/data_loader.py:205-260`
- **Dataset:** `${HOME}/docktkinase/src/attention_matrix/dataset.py:88-100`
- **Cross-Attention Model:** `${HOME}/docktkinase/src/attention_matrix/model.py:95-115`
- **Pipeline Extraction:** `${HOME}/docktkinase/src/attention_matrix/pipeline.py:extract_and_save_attention_maps()`


---

## WORKFLOW VERIFICATION ✓ (VERIFIED - FIX IS CORRECT)

### Component Chain Walkthrough

The complete Boltz→Attention Matrix pipeline is properly wired:

```
Script Shell (Option 6 with boltz2)
    └─> --esm-model boltz2
        └─> attention_matrix.py
            └─> Line 361: generate_protein_matrices(df, build_dir, "boltz2", device)
                └─> Line 219: ProteinEmbedding(model_name="boltz2")
                    └─> Line 109: factory.create_strategy("boltz2")
                        └─> ProteinModelFactory.create_strategy(line 116-117)
                            └─> "boltz2" in BOLTZ_MODELS? YES (line 64)
                                └─> return BoltzStrategy()
                                    └─> BoltzStrategy.generate_matrix(sequence)
                                        └─> Returns [seq_len, 768] matrix
                                            └─> Line 224: np.save({seq_id}_matrix.npy)
                                                └─> Data loader finds protein_matrices/*_matrix.npy ✓
                                                    └─> Cross-attention training ✓
```

### Verification Points

**1. Shell Script** ✓
- **File:** `scripts/run_all_protein_models.sh:358`
- **Status:** No conditional to skip Boltz
- **Code:** `ATTENTION_CMD+=" --generate-matrices"` (applies to ALL models)

**2. Factory Registration** ✓
- **File:** `src/build/embeddings/factories/protein_model_factory.py:64`
- **Status:** `boltz2` registered in `BOLTZ_MODELS`
- **Code:** `BOLTZ_MODELS = {'boltz2'}`

**3. Strategy Creation** ✓
- **File:** `src/build/embeddings/factories/protein_model_factory.py:116-117`
- **Status:** Boltz models detected and return `BoltzStrategy()`
- **Code:** `if model_name in ProteinModelFactory.BOLTZ_MODELS: return BoltzStrategy()`

**4. Matrix Generation** ✓
- **File:** `src/build/embeddings/strategies/boltz_strategy.py:461-531`
- **Status:** `generate_matrix()` returns full matrix, not pooled vector
- **Returns:** `[seq_len, 768]` shape

**5. Saving** ✓
- **File:** `attention_matrix.py:224`
- **Status:** Saves as `.npy` files with `{seq_id}_matrix.npy` pattern
- **Code:** `np.save(output_file, matrix)`

**CONCLUSION:** The fix (removing the conditional) is CORRECT and the pipeline is properly configured.

---

## Remaining Risks & Mitigations

### Risk 1: Boltz CLI Execution Failure

**When:** During matrix generation, `BoltzStrategy._run_boltz_cli()` executes

**Symptoms:**
- `RuntimeError: Boltz CLI execution failed`
- Missing `boltz_results_input/` directory
- `.npz` files not found in output

**Mitigation:**
- Ensure Boltz-2 is installed: `pip install boltz[cuda]`
- Check Boltz availability: `python -c "from boltz.run import main; print('OK')"`
- Monitor stderr for Boltz errors

### Risk 2: Memory Issues with Large Sequences

**When:** Boltz processes long sequences during matrix extraction

**Symptoms:**
- `torch.cuda.OutOfMemoryError` or `MemoryError`
- Process killed by OOM handler
- Partial matrix generation

**Mitigation:**
- Run on machine with 16+ GB VRAM (recommended: 24+ GB for Boltz)
- Use `--device cpu` for memory-constrained systems (slower but safe)
- Process in smaller batches if needed

### Risk 3: PAE Matrix Extraction (Future)

**Status:** Not yet implemented, but Boltz internally computes PAE

**Current:** Boltz only exposes mean-pooled `s` vectors (768-dim) via CLI

**Future Enhancement:**
- Modify `boltz_strategy.py` to extract PAE as pseudo-attention
- PAE shape: `[N_tokens, N_tokens]` - structural confidence matrix
- Could provide alternative attention visualization for Boltz

**Not Blocking:** Matrix mode uses per-token embeddings, not attention

---

## Test Checklist (Before Running)

- [ ] Boltz-2 installed: `pip list | grep boltz`
- [ ] GPU available (or CPU acceptable): `nvidia-smi` or skip for CPU
- [ ] Input TSV file exists and is valid
- [ ] Output directory writable
- [ ] Sufficient disk space (~5-10 GB recommended for full pipeline)
- [ ] Python environment activated with correct packages

## Test Checklist (After Running)

After Option 6 completes, verify:

```bash
# 1. Matrix files generated
ls -1 results/kinase_non_human_multi_model/boltz2/protein_matrices/*_matrix.npy | wc -l
# Should show > 0

# 2. Matrix dimensions correct
python3 << 'PYEOF'
import numpy as np
import os
matrix_dir = "results/kinase_non_human_multi_model/boltz2/protein_matrices"
for f in list(os.listdir(matrix_dir))[:3]:
    data = np.load(os.path.join(matrix_dir, f))
    print(f"{f}: {data.shape}")  # Should be (seq_len, 768), not (768,)
PYEOF

# 3. Data loader reads without fallback
python3 << 'PYEOF'
import sys
sys.path.insert(0, '${HOME}/docktkinase')
from src.attention_matrix.data_loader import EmbeddingDataLoader
from pathlib import Path
loader = EmbeddingDataLoader(
    results_dir=Path("results/kinase_non_human_multi_model/boltz2"),
    embedding_mode='matrix'
)
loader.load_dataset()
p_emb, l_emb = loader.load_embeddings_from_files()
print(f"Protein: {p_emb.shape}, Ligand: {l_emb.shape}")
# Should NOT print "falling back to embeddings" warning
PYEOF

# 4. Attention output generated
ls -lh results/kinase_non_human_multi_model/boltz2/attention_matrix/
# Should contain: attention_analysis.json, attention_summary.json, raw_matrices/
```

