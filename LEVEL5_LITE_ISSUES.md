# Level 5-Lite - Performance Issues Analysis

## Problem

After 2 epochs:
- **MCC: 0.3251** (vs Level 1 baseline: 0.428) → **24% worse**
- **AUC: 0.7300** (vs Level 1 baseline: 0.792) → **8% worse**

**The model is performing significantly worse than the simple Level 1 baseline.**

---

## Root Causes

### 1. **Architecture Mismatch** 🚨 CRITICAL

The current architecture has a **fundamental design flaw**:

```
Current (WRONG):
Per-residue ESM-2 [L, 320] → Transformer → Cross-Attention → Pool

Problem: ESM-2 already contains ALL biological information!
- ESM-2 was pre-trained on 250M sequences
- It already knows evolutionary patterns, structure, function
- Adding Transformer encoder AFTER ESM-2 is redundant
- We're asking the model to "re-learn" what ESM-2 already knows
```

**The Transformer encoder should NOT be used with ESM-2 per-residue embeddings!**

### 2. **Model Complexity vs. Training Time**

- **22.2M parameters** (6.3x larger than Level 1)
- **Only 2 epochs** (not enough for convergence)
- Complex models need 50-100+ epochs

### 3. **Information Bottleneck**

```python
# Current flow:
ESM-2 [L, 320] → Project to [L, 512] → Self-attention → Cross-attention → Pool to [512]
                   ↑ EXPANSION              ↑ REDUNDANT
```

**Issue**: We expand dimensions (320→512) then immediately apply self-attention, which is redundant because ESM-2 already has self-attention in it!

---

## Why Level 1 (FP+MLP) Works Better

```python
Level 1: Morgan FP [2048] + Mean(Ligand) [768] → MLP → Prediction
         ↑ Chemical structure  ↑ Averaged       ↑ Direct learning
```

**Advantages**:
1. **Simple**: 3.5M params vs 22M params
2. **Direct**: No redundant transformations
3. **Fast**: Converges in 10-20 epochs
4. **Stable**: No attention mechanisms to tune

---

## Recommended Fixes

### Option A: **Simplify Architecture (RECOMMENDED)**

```python
# Remove redundant Transformer encoders
ESM-2 [L, 320] → Project [L, 512] → Cross-Attention → Pool → Classifier
MoLFormer [T, 768] → Project [T, 512] ↗

# This reduces params from 22M to ~8M
# More importantly: removes redundancy
```

### Option B: **Use Mean-Pooled Vectors Instead**

```python
# Don't use per-residue embeddings with Transformer!
Mean(ESM-2) [320] → MLP [512] ↘
                                 Concat → Classifier
Mean(MoLFormer) [768] → MLP [512] ↗

# This is essentially Level 2, which should work better
```

### Option C: **Increase Training Budget**

- Train for **100-200 epochs** (not just 2)
- Reduce dropout to 0.1 everywhere
- Use smaller hidden_dim (256 instead of 512)

---

## Architectural Principles (Violated)

1. ❌ **Don't add self-attention after pre-trained attention models** (ESM-2, MoLFormer)
2. ❌ **Don't use per-token embeddings if you'll pool anyway** (defeats the purpose)
3. ❌ **Don't expand dimensions unnecessarily** (320→512 wastes capacity)
4. ✅ **DO use cross-attention between modalities** (this is the only novel part)

---

## Correct Level 5-Lite Architecture

```python
class Level5LiteFixed(nn.Module):
    def __init__(self):
        # NO Transformer encoders!
        self.protein_proj = nn.Linear(320, 512)
        self.ligand_proj = nn.Linear(768, 512)
        
        # Cross-attention is the ONLY learnable component
        self.cross_attn = BidirectionalCrossAttention(512, num_layers=2)
        
        # Attention pooling (good - this learns importance)
        self.protein_pool = AttentionPooling(512)
        self.ligand_pool = AttentionPooling(512)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Lower dropout
            nn.Linear(256, 1),
        )
    
    def forward(self, protein, ligand, protein_mask, ligand_mask):
        # Simple projection (NO self-attention)
        p = self.protein_proj(protein)
        l = self.ligand_proj(ligand)
        
        # Cross-attention (this is where learning happens)
        p, l = self.cross_attn(p, l, protein_mask, ligand_mask)
        
        # Pool and classify
        p_vec = self.protein_pool(p, protein_mask)
        l_vec = self.ligand_pool(l, ligand_mask)
        return self.classifier(torch.cat([p_vec, l_vec], -1))
```

**Expected parameters**: ~8M (vs current 22M)  
**Expected training time**: 20-50 epochs to converge  
**Expected MCC**: 0.45-0.52 (better than Level 1)

---

## Decision Tree

```
Is MCC < 0.40 after 10 epochs?
├─ YES → Architecture is fundamentally wrong
│         → Implement Level5LiteFixed (remove Transformer encoders)
│
└─ NO (MCC 0.40-0.45)
    ├─ Is MCC improving?
    │  ├─ YES → Train for 100 epochs, should reach 0.48-0.52
    │  └─ NO → Overfitting, reduce model size
    │
    └─ Is MCC > 0.45?
        └─ YES → Success! Continue training to 200 epochs
```

---

## Immediate Action

**RECOMMENDATION**: Redesign Level 5-Lite to remove Transformer encoders.

The current architecture violates the principle of not adding redundant transformations after pre-trained models. ESM-2 and MoLFormer already contain attention mechanisms - adding more is counterproductive.
