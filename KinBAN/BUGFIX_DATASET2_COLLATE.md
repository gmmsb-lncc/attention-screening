# Bug: `ValueError: too many values to unpack (expected 5)` after training

**Affected code**: `GraphBAN/run_baseline.py` · `ban-kinase-network/bkn/training.py`  
**Commit fix**: `6735e35`  
**Severity**: Crash — training completes but post-training evaluation fails on every run

---

## Traceback

```
File "run_baseline.py", line 596, in train_single_seed
    train_y_true, train_y_prob = _collect_predictions(...)
File "run_baseline.py", line 311, in _collect_predictions
    for batch in data_loader:
  ...
  File "utils.py", line 52, in graph_collate_func
    d, smile, p, esm, y = zip(*x)
ValueError: too many values to unpack (expected 5)
```

---

## Root Cause

GraphBAN uses **two different Dataset classes** with different output shapes:

| Class | Tuple returned per sample | Used for |
|-------|--------------------------|----------|
| `DTIDataset` | `(drug_graph, smiles_emb, protein_emb, esm_emb, label)` — **5 values** | val / test |
| `DTIDataset2` | `(drug_graph, smiles_emb, protein_emb, esm_emb, label, teacher_emb)` — **6 values** | train (knows teacher) |

And **two different collate functions**:

| Function | Expects | Used for |
|----------|---------|----------|
| `graph_collate_func` | 5-tuple | val / test loaders |
| `graph_collate_func2` | 6-tuple | train loader |

After training ends, `_collect_predictions` is called on the **train set** to measure
train-set metrics (overfit diagnosis). The bug was that this used the original
`train_dataset` — which is a `DTIDataset2` — paired with `params_eval`, which
specifies `graph_collate_func` (5-tuple). The mismatch causes the crash.

```python
# WRONG — DTIDataset2 yields 6 values, graph_collate_func expects 5
train_eval_generator = DataLoader(train_dataset, **params_eval)
```

The crash only surfaces **after** all epochs finish (post-training evaluation phase),
so the model trains correctly but no metrics are ever saved.

---

## Fix

Create a fresh `DTIDataset` (not `DTIDataset2`) for train-set evaluation. It reads
the same `train_df_seed` DataFrame but ignores the `teacher_emb` column, yielding
the 5-tuple that `graph_collate_func` expects.

```python
# CORRECT — DTIDataset yields 5 values, compatible with graph_collate_func
train_eval_dataset = modules["DTIDataset"](train_df_seed.index.values, train_df_seed)
train_eval_generator = DataLoader(train_eval_dataset, **params_eval)
```

---

## Impact in the Original GraphBAN Repository

This bug is present in the **original upstream GraphBAN codebase**
(`HamidHadipour/GraphBAN`). Any use of domain adaptation (`DA.USE = True`,
which is the default) triggers it because domain adaptation requires `DTIDataset2`
for the training loader, making the mismatch inevitable.

> If you run GraphBAN baseline on your own data with DA enabled and see this crash
> after the last epoch, apply the same one-line fix above.
