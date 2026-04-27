# LEVEL 6: Optimized Transformer-CrossAttention for MCC > 0.6

**Status**: Specification Phase  
**Goal**: Achieve validation MCC > 0.6 through systematic hyperparameter optimization  
**Baseline**: Level 5-Lite achieves MCC ~0.50 (Epoch 3: val_mcc=0.4986)

---

## 1. MOTIVATION

Level 5-Lite demonstrated that **Transformer + Cross-Attention architecture works**, achieving:
- **MCC 0.50** at epoch 3 (vs. Level 1 baseline MCC=0.428)
- **Rapid convergence** (improving from 0.42 → 0.50 in 3 epochs)
- **Stable training** (no collapse, smooth metrics)

However, we're still below our **target MCC > 0.6**. To bridge this gap, Level 6 focuses on **systematic optimization** of critical hyperparameters identified from ablation studies in similar architectures.

---

## 2. ARCHITECTURE (Unchanged from Level 5-Lite)

Level 6 keeps the proven Level 5-Lite architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                         │
│ • Protein: [batch, seq_len, 320] (ESM-2 8M per-token)       │
│ • Ligand:  [batch, mol_len, 768] (SMI-TED per-token)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRANSFORMER ENCODERS (Independent)                           │
│                                                               │
│ Protein Encoder:                                             │
│   TransformerEncoder(d_model=320, nhead=8, layers=2)        │
│   → [batch, seq_len, 320]                                    │
│                                                               │
│ Ligand Encoder:                                              │
│   TransformerEncoder(d_model=768, nhead=8, layers=2)        │
│   → [batch, mol_len, 768]                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ CROSS-ATTENTION (Bidirectional)                              │
│                                                               │
│ Protein attends to Ligand:                                   │
│   MultiheadAttention(Q=protein, K=ligand, V=ligand)         │
│   → [batch, seq_len, 320]                                    │
│                                                               │
│ Ligand attends to Protein:                                   │
│   MultiheadAttention(Q=ligand, K=protein, V=protein)        │
│   → [batch, mol_len, 768]                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ POOLING & FUSION                                             │
│                                                               │
│ • Max Pool protein_attn → [batch, 320]                      │
│ • Max Pool ligand_attn  → [batch, 768]                      │
│ • Concatenate          → [batch, 1088]                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ CLASSIFIER HEAD                                              │
│                                                               │
│ Linear(1088 → 512) → ReLU → Dropout(0.3)                    │
│ Linear(512 → 256)  → ReLU → Dropout(0.3)                    │
│ Linear(256 → 1)    → Sigmoid                                 │
└─────────────────────────────────────────────────────────────┘
```

**Architecture remains frozen** — Level 6 only optimizes hyperparameters.

---

## 3. OPTIMIZATION TARGETS

### 3.1 Priority 1: Learning Rate & Scheduler (Expected Δ MCC: +0.05 to +0.10)

**Current Configuration:**
```python
lr = 1e-4
scheduler = CosineAnnealingLR(T_max=50, eta_min=1e-6)
```

**Problem**: 
- Fixed LR may be too conservative for initial epochs
- Cosine decay starts immediately, no warmup phase
- Cross-attention may need different LR than encoders

**Proposed Changes:**

#### Option A: Linear Warmup + Cosine Decay (RECOMMENDED)
```python
# 10% of total epochs for warmup
warmup_epochs = max(1, epochs // 10)
base_lr = 1e-4
max_lr = 5e-4  # Higher peak LR after warmup

# Implementation in training loop:
def get_lr(epoch):
    if epoch < warmup_epochs:
        return base_lr + (max_lr - base_lr) * (epoch / warmup_epochs)
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return eta_min + (max_lr - eta_min) * 0.5 * (1 + cos(pi * progress))
```

**Justification**:
- **Warmup stabilizes training** of attention mechanisms (Vaswani et al., 2017)
- **Higher peak LR** enables faster convergence in early epochs
- Used successfully in BERT, GPT, ViT pretraining

#### Option B: Discriminative Learning Rates
```python
# Different LRs for different modules
optimizer = AdamW([
    {'params': protein_encoder.parameters(), 'lr': 1e-4},
    {'params': ligand_encoder.parameters(), 'lr': 1e-4},
    {'params': cross_attention.parameters(), 'lr': 5e-4},  # Higher for cross-attn
    {'params': classifier.parameters(), 'lr': 2e-4}
])
```

**Justification**:
- Cross-attention is trained from scratch → needs higher LR
- Encoders process pretrained embeddings → need lower LR
- Classifier adapts to fused representation → intermediate LR

---

### 3.2 Priority 2: Data Augmentation (Expected Δ MCC: +0.03 to +0.07)

**Current**: No augmentation, static embeddings

**Problem**:
- Overfitting on training sequences (especially with only ~1K active compounds)
- No robustness to minor variations in embeddings

**Proposed Changes:**

#### A. Embedding Noise Injection (RECOMMENDED)
```python
class EmbeddingAugmentation(nn.Module):
    def __init__(self, noise_std=0.05, dropout_rate=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, training=True):
        if not training:
            return x
        
        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        
        # Random dropout of embedding dimensions
        x = self.dropout(x)
        
        return x

# Apply after loading embeddings, before encoders
protein_emb = self.augment(protein_emb, training=self.training)
ligand_emb = self.augment(ligand_emb, training=self.training)
```

**Justification**:
- **Regularization**: Prevents memorization of exact embeddings
- **Robustness**: Models learn to handle embedding variations
- Used in contrastive learning (SimCLR, MoCo) with success
- Small noise (std=0.05) preserves semantic information while adding diversity

#### B. MixUp for Protein-Ligand Pairs
```python
def mixup_batch(protein, ligand, labels, alpha=0.2):
    """MixUp augmentation for batch"""
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(protein.size(0))
    
    mixed_protein = lam * protein + (1 - lam) * protein[indices]
    mixed_ligand = lam * ligand + (1 - lam) * ligand[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_protein, mixed_ligand, mixed_labels
```

**Justification**:
- Creates synthetic intermediate examples
- Smooths decision boundaries
- Proven effective in image classification (Zhang et al., 2018)

---

### 3.3 Priority 3: Attention Head Optimization (Expected Δ MCC: +0.02 to +0.05)

**Current Configuration:**
```python
nhead = 8  # Fixed for all attention layers
```

**Problem**:
- One size doesn't fit all: different attention mechanisms may need different head counts
- 8 heads may be suboptimal for 320-dim protein embeddings (320/8 = 40 dim per head)

**Proposed Changes:**

#### Grid Search on Attention Heads
```python
# Test configurations:
configs = [
    {'protein_heads': 4, 'ligand_heads': 8, 'cross_heads': 8},   # Fewer heads for smaller dim
    {'protein_heads': 8, 'ligand_heads': 8, 'cross_heads': 16},  # More cross-attn heads
    {'protein_heads': 10, 'ligand_heads': 12, 'cross_heads': 8}, # Divisible by dimensions
]
```

**Justification**:
- **Protein (320-dim)**: 4 heads → 80 dim/head (better capacity)
- **Ligand (768-dim)**: 8 or 12 heads (768/12 = 64 dim/head)
- **Cross-attention**: More heads capture diverse interaction patterns

---

### 3.4 Priority 4: Batch Size & Gradient Accumulation (Expected Δ MCC: +0.02 to +0.04)

**Current Configuration:**
```python
batch_size = 32
gradient_accumulation = 1  # No accumulation
```

**Problem**:
- Small batch size → noisy gradients, unstable training
- Cross-attention benefits from larger batches (more diverse pairs)

**Proposed Changes:**

#### Gradient Accumulation for Effective Batch Size 128
```python
batch_size = 32          # Physical batch (GPU memory limit)
accumulation_steps = 4   # Effective batch = 32 * 4 = 128

# Training loop:
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Justification**:
- **Larger effective batch** → more stable gradients
- **No extra GPU memory** needed
- Used in large-scale transformer training (BERT used batch size 256)

---

### 3.5 Priority 5: Regularization Tuning (Expected Δ MCC: +0.01 to +0.03)

**Current Configuration:**
```python
weight_decay = 1e-5
dropout = 0.3  # Fixed across all layers
```

**Proposed Grid Search:**
```python
configs = [
    {'weight_decay': 1e-5, 'encoder_dropout': 0.1, 'classifier_dropout': 0.3},
    {'weight_decay': 1e-4, 'encoder_dropout': 0.1, 'classifier_dropout': 0.4},
    {'weight_decay': 5e-5, 'encoder_dropout': 0.15, 'classifier_dropout': 0.35},
]
```

**Justification**:
- **Lower dropout in encoders** (0.1): Preserve pretrained representations
- **Higher dropout in classifier** (0.3-0.4): Prevent overfitting on fused features
- **Weight decay**: Balance between regularization and capacity

---

## 4. IMPLEMENTATION STRATEGY

### Phase 1: Quick Wins (1-2 days)
1. **Implement warmup + cosine decay scheduler**
   - Expected: +0.05 MCC
   - Low risk, proven technique
   
2. **Add embedding noise augmentation**
   - Expected: +0.03 MCC
   - Easy to implement, no architecture change

**If Phase 1 reaches MCC > 0.58**, proceed to Phase 2.

---

### Phase 2: Systematic Search (3-5 days)
3. **Grid search on attention heads**
   - Test 3-5 configurations
   - Use validation MCC as selection criterion

4. **Implement gradient accumulation**
   - Effective batch size: 64, 96, 128
   - Monitor training stability

5. **Fine-tune regularization**
   - Weight decay: [1e-5, 5e-5, 1e-4]
   - Dropout: encoder [0.1, 0.15], classifier [0.3, 0.4]

**If Phase 2 reaches MCC > 0.6**, success! Otherwise, proceed to Phase 3.

---

### Phase 3: Advanced Techniques (5-7 days)
6. **Discriminative learning rates**
   - Separate LR for each module
   - Use learning rate finder (fastai-style)

7. **MixUp augmentation**
   - Test alpha values: [0.1, 0.2, 0.3]

8. **Ensemble methods**
   - Average predictions from top-3 checkpoints per seed
   - Expected: +0.02 MCC from ensemble alone

---

## 5. EXPERIMENTAL PROTOCOL

### 5.1 Baseline Replication
```bash
# Current Level 5-Lite baseline
python attention_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 5 \
    --epochs 50 \
    --batch_size 32 \
    --patience 5 \
    --seeds 42 123 456 789 1024
```

**Target to beat**: val_mcc = 0.50 (Epoch 3)

---

### 5.2 Level 6 CLI Interface
```bash
# Phase 1: Warmup + Augmentation
python attention_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 6 \
    --epochs 50 \
    --batch_size 32 \
    --patience 10 \
    --seeds 42 123 456 789 1024 \
    --warmup_epochs 5 \
    --max_lr 5e-4 \
    --augment_noise 0.05

# Phase 2: Grid Search
python attention_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 6 \
    --grid_search \
    --search_space attention_heads \
    --seeds 42

# Phase 3: Full Optimization
python attention_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 6 \
    --epochs 100 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --warmup_epochs 10 \
    --max_lr 5e-4 \
    --augment_noise 0.05 \
    --discriminative_lr \
    --mixup_alpha 0.2 \
    --seeds 42 123 456 789 1024
```

---

## 6. SUCCESS CRITERIA

### Minimum Viable Product (MVP)
- ✅ **Validation MCC > 0.55** (10% improvement over Level 5-Lite)
- ✅ **Test MCC > 0.53** (accounting for val/test gap)
- ✅ **Stable across seeds** (std < 0.03)

### Stretch Goal
- 🎯 **Validation MCC > 0.60** (20% improvement)
- 🎯 **Test MCC > 0.58**
- 🎯 **AUC > 0.85**

---

## 7. RISK MITIGATION

### Risk 1: Overfitting with Augmentation
**Symptom**: Train MCC ≫ Val MCC  
**Mitigation**: 
- Monitor train/val gap at each epoch
- Reduce augmentation strength if gap > 0.1
- Early stopping on validation MCC

### Risk 2: Training Instability with High LR
**Symptom**: Loss spikes, NaN gradients  
**Mitigation**:
- Gradient clipping (max_norm=1.0)
- Reduce max_lr from 5e-4 to 2e-4
- Increase warmup epochs from 5 to 10

### Risk 3: No Improvement from Hyperparameter Search
**Symptom**: All configs plateau at MCC ~0.50  
**Fallback**:
- Consider architecture modifications (e.g., deeper encoders)
- Investigate data quality (outliers, label noise)
- Try larger embedding model (150M instead of 8M)

---

## 8. EXPECTED TIMELINE

| Phase | Duration | Expected MCC | Confidence |
|-------|----------|--------------|------------|
| Baseline (Level 5-Lite) | Complete | 0.50 | ✅ Validated |
| Phase 1 (Warmup + Augment) | 1-2 days | 0.55-0.58 | 🟢 High (90%) |
| Phase 2 (Grid Search) | 3-5 days | 0.58-0.62 | 🟡 Medium (70%) |
| Phase 3 (Advanced) | 5-7 days | 0.60-0.65 | 🟠 Low (50%) |

**Total**: 9-14 days to MCC > 0.6 (optimistic)

---

## 9. ABLATION STUDY DESIGN

To isolate the contribution of each optimization, run ablations:

```python
experiments = [
    # Baseline
    {'name': 'Level5-Lite', 'warmup': False, 'augment': False, 'accum': 1},
    
    # Phase 1: Individual components
    {'name': 'L6-Warmup', 'warmup': True, 'augment': False, 'accum': 1},
    {'name': 'L6-Augment', 'warmup': False, 'augment': True, 'accum': 1},
    
    # Phase 1: Combined
    {'name': 'L6-Phase1', 'warmup': True, 'augment': True, 'accum': 1},
    
    # Phase 2: Add gradient accumulation
    {'name': 'L6-Phase2', 'warmup': True, 'augment': True, 'accum': 4},
    
    # Phase 3: Full optimization
    {'name': 'L6-Full', 'warmup': True, 'augment': True, 'accum': 4, 
     'discriminative_lr': True, 'mixup': True},
]
```

Run each with **3 seeds** (42, 123, 456) to measure mean ± std MCC.

---

## 10. MONITORING & CHECKPOINTING

### Key Metrics to Track
```python
metrics = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_mcc': [],
    'val_auc': [],
    'val_acc': [],
    'lr': [],  # Current learning rate
    'grad_norm': [],  # For stability monitoring
}
```

### Checkpoint Strategy
```python
# Save best model per seed
if val_mcc > best_val_mcc:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mcc': val_mcc,
        'config': hyperparams,
    }, f'level6_best_seed{seed}.pt')
```

---

## 11. IMPLEMENTATION STATUS

### ✅ **Stage 1: Hyperparameter Optimization (IMPLEMENTADO)**

**Status**: Completo e funcional

**Procedimento**:
1. Optuna TPE sampler otimiza 12 hiperparâmetros
2. MedianPruner descarta trials ruins precocemente
3. Early stopping (patience=5) por trial
4. Salva best trial em `best_hparams.json`

**Outputs**:
- `optimization_results.json`: best trial + n_trials
- `level6_{dataset}_{embedding}.db`: SQLite study database
- `best_hparams.json`: melhores hiperparâmetros

**Tempo**: ~12-48h dependendo de n_trials e embedding

---

### ✅ **Stage 2: Multi-seed Training (IMPLEMENTADO)**

**Status**: Completo e funcional

**Objetivo**: Avaliar robustez dos melhores hiperparâmetros com diferentes inicializações.

**Procedimento**:
1. Carrega best_hparams do Stage 1
2. Treina 5 modelos independentes: seeds `[42, 123, 456, 789, 1024]`
3. Cada modelo:
   - Treinado com CosineAnnealingLR
   - Early stopping (patience=5) baseado em val_mcc
   - Salva best checkpoint: `stage2_seed_{seed}.pt`
4. Avalia todos os 5 modelos no test set
5. Computa estatísticas: MCC mean ± std, AUC mean, ACC mean

**Outputs**:
- `stage2_seed_{seed}.pt`: 5 checkpoints com model_state_dict, hparams, test_metrics
- `stage2_multiseed_results.json`: agregação com means/stds

**Justificativa**:
- **Robustness check**: se std(MCC) > 0.03, hiperparâmetros são instáveis
- **Baseline for ensemble**: providencia modelos para Stage 3
- **Variance estimation**: crítico para reportar intervalos de confiança

**Tempo**: 5× tempo de um trial (~5-10h para 8M, ~15-30h para 650M)

---

### ✅ **Stage 3: Ensemble Prediction (IMPLEMENTADO)**

**Status**: Completo e funcional

**Objetivo**: Maximizar performance final via ensemble averaging.

**Procedimento**:
1. Carrega os 5 checkpoints do Stage 2
2. Para cada amostra do test set:
   - Computa logits de todos os 5 modelos
   - Converte para probabilidades: `p_i = sigmoid(logit_i)`
   - **Ensemble averaging**: `p_final = mean([p_1, p_2, p_3, p_4, p_5])`
   - Classificação binária: `y_pred = 1 if p_final >= 0.5 else 0`
3. Computa métricas finais: MCC, ACC, F1, AUC, Precision, Recall

**Outputs**:
- `stage3_ensemble_results.json`: métricas finais do ensemble

**Justificativa Científica**:
- **Redução de variância**: Ensemble averaging cancela ruído aleatório de diferentes inicializações
- **Boosting de performance**: Literatura mostra ganho típico de +0.01 a +0.03 em MCC
- **State-of-the-art prática**: Usado em Kaggle (top teams), AlphaFold2, ESM-Fold
- **Bias-variance tradeoff**: Reduz variance sem aumentar bias (diferente de bagging)
- **Wisdom of crowds**: Modelos com seeds diferentes capturam patterns complementares

**Implementação**:
```python
# Load all 5 models
ensemble_models = []
for checkpoint_path in stage2_checkpoints:
    model = Level6OptimizedModel(**best_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    ensemble_models.append(model)

# Ensemble prediction
all_probs = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        batch_probs = []
        for model in ensemble_models:
            logits = model(batch['protein'], batch['ligand'], 
                          batch['protein_mask'], batch['ligand_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()
            batch_probs.append(probs)
        
        # Average probabilities
        ensemble_prob = np.mean(batch_probs, axis=0)
        all_probs.append(ensemble_prob)
        all_labels.append(batch['labels'].numpy())

# Compute final metrics
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
all_probs = np.concatenate(all_probs).flatten()
all_labels = np.concatenate(all_labels)
preds = (all_probs >= 0.5).astype(int)

ensemble_mcc = matthews_corrcoef(all_labels, preds)
ensemble_auc = roc_auc_score(all_labels, all_probs)
ensemble_f1 = f1_score(all_labels, preds)
```

**Tempo**: ~5-10 minutos (apenas inferência, sem treinamento)

**Expected Gain**: MCC Stage3 ≥ MCC Stage2_mean + 0.01

---

## 12. COMMAND LINE USAGE

```bash
# Full Level 6 pipeline (all 3 stages)
python attention_screening_models.py \
    --dataset human \
    --embedding 8M \
    --levels 6 \
    --opt \
    --n_trials 20 \
    --opt_timeout 48
```

**Flags**:
- `--opt`: Ativa Level 6 optimization mode
- `--n_trials`: Número de trials Optuna (Stage 1)
- `--opt_timeout`: Timeout em horas (0 = sem limite)

**Output Structure**:
```
results/benchmark_human_8M/level6_optimized_8M/
├── optimization_results.json       # Stage 1: best trial
├── best_hparams.json               # Best hyperparameters
├── level6_human_8M.db              # Optuna study database
├── stage2_seed_42.pt               # Stage 2: 5 checkpoints
├── stage2_seed_123.pt
├── stage2_seed_456.pt
├── stage2_seed_789.pt
├── stage2_seed_1024.pt
├── stage2_multiseed_results.json   # Stage 2: aggregated stats
└── stage3_ensemble_results.json    # Stage 3: final metrics
```

---

## 13. REFERENCES

1. **Warmup Scheduling**: Vaswani et al. (2017) - Attention Is All You Need
2. **Discriminative LR**: Howard & Ruder (2018) - Universal Language Model Fine-tuning
3. **Embedding Augmentation**: Chen et al. (2020) - SimCLR: A Simple Framework for Contrastive Learning
4. **MixUp**: Zhang et al. (2018) - mixup: Beyond Empirical Risk Minimization
5. **Gradient Accumulation**: Ott et al. (2018) - Scaling Neural Machine Translation (fairseq paper)
6. **Ensemble Methods**: Dietterich (2000) - Ensemble Methods in Machine Learning
7. **Optuna**: Akiba et al. (2019) - Optuna: A Next-generation Hyperparameter Optimization Framework

---

## 14. NEXT STEPS

1. **Immediate**: Run Level 6 Stage 1 (HPO) on human/8M
2. **Week 1**: Complete Stages 2+3, analyze results
3. **Week 2**: If MCC < 0.60, apply Phase 2 optimizations (augmentation, warmup)
4. **Week 3**: Scale to 650M embedding for maximum performance
5. **Deliverable**: Level 6 achieving **MCC > 0.60** on human kinase benchmark

---

**Document Version**: 2.0  
**Last Updated**: 2026-03-02  
**Status**: All 3 stages implemented and tested  
**Next Review**: After first HPO run completion
