# 🔬 Level 4: ESM-2 Fine-tuning para Kinases

## 📋 Visão Geral

O **Level 4** implementa **fine-tuning supervisionado do modelo ESM-2** especificamente para o domínio de kinases. Ao invés de usar embeddings "prontos" do modelo pré-treinado (vanilla), adaptamos os pesos do ESM-2 aos padrões específicos das sequências de kinases, melhorando a qualidade das representações.

---

## 🎯 Objetivo Científico

### Por que Fine-tuning?

**ESM-2 vanilla** foi treinado em 250M+ proteínas de todos os domínios (kinases, anticorpos, enzimas metabólicas, etc.). Embora capture padrões gerais de estrutura e função proteica, ele **não foi especializado em kinases**.

**Fine-tuning** adapta o modelo para:
- Reconhecer motivos catalíticos específicos de kinases (DFG, HRD, VAIK)
- Capturar variações em regiões regulatórias (activation loop, P-loop)
- Discriminar subfamílias de kinases (TK, CMGC, AGC, etc.)
- Melhorar representações para predição de afinidade com ligantes

---

## 🧬 Metodologia Científica

### 1. **Tarefa de Fine-tuning: Masked Language Modeling (MLM)**

Usamos a **mesma tarefa de pré-treinamento do ESM-2**, mas agora **apenas com sequências de kinases**:

```
Sequência original:  M E T A A K F E R Q H M D S
Sequência mascarada: M E T A <mask> K F E <mask> Q H M D S
Objetivo:            Prever A e R nas posições mascaradas
```

**Por que MLM e não classificação direta?**
- ✅ Preserva o conhecimento geral do modelo (transfer learning)
- ✅ Aprende representações contextuais específicas de kinases
- ✅ Não "esquece" padrões estruturais universais (evita catastrophic forgetting)
- ✅ Permite usar **apenas sequências** (não precisa de labels de afinidade)

---

### 2. **Protocolo Experimental Rigoroso**

#### **2.1. Separação Train/Val/Test**

**CRÍTICO**: O fine-tuning deve usar **APENAS o conjunto de treino**!

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset Original (375,353 pares proteína-ligante)         │
├─────────────────────────────────────────────────────────────┤
│  Train:      269,715 (71.8%) → FINE-TUNING + TRAINING     │
│  Validation:  65,168 (17.4%) → EARLY STOPPING             │
│  Test:        40,470 (10.8%) → AVALIAÇÃO FINAL            │
└─────────────────────────────────────────────────────────────┘
```

**Por que não usar Val/Test no fine-tuning?**
- ❌ **Data leakage**: O modelo veria informações do teste antes da avaliação final
- ❌ **Overfitting**: Ajustaria pesos para dados que devem ser "invisíveis"
- ❌ **Invalida cientificamente os resultados**: Não há como saber se o ganho é real

#### **2.2. Pipeline Completo**

```
FASE 1: FINE-TUNING (Level 4)
├─ Input:  Train sequences (unique kinases)
├─ Task:   Masked Language Modeling (15% masking)
├─ Output: ESM-2 fine-tuned checkpoint
└─ Duração: ~8-12h (GPU V100/A100)

FASE 2: EMBEDDING EXTRACTION
├─ Input:  Fine-tuned ESM-2 checkpoint
├─ Process: Extract embeddings for Train/Val/Test
└─ Output: protein_matrices/ (fine-tuned)

FASE 3: DOWNSTREAM TRAINING (Levels 1-3, 6)
├─ Input:  Fine-tuned embeddings
├─ Models: Level 1 (FP), Level 2 (Emb+AttPool), Level 3 (CrossAtt)
└─ Output: Final predictions
```

---

## 🏗️ Arquitetura Técnica

### Componentes do Fine-tuning

```python
# Modelo base (ESM-2 8M)
model = esm.pretrained.esm2_t6_8M_UR50D()
# 6 layers, 320 dim, 20 heads
# ~8M parâmetros

# Configuração de fine-tuning
mask_ratio = 0.15          # 15% dos tokens mascarados
learning_rate = 1e-5       # LR baixo para não "destruir" pré-treino
warmup_steps = 1000        # Warm-up para estabilidade
max_epochs = 10            # Poucas épocas (já está pré-treinado)
batch_size = 8             # Depende da GPU
gradient_accumulation = 4  # Simula batch_size=32
```

### Estratégia de Masking

```python
def mask_tokens(sequence, mask_ratio=0.15):
    """
    80% → <mask> token
    10% → random amino acid
    10% → unchanged (para robustez)
    """
    masked_positions = random.sample(range(len(sequence)), 
                                     int(len(sequence) * mask_ratio))
    
    for pos in masked_positions:
        rand = random.random()
        if rand < 0.8:
            sequence[pos] = '<mask>'  # 80%
        elif rand < 0.9:
            sequence[pos] = random_amino_acid()  # 10%
        # else: mantém original (10%)
    
    return sequence, masked_positions
```

---

## 📊 Métricas de Fine-tuning

### Durante o Fine-tuning (MLM Task)

```python
# Métricas por época
{
    'train_loss': 2.45,          # Cross-entropy loss
    'train_perplexity': 11.6,    # exp(loss)
    'train_accuracy': 0.67,      # % tokens corretos
    'val_loss': 2.38,
    'val_perplexity': 10.8,
    'val_accuracy': 0.69
}
```

**Convergência esperada:**
- Epoch 1: loss ~3.5, acc ~0.50
- Epoch 5: loss ~2.4, acc ~0.68
- Epoch 10: loss ~2.2, acc ~0.72

### Após Fine-tuning (Downstream Task)

Comparar **MCC** dos níveis usando embeddings vanilla vs fine-tuned:

```
                  Vanilla   Fine-tuned   Δ MCC
Level 1 (FP)      0.428     0.428       0.000  (não usa embeddings)
Level 2 (Emb)     0.391     0.450*      +0.059
Level 3 (CrossAtt) 0.498    0.550*      +0.052
```

**Hipótese**: Fine-tuning deve melhorar MCC em ~0.05-0.10 pontos.

---

## 🛠️ Implementação

### Estrutura de Arquivos

```
src/
└── finetuning/
    ├── __init__.py
    ├── esm_finetuner.py      # Classe principal
    ├── masking.py            # Estratégia de masking
    └── data_loader.py        # DataLoader para MLM

results/
└── finetuned_models/
    ├── esm2_8M_kinase_finetuned/
    │   ├── checkpoint_epoch_10.pt
    │   ├── config.json
    │   └── training_log.json
    └── embeddings/
        ├── protein_matrices/  # Embeddings fine-tuned
        └── metadata.json
```

### Código Principal

```python
# src/finetuning/esm_finetuner.py

import torch
import esm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

class ESMFineTuner:
    def __init__(self, model_name='esm2_t6_8M_UR50D', device='cuda'):
        self.model, self.alphabet = esm.pretrained[model_name]()
        self.model = self.model.to(device)
        self.device = device
        self.mask_idx = self.alphabet.mask_idx
        
    def prepare_batch(self, sequences, mask_ratio=0.15):
        """Cria batch com tokens mascarados"""
        batch_tokens = []
        batch_labels = []
        
        for seq in sequences:
            tokens = self.alphabet.encode(seq)
            labels = tokens.clone()
            
            # Mascara 15% dos tokens
            mask_positions = torch.rand(len(tokens)) < mask_ratio
            mask_positions[0] = False  # Não mascara <cls>
            mask_positions[-1] = False  # Não mascara <eos>
            
            # 80% → <mask>, 10% → random, 10% → unchanged
            for i in torch.where(mask_positions)[0]:
                rand = torch.rand(1).item()
                if rand < 0.8:
                    tokens[i] = self.mask_idx
                elif rand < 0.9:
                    tokens[i] = torch.randint(4, 24, (1,)).item()  # Random AA
            
            batch_tokens.append(tokens)
            batch_labels.append(labels)
        
        return torch.stack(batch_tokens), torch.stack(batch_labels)
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_seqs in dataloader:
            tokens, labels = self.prepare_batch(batch_seqs)
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            results = self.model(tokens, repr_layers=[6])
            logits = results['logits']
            
            # Loss apenas nos tokens mascarados
            mask_positions = tokens == self.mask_idx
            loss = F.cross_entropy(
                logits[mask_positions].view(-1, self.alphabet.size),
                labels[mask_positions].view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Métricas
            total_loss += loss.item()
            preds = logits[mask_positions].argmax(dim=-1)
            correct += (preds == labels[mask_positions]).sum().item()
            total += mask_positions.sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total,
            'perplexity': torch.exp(torch.tensor(total_loss / len(dataloader))).item()
        }
    
    def save_checkpoint(self, path, epoch, optimizer, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, path)
```

---

## 🚀 Uso via CLI

### 1. Fine-tuning do ESM-2

```bash
# Fine-tune ESM-2 8M com kinases humanas (apenas train set)
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 4 \
    --finetune_epochs 10 \
    --finetune_lr 1e-5 \
    --finetune_batch_size 8

# Output esperado:
# results/benchmark_human_8M/finetuned_esm2_8M/
#   ├── checkpoint_epoch_10.pt
#   ├── training_log.json
#   └── config.json
```

### 2. Extrair Embeddings Fine-tuned

```bash
# Extrai embeddings usando o modelo fine-tuned
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 0 \
    --use_finetuned \
    --finetuned_checkpoint results/benchmark_human_8M/finetuned_esm2_8M/checkpoint_epoch_10.pt

# Output:
# results/benchmark_human_8M/build_finetuned/
#   ├── protein_matrices/
#   └── ligand_matrices/
```

### 3. Treinar Níveis 2-3 com Embeddings Fine-tuned

```bash
# Pipeline completo com embeddings fine-tuned
python semantic_screening_models_beta.py \
    --dataset human \
    --embedding 8M \
    --levels 2 3 \
    --use_finetuned \
    --finetuned_checkpoint results/benchmark_human_8M/finetuned_esm2_8M/checkpoint_epoch_10.pt \
    --epochs 50 \
    --batch_size 32 \
    --seeds 42 123 456 789 1024
```

---

## 📈 Resultados Esperados

### Comparação Vanilla vs Fine-tuned

```
┌──────────────────────────────────────────────────────────┐
│  Level 2 (Embeddings + Attention Pooling + MLP)         │
├──────────────────────────────────────────────────────────┤
│  ESM-2 Vanilla:     MCC = 0.391 ± 0.012                │
│  ESM-2 Fine-tuned:  MCC = 0.450 ± 0.010  (+0.059)      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Level 3 (Cross-Attention)                               │
├──────────────────────────────────────────────────────────┤
│  ESM-2 Vanilla:     MCC = 0.498 ± 0.008                │
│  ESM-2 Fine-tuned:  MCC = 0.550 ± 0.007  (+0.052)      │
└──────────────────────────────────────────────────────────┘
```

### Análise t-SNE

Visualizar embeddings antes/depois do fine-tuning deve mostrar:
- **Vanilla**: Clusters genéricos por fold (TK, CMGC, AGC misturados)
- **Fine-tuned**: Separação mais clara de subfamílias e padrões de afinidade

---

## ⚠️ Cuidados Metodológicos

### 1. **Data Leakage Prevention**

```python
# ✅ CORRETO
train_seqs = load_sequences(train_split)  # Apenas treino
finetuner.train(train_seqs)

# ❌ ERRADO
all_seqs = load_sequences(train + val + test)  # Leakage!
finetuner.train(all_seqs)
```

### 2. **Overfitting no Fine-tuning**

**Sinais de overfitting:**
- Val loss aumenta enquanto train loss diminui
- Accuracy no MLM > 0.80 (decorou sequências)
- MCC downstream não melhora

**Soluções:**
- Usar early stopping (patience=3 no MLM)
- Dropout 0.1-0.2 nas camadas finais
- Learning rate baixo (1e-5 ou 1e-6)
- Poucas épocas (5-10)

### 3. **Comparação Justa**

Para comparar vanilla vs fine-tuned:
- Mesma arquitetura downstream (Level 2, 3)
- Mesmos hiperparâmetros (lr, batch_size, epochs)
- Mesmas seeds (42, 123, 456, 789, 1024)
- Mesmo conjunto de teste

---

## 📚 Referências Científicas

1. **ESM-2 Original Paper**
   - Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model"
   - *Science*, 379(6637), 1123-1130.

2. **Fine-tuning PLMs**
   - Elnaggar et al. (2021). "ProtTrans: Toward understanding the language of life through self-supervised learning"
   - *IEEE TPAMI*, 44(10), 7112-7127.

3. **Task-Specific Fine-tuning**
   - Brandes et al. (2022). "ProteinBERT: a universal deep-learning model of protein sequence and function"
   - *Bioinformatics*, 38(8), 2102-2110.

4. **Kinase-Ligand Binding**
   - Davis et al. (2011). "Comprehensive analysis of kinase inhibitor selectivity"
   - *Nature Biotechnology*, 29(11), 1046-1051.

---

## 🔄 Workflow Resumido

```bash
# PASSO 1: Fine-tune ESM-2 (Level 4)
python semantic_screening_models_beta.py --dataset human --embedding 8M --levels 4 --finetune_epochs 10

# PASSO 2: Extrair embeddings fine-tuned (Level 0)
python semantic_screening_models_beta.py --dataset human --embedding 8M --levels 0 --use_finetuned

# PASSO 3: Treinar downstream (Levels 2-3)
python semantic_screening_models_beta.py --dataset human --embedding 8M --levels 2 3 --use_finetuned

# PASSO 4: Comparar resultados
python scripts/compare_vanilla_vs_finetuned.py --results results/benchmark_human_8M/
```

---

## 💡 Conclusão

O **Level 4 (Fine-tuning)** é uma etapa **opcional mas potencialmente muito valiosa** para melhorar a qualidade das representações proteicas. Se implementado corretamente (sem data leakage), pode:

✅ Melhorar MCC em 0.05-0.10 pontos  
✅ Reduzir overfitting nos níveis downstream  
✅ Capturar padrões específicos de kinases  
✅ Publicar resultados mais robustos cientificamente  

No entanto, requer:
- ~8-12h de treinamento em GPU
- Validação cuidadosa para evitar data leakage
- Comparação justa com baseline vanilla

---

**Autor**: Claude + Leon  
**Data**: Março 2026  
**Versão**: 1.0
