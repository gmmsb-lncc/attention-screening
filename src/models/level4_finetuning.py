"""
Level 4: ESM-2 Fine-tuning for Kinase Domain

Fine-tunes ESM-2 models on kinase sequences using masked language modeling (MLM).
Implements early stopping based on validation perplexity to prevent overfitting.

Scientific Rationale:
    - Pre-trained ESM-2 models are trained on general protein sequences
    - Fine-tuning on kinase-specific sequences improves domain-specific representations
    - Better representations → better downstream task performance
    - Uses only TRAINING data for fine-tuning to prevent data leakage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
import json
import sys

# Add ESM path
esm_path = Path(__file__).resolve().parent.parent.parent / "llm" / "ESM"
if str(esm_path) not in sys.path:
    sys.path.insert(0, str(esm_path))

import esm


class KinaseMLMDataset(Dataset):
    """Dataset for Masked Language Modeling on kinase sequences."""
    
    def __init__(self, sequences: List[str], mask_prob: float = 0.15):
        """
        Args:
            sequences: List of protein sequences (strings)
            mask_prob: Probability of masking each token
        """
        self.sequences = sequences
        self.mask_prob = mask_prob
        _, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mask_idx = self.alphabet.mask_idx
        self.pad_idx = self.alphabet.padding_idx
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        # Convert sequence to tokens
        data = [("", seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens[0]  # Remove batch dimension
        
        # Create labels (clone original tokens)
        labels = tokens.clone()
        
        # Create mask for tokens to mask (exclude special tokens <cls>, <eos>, <pad>)
        maskable = (tokens != self.alphabet.cls_idx) & \
                   (tokens != self.alphabet.eos_idx) & \
                   (tokens != self.pad_idx)
        
        # Randomly select tokens to mask
        mask_positions = torch.rand(tokens.shape) < self.mask_prob
        mask_positions = mask_positions & maskable
        
        # Apply masking: 80% <mask>, 10% random, 10% unchanged
        rand = torch.rand(tokens.shape)
        tokens[mask_positions & (rand < 0.8)] = self.mask_idx
        
        # Random token replacement (10%)
        random_replacement = mask_positions & (rand >= 0.8) & (rand < 0.9)
        random_tokens = torch.randint(4, len(self.alphabet) - 1, tokens.shape)
        tokens[random_replacement] = random_tokens[random_replacement]
        
        # 10% unchanged (rand >= 0.9)
        
        # Set non-masked positions to -100 (ignore in loss)
        labels[~mask_positions] = -100
        
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': (tokens != self.pad_idx).long()
        }


def collate_mlm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for MLM batches with padding."""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    pad_idx = batch[0]['input_ids'].new_full((1,), 1)[0]  # Padding idx
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), pad_idx, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Pad labels
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)
        
        # Pad attention mask
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(padded_mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


def finetune_esm(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    warmup_steps: int = 100,
    patience: int = 3
) -> Dict[str, Any]:
    """
    Fine-tune ESM model with early stopping.
    
    Args:
        model: ESM model to fine-tune
        train_dataloader: Training data
        val_dataloader: Validation data
        device: Device to train on
        output_dir: Directory to save checkpoints
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        patience: Early stopping patience (epochs without improvement)
    
    Returns:
        Dictionary with training history and best metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
        'epochs': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n🔬 Fine-tuning ESM model")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {device}\n")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            results = model(input_ids, repr_layers=[])
            logits = results['logits']
            
            # Calculate loss only on masked positions
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_steps
        train_ppl = np.exp(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                results = model(input_ids, repr_layers=[])
                logits = results['logits']
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                val_loss += loss.item()
                val_steps += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_steps
        val_ppl = np.exp(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_perplexity'].append(train_ppl)
        history['val_loss'].append(avg_val_loss)
        history['val_perplexity'].append(val_ppl)
        history['epochs'].append(epoch + 1)
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_ppl:.2f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_perplexity': val_ppl,
                'history': history
            }, checkpoint_path)
            
            print(f"  ✅ New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break
    
    # Save final history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_perplexity': np.exp(best_val_loss),
        'history': history,
        'checkpoint_path': str(output_dir / "best_model.pt")
    }


def load_finetuned_model(checkpoint_path: Path, device: torch.device):
    """Load fine-tuned ESM model from checkpoint."""
    # Load base model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    
    # Load fine-tuned weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model, alphabet
