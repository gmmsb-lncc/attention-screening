#!/usr/bin/env python3
"""ESM-2 Fine-tuning Module for Kinase Domain Adaptation.

This module fine-tunes ESM-2 protein language models on kinase-specific data
to improve downstream task performance while preventing data leakage.

Key Features:
- Uses ONLY training split for fine-tuning (no val/test contamination)
- Masked Language Modeling (MLM) objective
- Saves fine-tuned model checkpoints
- Generates embeddings from fine-tuned model

Scientific Justification:
Domain adaptation via fine-tuning improves protein representations for
specialized protein families (kinases) compared to general pre-training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import json

try:
    from esm import pretrained
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "llm" / "ESM"))
    from esm import pretrained


class ProteinMLMDataset(Dataset):
    """Dataset for Masked Language Modeling on protein sequences."""
    
    def __init__(
        self,
        sequences: List[str],
        seq_ids: List[str],
        alphabet,
        max_len: int = 1024,
        mask_prob: float = 0.15
    ):
        """
        Args:
            sequences: List of protein sequences (amino acid strings)
            seq_ids: List of sequence identifiers
            alphabet: ESM alphabet for tokenization
            max_len: Maximum sequence length
            mask_prob: Probability of masking each token
        """
        self.sequences = sequences
        self.seq_ids = seq_ids
        self.alphabet = alphabet
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_idx = alphabet.mask_idx
        self.pad_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_id = self.seq_ids[idx]
        
        # Truncate if needed
        if len(seq) > self.max_len - 2:  # Reserve space for <cls> and <eos>
            seq = seq[:self.max_len - 2]
        
        # Tokenize
        tokens = [self.cls_idx]
        tokens.extend(self.alphabet.encode(seq))
        tokens.append(self.eos_idx)
        
        # Create labels (for MLM loss)
        labels = tokens.copy()
        
        # Apply masking (skip <cls> and <eos>)
        for i in range(1, len(tokens) - 1):
            if np.random.random() < self.mask_prob:
                # 80% mask, 10% random, 10% keep
                rand = np.random.random()
                if rand < 0.8:
                    tokens[i] = self.mask_idx
                elif rand < 0.9:
                    # Random token (excluding special tokens)
                    tokens[i] = np.random.randint(4, len(self.alphabet) - 5)
                # else: keep original
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_id': seq_id
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(len(item['tokens']) for item in batch)
    
    tokens_padded = []
    labels_padded = []
    seq_ids = []
    
    pad_idx = batch[0]['tokens'].new_full((1,), 1)[0]  # Use padding_idx=1
    
    for item in batch:
        tokens = item['tokens']
        labels = item['labels']
        
        # Pad
        padding = max_len - len(tokens)
        tokens_padded.append(torch.cat([tokens, tokens.new_full((padding,), pad_idx)]))
        labels_padded.append(torch.cat([labels, labels.new_full((padding,), -100)]))  # -100 is ignored by CrossEntropyLoss
        seq_ids.append(item['seq_id'])
    
    return {
        'tokens': torch.stack(tokens_padded),
        'labels': torch.stack(labels_padded),
        'seq_ids': seq_ids
    }


class ESMFineTuner:
    """Fine-tunes ESM-2 models on kinase sequences."""
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_len: int = 1024,
        mask_prob: float = 0.15
    ):
        """
        Args:
            model_name: ESM-2 model name
            device: Device to use
            learning_rate: Learning rate for fine-tuning
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_len: Maximum sequence length
            mask_prob: Probability of masking tokens
        """
        self.model_name = model_name
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        # Load model
        print(f"Loading ESM-2 model: {model_name}")
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        print(f"Model loaded on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def finetune(
        self,
        train_sequences: List[str],
        train_seq_ids: List[str],
        epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        save_path: Optional[Path] = None,
        save_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Fine-tune ESM-2 model on kinase training sequences.
        
        Args:
            train_sequences: List of training protein sequences
            train_seq_ids: List of training sequence IDs
            epochs: Number of training epochs
            batch_size: Batch size for training
            gradient_accumulation_steps: Steps to accumulate gradients
            save_path: Path to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Dictionary with training history
        """
        # Create dataset
        dataset = ProteinMLMDataset(
            train_sequences,
            train_seq_ids,
            self.alphabet,
            max_len=self.max_len,
            mask_prob=self.mask_prob
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Training loop
        self.model.train()
        history = {'loss': [], 'perplexity': []}
        
        total_steps = len(dataloader) * epochs
        global_step = 0
        
        print(f"\nFine-tuning on {len(train_sequences)} sequences")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Total steps: {total_steps}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(pbar):
                tokens = batch['tokens'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(tokens, repr_layers=[])
                logits = outputs['logits']
                
                # Compute MLM loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Normalize loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Learning rate warmup
                    if global_step < self.warmup_steps:
                        lr_scale = min(1.0, float(global_step + 1) / self.warmup_steps)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = lr_scale * self.learning_rate
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
            
            # Epoch statistics
            avg_loss = epoch_loss / epoch_steps
            perplexity = np.exp(avg_loss)
            history['loss'].append(avg_loss)
            history['perplexity'].append(perplexity)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # Save checkpoint
            if save_path and (epoch + 1) % save_every == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, avg_loss)
        
        # Save final model
        if save_path:
            final_path = save_path / "final_model.pt"
            self.save_checkpoint(final_path, epochs, history['loss'][-1])
            
            # Save training history
            history_path = save_path / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        print("\nFine-tuning completed!")
        return history
    
    def save_checkpoint(self, path: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_name': self.model_name,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
        return checkpoint
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        sequences: List[str],
        seq_ids: List[str],
        batch_size: int = 8,
        repr_layer: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from fine-tuned model.
        
        Args:
            sequences: List of protein sequences
            seq_ids: List of sequence IDs
            batch_size: Batch size for inference
            repr_layer: Which layer to extract representations from (-1 = last layer)
            
        Returns:
            Dictionary mapping seq_id to embedding matrix [L, D]
        """
        self.model.eval()
        embeddings = {}
        
        # Determine which layer to extract
        if repr_layer == -1:
            repr_layer = self.model.num_layers
        
        print(f"\nExtracting embeddings from layer {repr_layer}")
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids = seq_ids[i:i+batch_size]
            
            # Prepare batch
            batch_labels, batch_strs, batch_tokens = self.batch_converter(
                [(seq_id, seq) for seq_id, seq in zip(batch_ids, batch_seqs)]
            )
            batch_tokens = batch_tokens.to(self.device)
            
            # Extract representations
            results = self.model(batch_tokens, repr_layers=[repr_layer])
            representations = results['representations'][repr_layer]
            
            # Store per-sequence embeddings (remove <cls> and <eos>)
            for j, seq_id in enumerate(batch_ids):
                seq_len = len(batch_seqs[j])
                emb = representations[j, 1:seq_len+1].cpu().numpy()  # [L, D]
                embeddings[seq_id] = emb
        
        return embeddings
