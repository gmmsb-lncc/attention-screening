"""
ESM-2 Fine-tuning Module for Level 4
Fine-tunes ESM-2 models on kinase training data using masked language modeling.
Implements memory-efficient training with gradient checkpointing and mixed precision.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import List, Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path
import numpy as np
import gc

# Add ESM to path
esm_path = Path(__file__).parent.parent.parent / "llm" / "ESM"
if str(esm_path) not in sys.path:
    sys.path.insert(0, str(esm_path))

import esm


class ProteinMLMDataset(Dataset):
    """Dataset for masked language modeling on protein sequences."""
    
    def __init__(
        self,
        sequences: List[str],
        seq_ids: List[str],
        alphabet,
        max_length: int = 1024,
        mask_prob: float = 0.15
    ):
        self.sequences = sequences
        self.seq_ids = seq_ids
        self.alphabet = alphabet
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_idx = alphabet.mask_idx
        self.batch_converter = alphabet.get_batch_converter()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_id = self.seq_ids[idx]
        
        # Truncate if needed
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        
        return seq_id, seq


def create_masked_batch(batch, alphabet, mask_prob=0.15):
    """
    Create masked batch for MLM training.
    Returns original tokens and masked tokens.
    """
    _, _, batch_tokens = batch
    
    # Clone tokens for masking
    masked_tokens = batch_tokens.clone()
    
    # Create mask (ignore special tokens: <cls>=0, <eos>=2, <pad>=1)
    special_tokens_mask = (batch_tokens == alphabet.cls_idx) | \
                          (batch_tokens == alphabet.eos_idx) | \
                          (batch_tokens == alphabet.padding_idx)
    
    # Random masking
    mask = torch.rand(batch_tokens.shape) < mask_prob
    mask = mask & ~special_tokens_mask
    
    # Apply masking
    masked_tokens[mask] = alphabet.mask_idx
    
    return batch_tokens, masked_tokens, mask


class ESMFinetuner:
    """Fine-tunes ESM-2 models on kinase sequences using MLM with memory-efficient training."""
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mask_prob: float = 0.15,
        use_amp: bool = True,  # Mixed precision for memory efficiency
    ):
        self.model_name = model_name
        self.device = device
        self.mask_prob = mask_prob
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        print(f"  Loading ESM-2 model: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Enable gradient checkpointing for memory efficiency (if available)
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)
            print("  ✓ Gradient checkpointing enabled")
        
        print(f"  Model loaded on {device}")
        print(f"  Mixed precision (AMP): {self.use_amp}")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {n_params:,}")
    
    def prepare_data(
        self,
        train_tsv: str,
        val_tsv: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare dataloaders from TSV files."""
        
        print(f"  Loading training data from {train_tsv}...")
        train_df = pd.read_csv(train_tsv, sep='\t', compression='gzip' if train_tsv.endswith('.gz') else None)
        
        # Get unique sequences
        unique_seqs = train_df[['seq_id', 'seq']].drop_duplicates()
        train_sequences = unique_seqs['seq'].tolist()
        train_ids = unique_seqs['seq_id'].tolist()
        
        print(f"  Training set: {len(train_sequences)} unique sequences")
        
        train_dataset = ProteinMLMDataset(
            train_sequences, train_ids, self.alphabet, max_length, self.mask_prob
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.batch_converter,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = None
        if val_tsv:
            print(f"  Loading validation data from {val_tsv}...")
            val_df = pd.read_csv(val_tsv, sep='\t', compression='gzip' if val_tsv.endswith('.gz') else None)
            unique_val_seqs = val_df[['seq_id', 'seq']].drop_duplicates()
            val_sequences = unique_val_seqs['seq'].tolist()
            val_ids = unique_val_seqs['seq_id'].tolist()
            
            print(f"  Validation set: {len(val_sequences)} unique sequences")
            
            val_dataset = ProteinMLMDataset(
                val_sequences, val_ids, self.alphabet, max_length, self.mask_prob
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.batch_converter,
                num_workers=2,
                pin_memory=True
            )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,  # Higher for memory efficiency
        save_path: Optional[str] = None,
        patience: int = 3,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model using masked language modeling.
        
        Uses:
        - Mixed precision training (FP16) for memory efficiency
        - Gradient accumulation
        - Gradient clipping
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_accumulation_steps: Gradient accumulation steps
            save_path: Path to save the fine-tuned model
            patience: Early stopping patience (epochs without improvement)
            max_grad_norm: Max gradient norm for clipping
        
        Returns:
            Dictionary with training history
        """
        
        self.model.train()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.alphabet.padding_idx)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': []
        }
        
        print(f"\n  Starting ESM-2 fine-tuning for {epochs} epochs (patience={patience})...")
        print(f"  Learning rate: {learning_rate}, Warmup steps: {warmup_steps}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"  Mixed precision (AMP): {self.use_amp}")
        
        global_step = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()
            
            # Clear cache before each epoch
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            pbar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch+1}/{epochs} [Train]",
                leave=True
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Create masked batch
                original_tokens, masked_tokens, mask = create_masked_batch(
                    batch, self.alphabet, self.mask_prob
                )
                
                original_tokens = original_tokens.to(self.device)
                masked_tokens = masked_tokens.to(self.device)
                mask = mask.to(self.device)
                
                try:
                    # Mixed precision forward pass
                    if self.use_amp:
                        with autocast():
                            results = self.model(masked_tokens, repr_layers=[])
                            logits = results["logits"]
                            
                            # Compute loss only on masked tokens
                            masked_logits = logits[mask]
                            masked_labels = original_tokens[mask]
                            
                            if masked_logits.numel() > 0:
                                loss = criterion(masked_logits, masked_labels)
                            else:
                                continue
                        
                        # Scaled backward
                        loss = loss / gradient_accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        results = self.model(masked_tokens, repr_layers=[])
                        logits = results["logits"]
                        
                        masked_logits = logits[mask]
                        masked_labels = original_tokens[mask]
                        
                        if masked_logits.numel() > 0:
                            loss = criterion(masked_logits, masked_labels)
                        else:
                            continue
                        
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n  WARNING: OOM at batch {batch_idx}, skipping...")
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
                
                # Update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_perplexity = np.exp(min(avg_train_loss, 10))  # Cap to prevent overflow
            
            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(current_lr)
            history['perplexity'].append(train_perplexity)
            
            # Validation or use train loss for early stopping
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                val_perplexity = np.exp(min(val_loss, 10))
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} (PPL: {train_perplexity:.2f}), "
                      f"Val Loss: {val_loss:.4f} (PPL: {val_perplexity:.2f})")
                
                # Early stopping based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if save_path:
                        self.save_model(save_path)
                        print(f"  → Best model saved (val_loss={val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"  → No improvement for {epochs_without_improvement}/{patience} epochs")
            else:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} (PPL: {train_perplexity:.2f})")
                
                # Early stopping based on train loss when no validation
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    epochs_without_improvement = 0
                    if save_path:
                        self.save_model(save_path)
                        print(f"  → Best model saved (train_loss={avg_train_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"  → No improvement for {epochs_without_improvement}/{patience} epochs")
            
            # Check early stopping
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs")
                break
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate the model on validation set with mixed precision."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="  Evaluating", leave=False):
                original_tokens, masked_tokens, mask = create_masked_batch(
                    batch, self.alphabet, self.mask_prob
                )
                
                original_tokens = original_tokens.to(self.device)
                masked_tokens = masked_tokens.to(self.device)
                mask = mask.to(self.device)
                
                try:
                    if self.use_amp:
                        with autocast():
                            results = self.model(masked_tokens, repr_layers=[])
                            logits = results["logits"]
                            
                            masked_logits = logits[mask]
                            masked_labels = original_tokens[mask]
                            
                            if masked_logits.numel() > 0:
                                loss = criterion(masked_logits, masked_labels)
                                total_loss += loss.item()
                                num_batches += 1
                    else:
                        results = self.model(masked_tokens, repr_layers=[])
                        logits = results["logits"]
                        
                        masked_logits = logits[mask]
                        masked_labels = original_tokens[mask]
                        
                        if masked_logits.numel() > 0:
                            loss = criterion(masked_logits, masked_labels)
                            total_loss += loss.item()
                            num_batches += 1
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def save_model(self, path: str):
        """Save the fine-tuned model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
        }, path)
        print(f"  Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Model loaded from {path}")
    
    def extract_embeddings(
        self,
        tsv_file: str,
        output_dir: str,
        batch_size: int = 8,
        repr_layer: int = -1,
        save_matrices: bool = True,
        save_vectors: bool = True,
    ):
        """
        Extract embeddings from fine-tuned model.
        
        Args:
            tsv_file: Path to TSV file with seq_id and seq columns
            output_dir: Output directory for embeddings
            batch_size: Batch size for extraction
            repr_layer: Which layer to extract (-1 = last layer)
            save_matrices: Save per-token matrices
            save_vectors: Save mean-pooled vectors
        """
        
        self.model.eval()
        
        # Read sequences
        df = pd.read_csv(tsv_file, sep='\t', compression='gzip' if tsv_file.endswith('.gz') else None)
        unique_seqs = df[['seq_id', 'seq']].drop_duplicates()
        sequences = [(row['seq_id'], row['seq']) for _, row in unique_seqs.iterrows()]
        
        if save_matrices:
            matrix_dir = os.path.join(output_dir, "protein_matrices")
            os.makedirs(matrix_dir, exist_ok=True)
        
        if save_vectors:
            vector_dir = os.path.join(output_dir, "protein_embeddings")
            os.makedirs(vector_dir, exist_ok=True)
        
        print(f"\n  Extracting embeddings for {len(sequences)} sequences...")
        print(f"  Output directory: {output_dir}")
        
        # Get repr layer index
        if repr_layer == -1:
            repr_layer = self.model.num_layers
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Extracting"):
            batch = sequences[i:i+batch_size]
            
            # Convert batch
            batch_labels, batch_strs, batch_tokens = self.batch_converter(batch)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[repr_layer])
                embeddings = results["representations"][repr_layer]
            
            # Remove special tokens and extract
            for j, (seq_id, seq) in enumerate(batch):
                seq_len = len(seq)
                # Remove <cls> and <eos> tokens
                seq_embedding = embeddings[j, 1:seq_len+1].cpu().numpy()
                
                if save_matrices:
                    matrix_path = os.path.join(matrix_dir, f"{seq_id}_matrix.npy")
                    np.save(matrix_path, seq_embedding)
                
                if save_vectors:
                    vector_path = os.path.join(vector_dir, f"{seq_id}_embedding.npy")
                    mean_embedding = seq_embedding.mean(axis=0)
                    np.save(vector_path, mean_embedding)
        
        print(f"  ✓ Embeddings extracted successfully")
