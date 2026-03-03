"""
ESM-2 Fine-tuning Module for Kinase Domain Adaptation
Level 4: Fine-tune ESM-2 on kinase training data using MLM objective
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import gzip
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from datetime import datetime

# Add ESM to path
import sys
esm_path = Path(__file__).parent.parent.parent / "llm" / "ESM"
if str(esm_path) not in sys.path:
    sys.path.insert(0, str(esm_path))

import esm


class KinaseSequenceDataset(Dataset):
    """Dataset for kinase sequences (MLM fine-tuning)"""
    
    def __init__(self, sequences: List[Tuple[str, str]], alphabet, max_length: int = 1024):
        """
        Args:
            sequences: List of (seq_id, sequence) tuples
            alphabet: ESM alphabet
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_id, sequence = self.sequences[idx]
        # Truncate if needed
        if len(sequence) > self.max_length - 2:  # -2 for <cls> and <eos>
            sequence = sequence[:self.max_length - 2]
        return seq_id, sequence


def load_sequences_from_tsv(tsv_path: Path) -> List[Tuple[str, str]]:
    """Load unique protein sequences from TSV file"""
    if tsv_path.suffix == '.gz':
        df = pd.read_csv(tsv_path, sep='\t', compression='gzip')
    else:
        df = pd.read_csv(tsv_path, sep='\t')
    
    # Get unique sequences
    unique_seqs = df[['seq_id', 'seq']].drop_duplicates('seq_id')
    sequences = [(row['seq_id'], row['seq']) for _, row in unique_seqs.iterrows()]
    
    return sequences


def mask_tokens(tokens, alphabet, mask_prob=0.15):
    """
    Mask tokens for MLM objective (BERT-style masking)
    
    Args:
        tokens: Input token ids [batch, seq_len]
        alphabet: ESM alphabet
        mask_prob: Probability of masking each token
    
    Returns:
        masked_tokens: Tokens with some replaced by <mask>
        labels: Original tokens (for computing loss)
    """
    labels = tokens.clone()
    
    # Get special token ids
    mask_token_id = alphabet.mask_idx
    pad_token_id = alphabet.padding_idx
    cls_token_id = alphabet.cls_idx
    eos_token_id = alphabet.eos_idx
    
    # Create probability matrix
    probability_matrix = torch.full(labels.shape, mask_prob)
    
    # Don't mask special tokens
    special_tokens_mask = (
        (tokens == pad_token_id) | 
        (tokens == cls_token_id) | 
        (tokens == eos_token_id)
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # Get masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Set labels to -100 for non-masked tokens (ignored in loss)
    labels[~masked_indices] = -100
    
    # 80% of the time, replace with <mask>
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens = tokens.clone()
    tokens[indices_replaced] = mask_token_id
    
    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(len(alphabet), labels.shape, dtype=torch.long)
    # Don't use special tokens as random replacements
    random_tokens = random_tokens % (len(alphabet) - 4) + 4
    tokens[indices_random] = random_tokens[indices_random]
    
    # 10% of the time, keep original (already done)
    
    return tokens, labels


class ESMFinetuner:
    """Fine-tune ESM-2 models on kinase sequences"""
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: Optional[Path] = None
    ):
        """
        Args:
            model_name: ESM-2 model name
            device: Device to use
            output_dir: Directory to save fine-tuned model
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.output_dir = output_dir or Path("./checkpoints/finetuned_esm")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading ESM-2 model: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Get model dimensions
        self.repr_dim = self.model.embed_dim
        
        print(f"Model loaded. Embedding dim: {self.repr_dim}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def fine_tune(
        self,
        train_sequences: List[Tuple[str, str]],
        val_sequences: Optional[List[Tuple[str, str]]] = None,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        mask_prob: float = 0.15,
        gradient_accumulation_steps: int = 4,
        max_length: int = 1024,
        patience: int = 3
    ) -> Dict:
        """
        Fine-tune ESM-2 using masked language modeling
        
        Args:
            train_sequences: Training sequences
            val_sequences: Validation sequences (optional)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            mask_prob: Masking probability
            gradient_accumulation_steps: Gradient accumulation steps
            max_length: Maximum sequence length
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        print("\n" + "="*70)
        print("FINE-TUNING ESM-2 ON KINASE SEQUENCES")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Training sequences: {len(train_sequences)}")
        if val_sequences:
            print(f"Validation sequences: {len(val_sequences)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Mask probability: {mask_prob}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print("="*70 + "\n")
        
        # Create datasets
        train_dataset = KinaseSequenceDataset(train_sequences, self.alphabet, max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # ESM doesn't play well with multiprocessing
            collate_fn=self._collate_fn
        )
        
        val_loader = None
        if val_sequences:
            val_dataset = KinaseSequenceDataset(val_sequences, self.alphabet, max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self._collate_fn
            )
        
        # Setup training
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }
        
        steps_without_improvement = 0
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            # Training
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc="Training", unit="batch")
            for batch_idx, batch in enumerate(pbar):
                tokens = batch['tokens'].to(self.device)
                
                # Mask tokens
                masked_tokens, labels = mask_tokens(tokens, self.alphabet, mask_prob)
                
                # Forward pass
                results = self.model(masked_tokens, repr_layers=[])
                logits = results['logits']
                
                # Compute loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                # Update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                train_loss += loss.item() * gradient_accumulation_steps
                pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f"Train loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    pbar = tqdm(val_loader, desc="Validation", unit="batch")
                    for batch in pbar:
                        tokens = batch['tokens'].to(self.device)
                        masked_tokens, labels = mask_tokens(tokens, self.alphabet, mask_prob)
                        
                        results = self.model(masked_tokens, repr_layers=[])
                        logits = results['logits']
                        
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        val_loss += loss.item()
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                print(f"Val loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < history['best_val_loss']:
                    history['best_val_loss'] = avg_val_loss
                    history['best_epoch'] = epoch + 1
                    steps_without_improvement = 0
                    
                    # Save best model
                    self.save_model(suffix="best")
                    print(f"✓ New best model saved (val_loss={avg_val_loss:.4f})")
                else:
                    steps_without_improvement += 1
                    print(f"No improvement for {steps_without_improvement} epoch(s)")
                    
                    if steps_without_improvement >= patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break
        
        # Save final model
        self.save_model(suffix="final")
        
        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*70)
        print("FINE-TUNING COMPLETED")
        print("="*70)
        print(f"Best epoch: {history['best_epoch']}")
        print(f"Best val loss: {history['best_val_loss']:.4f}")
        print(f"Models saved to: {self.output_dir}")
        print("="*70 + "\n")
        
        return history
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader"""
        labels, strs, tokens = self.batch_converter(batch)
        return {'tokens': tokens, 'labels': labels, 'strs': strs}
    
    def save_model(self, suffix: str = ""):
        """Save fine-tuned model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_finetuned_{suffix}_{timestamp}.pt" if suffix else f"{self.model_name}_finetuned_{timestamp}.pt"
        save_path = self.output_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'alphabet': self.alphabet,
            'timestamp': timestamp
        }, save_path)
        
        return save_path
    
    def load_finetuned_model(self, checkpoint_path: Path):
        """Load fine-tuned model from checkpoint"""
        print(f"Loading fine-tuned model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Fine-tuned model loaded successfully!")
        
    def extract_embeddings(
        self,
        sequences: List[Tuple[str, str]],
        output_dir: Path,
        batch_size: int = 4,
        repr_layers: List[int] = None
    ):
        """
        Extract embeddings from fine-tuned model
        
        Args:
            sequences: List of (seq_id, sequence) tuples
            output_dir: Output directory for embeddings
            batch_size: Batch size
            repr_layers: Representation layers to extract (default: last layer)
        """
        if repr_layers is None:
            repr_layers = [self.model.num_layers]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        dataset = KinaseSequenceDataset(sequences, self.alphabet)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        
        print(f"\nExtracting embeddings for {len(sequences)} sequences...")
        print(f"Output: {output_dir}")
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                tokens = batch['tokens'].to(self.device)
                labels = batch['labels']
                
                results = self.model(tokens, repr_layers=repr_layers)
                
                # Extract per-token embeddings (last layer)
                embeddings = results['representations'][repr_layers[-1]]
                
                # Save each sequence
                for i, seq_id in enumerate(labels):
                    # Get actual sequence length (excluding padding, cls, eos)
                    seq_len = (tokens[i] != self.alphabet.padding_idx).sum().item()
                    
                    # Extract embedding (remove cls and eos tokens)
                    emb = embeddings[i, 1:seq_len-1].cpu().numpy()
                    
                    # Save as .npy
                    output_path = output_dir / f"{seq_id}_matrix.npy"
                    import numpy as np
                    np.save(output_path, emb)
        
        print(f"✓ Embeddings saved to {output_dir}\n")
