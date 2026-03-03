"""
MolFormer Fine-tuning Module for Level 4
Fine-tunes MolFormer models on kinase ligand data using masked language modeling.
Implements memory-efficient training with gradient checkpointing and mixed precision.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# PyTorch 2.0+ uses torch.amp instead of torch.cuda.amp
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None
from typing import List, Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path
import numpy as np
import gc


class MoleculeMLMDataset(Dataset):
    """Dataset for masked language modeling on SMILES sequences."""
    
    # MolFormer vocabulary (simplified - actual MolFormer uses BPE)
    VOCAB = {
        '<pad>': 0, '<cls>': 1, '<sep>': 2, '<unk>': 3, '<mask>': 4,
        'C': 5, 'c': 6, 'N': 7, 'n': 8, 'O': 9, 'o': 10, 'S': 11, 's': 12,
        'F': 13, 'Cl': 14, 'Br': 15, 'I': 16, 'P': 17, 'B': 18,
        '(': 19, ')': 20, '[': 21, ']': 22, '=': 23, '#': 24, '@': 25,
        '+': 26, '-': 27, '/': 28, '\\': 29, '.': 30,
        '0': 31, '1': 32, '2': 33, '3': 34, '4': 35, '5': 36, '6': 37, '7': 38, '8': 39, '9': 40,
        'H': 41, '%': 42
    }
    
    def __init__(
        self,
        smiles_list: List[str],
        chembl_ids: List[str],
        max_length: int = 202,  # MolFormer default
        mask_prob: float = 0.15
    ):
        self.smiles_list = smiles_list
        self.chembl_ids = chembl_ids
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.vocab_size = len(self.VOCAB)
        
    def __len__(self):
        return len(self.smiles_list)
    
    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize SMILES string."""
        tokens = [self.VOCAB['<cls>']]
        i = 0
        while i < len(smiles) and len(tokens) < self.max_length - 1:
            # Check for two-character tokens
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.VOCAB:
                    tokens.append(self.VOCAB[two_char])
                    i += 2
                    continue
            # Single character
            char = smiles[i]
            if char in self.VOCAB:
                tokens.append(self.VOCAB[char])
            else:
                tokens.append(self.VOCAB['<unk>'])
            i += 1
        tokens.append(self.VOCAB['<sep>'])
        return tokens
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        chembl_id = self.chembl_ids[idx]
        
        # Tokenize
        tokens = self.tokenize(smiles)
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.VOCAB['<pad>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length-1] + [self.VOCAB['<sep>']]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'chembl_id': chembl_id,
            'smiles': smiles
        }


def collate_mlm_batch(batch):
    """Collate batch for MLM training."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    chembl_ids = [item['chembl_id'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    return {
        'input_ids': input_ids,
        'chembl_ids': chembl_ids,
        'smiles': smiles
    }


class MolFormerFinetuner:
    """Fine-tunes MolFormer models on kinase ligands using MLM."""
    
    def __init__(
        self,
        model_path: str = "ibm/MoLFormer-XL-both-10pct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mask_prob: float = 0.15,
        use_amp: bool = True,  # Mixed precision for memory efficiency
    ):
        self.device = device
        self.mask_prob = mask_prob
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None
        
        print(f"  Loading MolFormer model...")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = self.model.to(device)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  ✓ Gradient checkpointing enabled")
            
            print(f"  Model loaded on {device}")
            n_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"  Total parameters: {n_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            self.model_type = "transformers"
            
        except Exception as e:
            print(f"  Warning: Could not load from transformers: {e}")
            print(f"  Falling back to local MolFormer...")
            self._load_local_molformer(model_path)
            self.model_type = "local"
    
    def _load_local_molformer(self, model_path: str):
        """Load MolFormer from local files."""
        molformer_path = Path(__file__).parent.parent.parent / "model_files" / "Molformer"
        if molformer_path.exists():
            sys.path.insert(0, str(molformer_path))
            # Import local molformer implementation
            # This is a fallback - actual implementation depends on local setup
            raise NotImplementedError("Local MolFormer loading not yet implemented")
        else:
            raise FileNotFoundError(f"MolFormer not found at {molformer_path}")
    
    def prepare_data(
        self,
        train_tsv: str,
        val_tsv: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 202,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare dataloaders from TSV files."""
        
        print(f"  Loading training data from {train_tsv}...")
        train_df = pd.read_csv(train_tsv, sep='\t', compression='gzip' if train_tsv.endswith('.gz') else None)
        
        # Get unique SMILES
        unique_mols = train_df[['chembl_id', 'smiles']].drop_duplicates()
        train_smiles = unique_mols['smiles'].tolist()
        train_ids = unique_mols['chembl_id'].tolist()
        
        print(f"  Training set: {len(train_smiles)} unique molecules")
        
        train_dataset = MoleculeMLMDataset(
            train_smiles, train_ids, max_length, self.mask_prob
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_mlm_batch,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_tsv:
            print(f"  Loading validation data from {val_tsv}...")
            val_df = pd.read_csv(val_tsv, sep='\t', compression='gzip' if val_tsv.endswith('.gz') else None)
            unique_val_mols = val_df[['chembl_id', 'smiles']].drop_duplicates()
            val_smiles = unique_val_mols['smiles'].tolist()
            val_ids = unique_val_mols['chembl_id'].tolist()
            
            print(f"  Validation set: {len(val_smiles)} unique molecules")
            
            val_dataset = MoleculeMLMDataset(
                val_smiles, val_ids, max_length, self.mask_prob
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_mlm_batch,
                num_workers=4,
                pin_memory=True
            )
        
        return train_loader, val_loader
    
    def create_masked_batch(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked batch for MLM training.
        Returns: (original_ids, masked_ids, mask)
        """
        # Clone for masking
        masked_ids = input_ids.clone()
        
        # Create mask (ignore special tokens: <pad>=0, <cls>=1, <sep>=2)
        special_tokens_mask = (input_ids == 0) | (input_ids == 1) | (input_ids == 2)
        
        # Random masking
        mask = torch.rand(input_ids.shape, device=input_ids.device) < self.mask_prob
        mask = mask & ~special_tokens_mask
        
        # Apply masking (mask token = 4)
        masked_ids[mask] = 4
        
        return input_ids, masked_ids, mask
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,  # Higher for memory efficiency
        save_path: Optional[str] = None,
        patience: int = 3,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model using masked language modeling.
        
        Uses:
        - Mixed precision training (FP16)
        - Gradient accumulation
        - Gradient checkpointing
        - Memory-efficient optimizer settings
        """
        
        self.model.train()
        
        # Memory-efficient optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 2362
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': []
        }
        
        print(f"\n  Starting MolFormer fine-tuning for {epochs} epochs (patience={patience})...")
        print(f"  Learning rate: {learning_rate}, Warmup steps: {warmup_steps}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"  Mixed precision (AMP): {self.use_amp}")
        
        global_step = 0
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
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
                input_ids = batch['input_ids'].to(self.device)
                
                # Create masked batch
                original_ids, masked_ids, mask = self.create_masked_batch(input_ids)
                
                try:
                    # Mixed precision forward pass
                    if self.use_amp:
                        with autocast(device_type="cuda"):
                            outputs = self.model(masked_ids)
                            
                            # Get logits (depends on model architecture)
                            if hasattr(outputs, 'logits'):
                                logits = outputs.logits
                            elif hasattr(outputs, 'last_hidden_state'):
                                # Need to project to vocabulary
                                hidden = outputs.last_hidden_state
                                if hasattr(self.model, 'lm_head'):
                                    logits = self.model.lm_head(hidden)
                                else:
                                    # Skip this batch if no lm_head
                                    continue
                            else:
                                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                            
                            # Compute loss on masked tokens
                            masked_logits = logits[mask]
                            masked_labels = original_ids[mask]
                            
                            if masked_logits.numel() > 0:
                                loss = criterion(masked_logits, masked_labels)
                            else:
                                continue
                        
                        # Scaled backward
                        loss = loss / gradient_accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        outputs = self.model(masked_ids)
                        
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                            if hasattr(self.model, 'lm_head'):
                                logits = self.model.lm_head(hidden)
                            else:
                                continue
                        else:
                            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        
                        masked_logits = logits[mask]
                        masked_labels = original_ids[mask]
                        
                        if masked_logits.numel() > 0:
                            loss = criterion(masked_logits, masked_labels)
                        else:
                            continue
                        
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                    
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    
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
            
            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            train_perplexity = np.exp(min(avg_train_loss, 10))  # Cap to prevent overflow
            
            history['train_loss'].append(avg_train_loss)
            history['learning_rate'].append(current_lr)
            history['perplexity'].append(train_perplexity)
            
            # Validation
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
                
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    epochs_without_improvement = 0
                    if save_path:
                        self.save_model(save_path)
                        print(f"  → Best model saved (train_loss={avg_train_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    print(f"  → No improvement for {epochs_without_improvement}/{patience} epochs")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs")
                break
            
            # Clear GPU cache between epochs
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="  Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                original_ids, masked_ids, mask = self.create_masked_batch(input_ids)
                
                try:
                    if self.use_amp:
                        with autocast(device_type="cuda"):
                            outputs = self.model(masked_ids)
                            if hasattr(outputs, 'logits'):
                                logits = outputs.logits
                            elif hasattr(outputs, 'last_hidden_state'):
                                hidden = outputs.last_hidden_state
                                if hasattr(self.model, 'lm_head'):
                                    logits = self.model.lm_head(hidden)
                                else:
                                    continue
                            else:
                                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                            
                            masked_logits = logits[mask]
                            masked_labels = original_ids[mask]
                            
                            if masked_logits.numel() > 0:
                                loss = criterion(masked_logits, masked_labels)
                                total_loss += loss.item()
                                num_batches += 1
                    else:
                        outputs = self.model(masked_ids)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                            if hasattr(self.model, 'lm_head'):
                                logits = self.model.lm_head(hidden)
                            else:
                                continue
                        else:
                            logits = outputs[0] if isinstance(outputs, tuple) else outputs
                        
                        masked_logits = logits[mask]
                        masked_labels = original_ids[mask]
                        
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
        
        if self.model_type == "transformers":
            self.model.save_pretrained(path)
            if hasattr(self, 'tokenizer'):
                self.tokenizer.save_pretrained(path)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, path)
        
        print(f"  Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model."""
        if self.model_type == "transformers":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
            self.model = self.model.to(self.device)
        else:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  Model loaded from {path}")
    
    def extract_embeddings(
        self,
        tsv_file: str,
        output_dir: str,
        batch_size: int = 32,
        save_matrices: bool = True,
        save_vectors: bool = True,
    ):
        """
        Extract embeddings from fine-tuned model.
        
        Args:
            tsv_file: Path to TSV file with chembl_id and smiles columns
            output_dir: Output directory for embeddings
            batch_size: Batch size for extraction
            save_matrices: Save per-token matrices
            save_vectors: Save mean-pooled vectors
        """
        
        self.model.eval()
        
        # Read molecules
        df = pd.read_csv(tsv_file, sep='\t', compression='gzip' if tsv_file.endswith('.gz') else None)
        unique_mols = df[['chembl_id', 'smiles']].drop_duplicates()
        
        if save_matrices:
            matrix_dir = os.path.join(output_dir, "ligand_matrices_finetuned")
            os.makedirs(matrix_dir, exist_ok=True)
        
        if save_vectors:
            vector_dir = os.path.join(output_dir, "ligand_embeddings_finetuned")
            os.makedirs(vector_dir, exist_ok=True)
        
        print(f"\n  Extracting embeddings for {len(unique_mols)} molecules...")
        print(f"  Output directory: {output_dir}")
        
        # Process in batches
        for i in tqdm(range(0, len(unique_mols), batch_size), desc="  Extracting"):
            batch_df = unique_mols.iloc[i:i+batch_size]
            smiles_list = batch_df['smiles'].tolist()
            chembl_ids = batch_df['chembl_id'].tolist()
            
            # Tokenize
            if self.model_type == "transformers":
                inputs = self.tokenizer(
                    smiles_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=202
                ).to(self.device)
                
                with torch.no_grad():
                    if self.use_amp:
                        with autocast(device_type="cuda"):
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    else:
                        embeddings = outputs[0]
                
                attention_mask = inputs['attention_mask']
            else:
                # Local model handling
                continue
            
            # Save embeddings
            for j, chembl_id in enumerate(chembl_ids):
                # Get valid tokens (non-padding)
                valid_len = attention_mask[j].sum().item()
                seq_embedding = embeddings[j, :valid_len].cpu().numpy()
                
                if save_matrices:
                    matrix_path = os.path.join(matrix_dir, f"{chembl_id}_matrix.npy")
                    np.save(matrix_path, seq_embedding)
                
                if save_vectors:
                    vector_path = os.path.join(vector_dir, f"{chembl_id}_embedding.npy")
                    mean_embedding = seq_embedding.mean(axis=0)
                    np.save(vector_path, mean_embedding)
        
        print(f"  ✓ Embeddings extracted successfully")
