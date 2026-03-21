#!/usr/bin/env python3
"""
Standalone test for attention-matrix module.

This script tests the attention-matrix module independently, using pre-computed
embeddings and splits from the existing pipeline results.

Usage:
    python tests/test_attention_matrix_standalone.py

Author: DockTKinase Team
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention_matrix import (
    AttentionMatrixPipeline,
    AttentionMatrixConfig,
    AttentionTrainer,
    AttentionEvaluator,
    LeakageAwareSplitter,
    SimpleSplitter,
    create_dataloaders,
    ImprovedCrossAttentionModel,
    CrossAttentionModel
)


def load_existing_data(base_path: str, data_file: str):
    """Load pre-computed embeddings and splits from existing pipeline results."""
    
    base = Path(base_path)
    
    # Load the original dataset to get sample mapping
    print(f"[INFO] Loading original dataset: {data_file}")
    df = pd.read_csv(data_file, sep='\t')
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Get column names (based on kinase_non_human_compounds.tsv schema)
    protein_col = 'seq_id'  # UniProt ID / protein sequence identifier
    ligand_col = 'chembl_id'  # Compound ChEMBL ID
    
    # Load protein embeddings
    print("[INFO] Loading protein embeddings...")
    protein_matrices_dir = base / 'protein_matrices'
    protein_embeddings_dict = {}
    
    for f in protein_matrices_dir.glob('*_embedding.npy'):
        protein_id = f.stem.replace('_embedding', '')
        emb = np.load(f)
        protein_embeddings_dict[protein_id] = emb.flatten()
    
    print(f"[INFO] Loaded {len(protein_embeddings_dict)} protein embeddings")
    
    # Load ligand embeddings
    print("[INFO] Loading ligand embeddings...")
    ligand_matrices_dir = base / 'ligand_matrices'
    ligand_embeddings_dict = {}
    
    for f in ligand_matrices_dir.glob('*_embedding.npy'):
        ligand_id = f.stem.replace('_embedding', '')
        ligand_embeddings_dict[ligand_id] = np.load(f).flatten()
    
    print(f"[INFO] Loaded {len(ligand_embeddings_dict)} ligand embeddings")
    
    # Build embedding matrix aligned with dataset
    print("[INFO] Building embedding matrix...")
    protein_embs = []
    ligand_embs = []
    binary_labels = []  # For classification (active/inactive)
    regression_targets = []  # For regression (pChEMBL values)
    valid_indices = []
    protein_ids_list = []
    
    # Determine label threshold (pChEMBL >= 7.0 means IC50 <= 100nM = active)
    threshold = 7.0
    
    for idx, row in df.iterrows():
        prot_id = str(row[protein_col])
        lig_id = str(row[ligand_col])
        
        if prot_id in protein_embeddings_dict and lig_id in ligand_embeddings_dict:
            protein_embs.append(protein_embeddings_dict[prot_id])
            ligand_embs.append(ligand_embeddings_dict[lig_id])
            
            # Get pChEMBL value for regression
            pchembl = row.get('pchembl_value', 5.0)
            if pd.isna(pchembl):
                pchembl = 5.0  # Default if missing
            pchembl = float(pchembl)
            
            # Binary label for classification
            binary_label = 1 if pchembl >= threshold else 0
            binary_labels.append(binary_label)
            
            # Continuous target for regression
            regression_targets.append(pchembl)
            
            valid_indices.append(idx)
            protein_ids_list.append(prot_id)
    
    protein_embeddings = np.array(protein_embs)
    ligand_embeddings = np.array(ligand_embs)
    labels = np.array(binary_labels)  # Binary labels for classification
    regression_values = np.array(regression_targets)  # pChEMBL for regression
    protein_ids = np.array(protein_ids_list)
    
    print(f"[INFO] Valid samples: {len(valid_indices)} / {len(df)}")
    print(f"[INFO] Protein embeddings shape: {protein_embeddings.shape}")
    print(f"[INFO] Ligand embeddings shape: {ligand_embeddings.shape}")
    print(f"[INFO] Classification labels: {labels.sum()} active, {len(labels) - labels.sum()} inactive")
    print(f"[INFO] Regression targets (pChEMBL): min={regression_values.min():.2f}, max={regression_values.max():.2f}, mean={regression_values.mean():.2f}")
    
    # Concatenate embeddings
    embedding_matrix = np.concatenate([protein_embeddings, ligand_embeddings], axis=1)
    print(f"[INFO] Embedding matrix shape: {embedding_matrix.shape}")
    
    # Load splits
    splits_path = base / 'splits_leakage_aware'
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_path}")
    
    train_idx = np.load(splits_path / 'train_idx.npy')
    val_idx = np.load(splits_path / 'val_idx.npy')
    test_idx = np.load(splits_path / 'test_idx.npy')
    
    # Remap indices to valid samples
    valid_set = set(valid_indices)
    idx_remap = {old: new for new, old in enumerate(valid_indices)}
    
    train_idx_new = np.array([idx_remap[i] for i in train_idx if i in valid_set])
    val_idx_new = np.array([idx_remap[i] for i in val_idx if i in valid_set])
    test_idx_new = np.array([idx_remap[i] for i in test_idx if i in valid_set])
    
    print(f"[INFO] Remapped splits: train={len(train_idx_new)}, val={len(val_idx_new)}, test={len(test_idx_new)}")
    
    # Verify 80/10/10 split
    total = len(train_idx_new) + len(val_idx_new) + len(test_idx_new)
    print(f"[INFO] Split ratios: train={100*len(train_idx_new)/total:.1f}%, val={100*len(val_idx_new)/total:.1f}%, test={100*len(test_idx_new)/total:.1f}%")
    
    # Load metadata
    metadata_path = splits_path / 'split_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"[INFO] Split metadata: {metadata}")
    
    return embedding_matrix, labels, regression_values, train_idx_new, val_idx_new, test_idx_new, protein_ids


def extract_features_and_labels(embedding_matrix: np.ndarray, 
                                 interaction_labels: np.ndarray,
                                 protein_dim: int = 320,
                                 ligand_dim: int = 768):
    """Extract protein embeddings, ligand embeddings, and binary labels."""
    
    # Split embedding matrix
    protein_embeddings = embedding_matrix[:, :protein_dim]
    ligand_embeddings = embedding_matrix[:, protein_dim:protein_dim + ligand_dim]
    
    print(f"[INFO] Protein embeddings: {protein_embeddings.shape}")
    print(f"[INFO] Ligand embeddings: {ligand_embeddings.shape}")
    
    # Extract binary labels (column 0 is typically the binary activity)
    if interaction_labels.ndim == 1:
        binary_labels = interaction_labels
    else:
        # Assume column 0 is binary activity label
        binary_labels = interaction_labels[:, 0]
    
    print(f"[INFO] Binary labels: {binary_labels.shape}")
    print(f"[INFO] Class distribution: {np.bincount(binary_labels.astype(int))}")
    
    return protein_embeddings, ligand_embeddings, binary_labels


def test_attention_matrix_training():
    """Test the attention-matrix module with pre-computed data."""
    
    import torch
    
    print("\n" + "="*80)
    print("ATTENTION-MATRIX MODULE TEST")
    print("="*80 + "\n")
    
    # Paths
    base_path = '/Users/sulfierry/docktkinase/results/kinase_non_human_full'
    data_file = '/Users/sulfierry/docktkinase/tests/datasets/kinase_non_human_compounds.tsv'
    output_path = '/Users/sulfierry/docktkinase/results/test_attention_matrix_standalone'
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    print("[STEP 1] Loading pre-computed data...")
    try:
        embedding_matrix, labels, regression_targets, train_idx, val_idx, test_idx, protein_ids = load_existing_data(
            base_path, data_file
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Please run the full pipeline first to generate embeddings and splits.")
        return False
    
    # Extract features
    print("\n[STEP 2] Extracting features...")
    protein_dim = 320
    ligand_dim = 768
    protein_emb = embedding_matrix[:, :protein_dim]
    ligand_emb = embedding_matrix[:, protein_dim:protein_dim + ligand_dim]
    
    print(f"  Protein embeddings: {protein_emb.shape}")
    print(f"  Ligand embeddings: {ligand_emb.shape}")
    print(f"  Classification labels: {np.bincount(labels.astype(int))}")
    print(f"  Regression targets (pChEMBL): range [{regression_targets.min():.2f}, {regression_targets.max():.2f}]")
    
    # Apply splits (80% train, 10% val, 10% test)
    print("\n[STEP 3] Applying splits (80/10/10)...")
    
    # Classification labels
    y_train_cls = labels[train_idx]
    y_val_cls = labels[val_idx]
    y_test_cls = labels[test_idx]
    
    # Regression targets (pChEMBL values)
    y_train_reg = regression_targets[train_idx]
    y_val_reg = regression_targets[val_idx]
    y_test_reg = regression_targets[test_idx]
    
    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"  Train: {len(train_idx)} samples ({100*len(train_idx)/total:.1f}%) - active: {y_train_cls.sum():.0f}")
    print(f"  Val:   {len(val_idx)} samples ({100*len(val_idx)/total:.1f}%) - active: {y_val_cls.sum():.0f}")
    print(f"  Test:  {len(test_idx)} samples ({100*len(test_idx)/total:.1f}%) - active: {y_test_cls.sum():.0f}")
    
    # Create configuration
    print("\n[STEP 4] Creating model configuration...")
    config = AttentionMatrixConfig(
        protein_dim=320,
        ligand_dim=768,
        hidden_dim=256,
        num_heads=8,
        dropout=0.2,
        learning_rate=1e-4,
        batch_size=32,
        epochs=10,  # Small for testing
        early_stopping_patience=5
    )
    
    # Save config
    config.save(os.path.join(output_path, 'config.json'))
    print(f"  Config saved to {output_path}/config.json")
    
    # Create simple numpy-based dataloaders
    print("\n[STEP 5] Creating dataloaders...")
    
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors - need to add sequence dimension for the model
    # Model expects (batch, seq_len, features)
    # Include both classification (binary) and regression (pChEMBL) targets
    train_prot_t = torch.from_numpy(protein_emb[train_idx]).float().unsqueeze(1)
    train_lig_t = torch.from_numpy(ligand_emb[train_idx]).float().unsqueeze(1)
    train_cls_t = torch.from_numpy(y_train_cls).float()  # Binary classification
    train_reg_t = torch.from_numpy(y_train_reg).float()  # pChEMBL regression
    
    val_prot_t = torch.from_numpy(protein_emb[val_idx]).float().unsqueeze(1)
    val_lig_t = torch.from_numpy(ligand_emb[val_idx]).float().unsqueeze(1)
    val_cls_t = torch.from_numpy(y_val_cls).float()
    val_reg_t = torch.from_numpy(y_val_reg).float()
    
    test_prot_t = torch.from_numpy(protein_emb[test_idx]).float().unsqueeze(1)
    test_lig_t = torch.from_numpy(ligand_emb[test_idx]).float().unsqueeze(1)
    test_cls_t = torch.from_numpy(y_test_cls).float()
    test_reg_t = torch.from_numpy(y_test_reg).float()
    
    # Datasets with both classification and regression targets
    train_dataset = TensorDataset(train_prot_t, train_lig_t, train_cls_t, train_reg_t)
    val_dataset = TensorDataset(val_prot_t, val_lig_t, val_cls_t, val_reg_t)
    test_dataset = TensorDataset(test_prot_t, test_lig_t, test_cls_t, test_reg_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n[STEP 6] Creating model...")
    model = ImprovedCrossAttentionModel(
        protein_dim=config.protein_dim,
        ligand_dim=config.ligand_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Move to device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model = model.to(device)
    
    # Manual training loop with combined classification + regression loss
    print("\n[STEP 7] Training model (Classification + Regression)...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    cls_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.HuberLoss(delta=1.0)
    
    # Loss weights (can be tuned)
    cls_weight = 0.5
    reg_weight = 0.5
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_reg_loss': [], 'val_reg_loss': []
    }
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses, train_cls_losses, train_reg_losses = [], [], []
        
        for batch_prot, batch_lig, batch_cls, batch_reg in train_loader:
            batch_prot = batch_prot.to(device)
            batch_lig = batch_lig.to(device)
            batch_cls = batch_cls.to(device)
            batch_reg = batch_reg.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_prot, batch_lig)
            
            # Combined loss
            cls_loss = cls_criterion(outputs['classification'], batch_cls)
            reg_loss = reg_criterion(outputs['regression'], batch_reg)
            loss = cls_weight * cls_loss + reg_weight * reg_loss
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_cls_losses.append(cls_loss.item())
            train_reg_losses.append(reg_loss.item())
        
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(np.mean(train_cls_losses))
        history['train_reg_loss'].append(np.mean(train_reg_losses))
        
        # Validation
        model.eval()
        val_losses, val_cls_losses, val_reg_losses = [], [], []
        
        with torch.no_grad():
            for batch_prot, batch_lig, batch_cls, batch_reg in val_loader:
                batch_prot = batch_prot.to(device)
                batch_lig = batch_lig.to(device)
                batch_cls = batch_cls.to(device)
                batch_reg = batch_reg.to(device)
                
                outputs = model(batch_prot, batch_lig)
                cls_loss = cls_criterion(outputs['classification'], batch_cls)
                reg_loss = reg_criterion(outputs['regression'], batch_reg)
                loss = cls_weight * cls_loss + reg_weight * reg_loss
                
                val_losses.append(loss.item())
                val_cls_losses.append(cls_loss.item())
                val_reg_losses.append(reg_loss.item())
        
        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)
        history['val_cls_loss'].append(np.mean(val_cls_losses))
        history['val_reg_loss'].append(np.mean(val_reg_losses))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pt'))
        
        print(f"  Epoch {epoch+1}/{config.epochs}: loss={train_loss:.4f}, val_loss={val_loss:.4f} "
              f"(cls: {history['train_cls_loss'][-1]:.4f}/{history['val_cls_loss'][-1]:.4f}, "
              f"reg: {history['train_reg_loss'][-1]:.4f}/{history['val_reg_loss'][-1]:.4f})")
    
    print(f"\n  Training completed!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")
    
    # Evaluate on test set
    print("\n[STEP 8] Evaluating model on test set...")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_path, 'best_model.pt')))
    model.eval()
    
    # Import all metrics
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
        matthews_corrcoef, balanced_accuracy_score, average_precision_score,
        r2_score, mean_squared_error, mean_absolute_error
    )
    from scipy.stats import pearsonr, spearmanr
    
    # Collect predictions
    all_cls_preds = []
    all_cls_probs = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_targets = []
    
    with torch.no_grad():
        for batch_prot, batch_lig, batch_cls, batch_reg in test_loader:
            batch_prot = batch_prot.to(device)
            batch_lig = batch_lig.to(device)
            
            outputs = model(batch_prot, batch_lig)
            
            # Classification outputs
            cls_probs = torch.sigmoid(outputs['classification']).cpu().numpy()
            cls_preds = (cls_probs >= 0.5).astype(int)
            
            all_cls_probs.extend(cls_probs.tolist())
            all_cls_preds.extend(cls_preds.tolist())
            all_cls_labels.extend(batch_cls.numpy().tolist())
            
            # Regression outputs
            reg_preds = outputs['regression'].cpu().numpy()
            all_reg_preds.extend(reg_preds.tolist())
            all_reg_targets.extend(batch_reg.numpy().tolist())
    
    # Convert to numpy arrays
    all_cls_preds = np.array(all_cls_preds)
    all_cls_probs = np.array(all_cls_probs)
    all_cls_labels = np.array(all_cls_labels)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_targets = np.array(all_reg_targets)
    
    # Calculate classification metrics
    print("\n  === CLASSIFICATION METRICS (Binary: Active/Inactive) ===")
    cls_metrics = {
        'accuracy': accuracy_score(all_cls_labels, all_cls_preds),
        'balanced_accuracy': balanced_accuracy_score(all_cls_labels, all_cls_preds),
        'roc_auc': roc_auc_score(all_cls_labels, all_cls_probs),
        'average_precision': average_precision_score(all_cls_labels, all_cls_probs),
        'f1': f1_score(all_cls_labels, all_cls_preds),
        'precision': precision_score(all_cls_labels, all_cls_preds),
        'recall': recall_score(all_cls_labels, all_cls_preds),
        'mcc': matthews_corrcoef(all_cls_labels, all_cls_preds)
    }
    
    for key, value in cls_metrics.items():
        print(f"    {key}: {value:.4f}")
    
    # Calculate regression metrics
    print("\n  === REGRESSION METRICS (pChEMBL prediction) ===")
    
    # R² (coefficient of determination)
    r2 = r2_score(all_reg_targets, all_reg_preds)
    
    # Pearson correlation coefficient
    pearson_r, pearson_p = pearsonr(all_reg_targets, all_reg_preds)
    
    # Spearman rank correlation coefficient
    spearman_r, spearman_p = spearmanr(all_reg_targets, all_reg_preds)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(all_reg_targets, all_reg_preds))
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(all_reg_targets, all_reg_preds)
    
    reg_metrics = {
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p_value': spearman_p,
        'rmse': rmse,
        'mae': mae
    }
    
    print(f"    R² (coefficient of determination): {r2:.4f}")
    print(f"    Pearson r: {pearson_r:.4f} (p-value: {pearson_p:.2e})")
    print(f"    Spearman ρ: {spearman_r:.4f} (p-value: {spearman_p:.2e})")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    # Combine all metrics
    all_metrics = {
        'classification': cls_metrics,
        'regression': reg_metrics,
        'training_history': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss']),
            'epochs_trained': config.epochs
        },
        'data_split': {
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
            'train_ratio': len(train_idx) / total,
            'val_ratio': len(val_idx) / total,
            'test_ratio': len(test_idx) / total
        },
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(device)
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(output_path, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert_to_serializable(all_metrics), f, indent=2)
    print(f"\n  All metrics saved to {metrics_path}")
    
    # Save training history
    history_path = os.path.join(output_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to {history_path}")
    
    # Save model
    model_path = os.path.join(output_path, 'best_model.pt')
    print(f"  Model saved to {model_path}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print("\n=== KEY METRICS SUMMARY ===")
    print("\nClassification (Active/Inactive):")
    print(f"  - Accuracy:     {cls_metrics['accuracy']:.4f}")
    print(f"  - ROC-AUC:      {cls_metrics['roc_auc']:.4f}")
    print(f"  - F1 Score:     {cls_metrics['f1']:.4f}")
    print(f"  - MCC:          {cls_metrics['mcc']:.4f}")
    print("\nRegression (pChEMBL):")
    print(f"  - R²:           {reg_metrics['r2']:.4f}")
    print(f"  - Pearson r:    {reg_metrics['pearson_r']:.4f}")
    print(f"  - Spearman ρ:   {reg_metrics['spearman_r']:.4f}")
    print(f"  - RMSE:         {reg_metrics['rmse']:.4f}")
    print(f"  - MAE:          {reg_metrics['mae']:.4f}")
    
    return True


def test_module_imports():
    """Test that all module imports work correctly."""
    
    print("\n[TEST] Verifying module imports...")
    
    try:
        from attention_matrix import (
            AttentionMatrixPipeline,
            AttentionMatrixConfig,
            AttentionTrainer,
            AttentionEvaluator,
            AttentionAnalyzer,
            LeakageAwareSplitter,
            SimpleSplitter,
            CrossAttentionModel,
            ImprovedCrossAttentionModel,
            ProteinLigandDataset,
            create_dataloaders,
            __version__
        )
        print(f"  ✓ All imports successful (version {__version__})")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_config_serialization():
    """Test configuration save/load."""
    
    print("\n[TEST] Testing configuration serialization...")
    
    import tempfile
    
    config = AttentionMatrixConfig(
        protein_dim=320,
        ligand_dim=768,
        hidden_dim=256,
        num_heads=8
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config.save(f.name)
        loaded = AttentionMatrixConfig.load(f.name)
        
        assert loaded.protein_dim == config.protein_dim
        assert loaded.ligand_dim == config.ligand_dim
        assert loaded.hidden_dim == config.hidden_dim
        assert loaded.num_heads == config.num_heads
        
        os.unlink(f.name)
    
    print("  ✓ Configuration serialization works correctly")
    return True


def test_model_forward_pass():
    """Test model forward pass."""
    
    print("\n[TEST] Testing model forward pass...")
    
    import torch
    
    # Create model
    model = ImprovedCrossAttentionModel(
        protein_dim=320,
        ligand_dim=768,
        hidden_dim=256,
        num_heads=8
    )
    
    # Create dummy input (batch, seq_len, features)
    batch_size = 4
    seq_len = 1  # Single embedding per sample
    protein_emb = torch.randn(batch_size, seq_len, 320)
    ligand_emb = torch.randn(batch_size, seq_len, 768)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(protein_emb, ligand_emb, return_attention=True)
    
    assert 'classification' in output
    assert 'regression' in output
    assert 'attention' in output
    
    print(f"  ✓ Forward pass successful")
    print(f"    Classification shape: {output['classification'].shape}")
    print(f"    Regression shape: {output['regression'].shape}")
    print(f"    Attention shape: {output['attention'].shape if output['attention'] is not None else 'N/A'}")
    
    return True


def test_splitter():
    """Test the splitter implementations."""
    
    print("\n[TEST] Testing splitter implementations...")
    
    # Create dummy data as DataFrame
    n_samples = 100
    df = pd.DataFrame({
        'protein_id': [f'P{i % 10}' for i in range(n_samples)],  # 10 unique proteins
        'ligand_id': [f'L{i}' for i in range(n_samples)],
        'is_active': np.random.randint(0, 2, n_samples)
    })
    
    # Test SimpleSplitter
    splitter = SimpleSplitter(
        test_size=0.1,
        val_size=0.1,
        random_state=42
    )
    
    train_idx, val_idx, test_idx = splitter.split(
        df=df,
        stratify_col='is_active'
    )
    
    # Verify splits
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == n_samples, f"Split indices don't cover all samples: {total} != {n_samples}"
    
    # Verify no overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set & test_set) == 0, "Overlap between val and test!"
    
    print(f"  ✓ SimpleSplitter works correctly")
    print(f"    Train samples: {len(train_idx)}")
    print(f"    Val samples: {len(val_idx)}")
    print(f"    Test samples: {len(test_idx)}")
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ATTENTION-MATRIX MODULE - STANDALONE TESTS")
    print("="*80)
    
    # Run unit tests
    tests_passed = 0
    tests_total = 4
    
    if test_module_imports():
        tests_passed += 1
    
    if test_config_serialization():
        tests_passed += 1
    
    if test_model_forward_pass():
        tests_passed += 1
    
    if test_splitter():
        tests_passed += 1
    
    print(f"\n[SUMMARY] Unit tests: {tests_passed}/{tests_total} passed")
    
    # Run full training test if all unit tests pass
    if tests_passed == tests_total:
        print("\n" + "-"*80)
        print("Running full training test...")
        print("-"*80)
        
        success = test_attention_matrix_training()
        
        if success:
            print("\n[FINAL] All tests passed! ✓")
        else:
            print("\n[FINAL] Training test failed. Check the logs above.")
    else:
        print("\n[FINAL] Some unit tests failed. Fix them before running training test.")
