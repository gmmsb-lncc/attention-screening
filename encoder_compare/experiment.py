"""Main experiment runner for encoder comparison."""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from src.classifier.utils.matrix_dataloader import create_matrix_dataloader

from .config import TrainingConfig, ENCODER_TYPES, EMBEDDING_BASE_PATH, DATASET_PATHS
from .models import FlexibleCrossAttentionModel
from .data import get_scenarios
from .training import train_model, evaluate
from .utils import get_device, save_checkpoint, load_checkpoint, get_checkpoint_path


def run_encoder_comparison(
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    config: TrainingConfig,
    seeds: List[int] = [42],
    resume: bool = True
) -> Optional[Dict]:
    """
    Run comparison of all encoder types across multiple seeds.

    Args:
        embedding_name: Name of embedding model
        dataset_type: Type of dataset ('human' or 'non_human')
        output_dir: Output directory for results
        config: Training configuration
        seeds: List of random seeds
        resume: Whether to resume from checkpoint

    Returns:
        Results dictionary or None if failed
    """
    # Initialize experiment
    checkpoint_path, all_results = _initialize_experiment(
        output_dir, embedding_name, dataset_type, resume
    )

    # Load and validate data
    df, protein_matrix_dir, ligand_matrix_dir = _load_data(
        embedding_name, dataset_type
    )
    if df is None:
        return None

    device = get_device()
    print(f"  Device: {device}")

    # Run experiments for each encoder
    scenarios = get_scenarios()
    for encoder_type in ENCODER_TYPES:
        if _should_skip_encoder(encoder_type, all_results):
            continue

        _run_encoder_experiments(
            encoder_type, scenarios, df,
            protein_matrix_dir, ligand_matrix_dir,
            config, seeds, device,
            all_results, checkpoint_path
        )

    # Mark as completed
    save_checkpoint(checkpoint_path, None, None, None, None, all_results, completed=True)
    print(f"\n✅ All experiments completed! Final checkpoint saved.")

    return all_results


def _initialize_experiment(
    output_dir: str,
    embedding_name: str,
    dataset_type: str,
    resume: bool
) -> tuple:
    """Initialize experiment and check for existing checkpoints."""
    checkpoint_path = get_checkpoint_path(output_dir, embedding_name, dataset_type)
    checkpoint = load_checkpoint(checkpoint_path) if resume else None

    if checkpoint is not None and checkpoint.get('completed', False):
        print(f"\n  ✓ Found completed checkpoint: {checkpoint_path}")
        print(f"    Completed at: {checkpoint['timestamp']}")
        return checkpoint_path, checkpoint['results']

    if checkpoint is not None and not checkpoint.get('completed', False):
        print(f"\n  📂 Resuming from checkpoint: {checkpoint_path}")
        print(f"    Last saved: {checkpoint['timestamp']}")
        all_results = checkpoint.get('results', {})
    else:
        print(f"\n  Starting fresh comparison...")
        all_results = {}

    return checkpoint_path, all_results


def _load_data(embedding_name: str, dataset_type: str) -> tuple:
    """Load dataset and validate paths."""
    # Build paths
    embedding_dir = os.path.join(
        EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
        embedding_name, 'build'
    )
    protein_matrix_dir = os.path.join(embedding_dir, 'protein_matrices')
    ligand_matrix_dir = os.path.join(embedding_dir, 'ligand_matrices')
    dataset_path = DATASET_PATHS[dataset_type]

    # Validate paths
    if not os.path.exists(protein_matrix_dir):
        print(f"ERROR: {protein_matrix_dir} not found")
        return None, None, None

    if not os.path.exists(ligand_matrix_dir):
        print(f"ERROR: {ligand_matrix_dir} not found")
        return None, None, None

    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found")
        return None, None, None

    # Load data
    print(f"\nLoading dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path, sep='\t')
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return None, None, None

    df['label'] = (df['standard_value'] <= 1000).astype(int)
    df['seq_id'] = df['seq_id'].astype(str)
    print(f"  Total: {len(df)} samples")

    return df, protein_matrix_dir, ligand_matrix_dir


def _should_skip_encoder(encoder_type: str, all_results: Dict) -> bool:
    """Check if encoder should be skipped."""
    if encoder_type in all_results:
        print(f"\n{'='*60}")
        print(f"ENCODER: {encoder_type.upper()} - ✓ ALREADY COMPLETED (skipping)")
        print(f"{'='*60}")
        return True
    return False


def _run_encoder_experiments(
    encoder_type: str,
    scenarios: list,
    df: pd.DataFrame,
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    config: TrainingConfig,
    seeds: List[int],
    device: torch.device,
    all_results: Dict,
    checkpoint_path: str
) -> None:
    """Run all experiments for a single encoder."""
    print(f"\n{'='*60}")
    print(f"ENCODER: {encoder_type.upper()}")
    print(f"{'='*60}")

    encoder_results = {}

    for scenario_name, split_fn in scenarios:
        print(f"\n  Scenario: {scenario_name}")

        scenario_seed_results = _run_scenario_experiments(
            encoder_type, scenario_name, split_fn, df,
            protein_matrix_dir, ligand_matrix_dir,
            config, seeds, device,
            all_results, checkpoint_path
        )

        # Calculate statistics and save
        encoder_results[scenario_name] = _calculate_statistics(
            scenario_seed_results
        )
        all_results[encoder_type] = encoder_results

        # Print summary
        _print_scenario_summary(scenario_name, encoder_results[scenario_name])

    all_results[encoder_type] = encoder_results


def _run_scenario_experiments(
    encoder_type: str,
    scenario_name: str,
    split_fn: callable,
    df: pd.DataFrame,
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    config: TrainingConfig,
    seeds: List[int],
    device: torch.device,
    all_results: Dict,
    checkpoint_path: str
) -> Dict:
    """Run experiments for a scenario across multiple seeds."""
    scenario_seed_results = {}
    n_params = None

    for seed in seeds:
        # Check if already completed
        if _is_seed_completed(encoder_type, scenario_name, seed, all_results):
            scenario_seed_results[seed] = all_results[encoder_type][scenario_name]['seed_results'][seed]
            if n_params is None:
                n_params = scenario_seed_results[seed]['n_params']
            print(f"\n    Seed: {seed} - ✓ Already completed (skipping)")
            continue

        print(f"\n    Seed: {seed}")

        # Run single seed experiment
        test_metrics, n_params = _run_single_seed(
            encoder_type, split_fn, df, seed,
            protein_matrix_dir, ligand_matrix_dir,
            config, device, n_params
        )

        scenario_seed_results[seed] = {
            'accuracy': test_metrics['accuracy'],
            'mcc': test_metrics['mcc'],
            'auc': test_metrics['auc'],
            'f1': test_metrics['f1'],
            'n_params': n_params
        }

        # Save checkpoint after each seed
        _save_intermediate_checkpoint(
            checkpoint_path, encoder_type, scenario_name,
            scenario_seed_results, all_results, n_params
        )

    return scenario_seed_results


def _run_single_seed(
    encoder_type: str,
    split_fn: callable,
    df: pd.DataFrame,
    seed: int,
    protein_matrix_dir: str,
    ligand_matrix_dir: str,
    config: TrainingConfig,
    device: torch.device,
    n_params: Optional[int]
) -> tuple:
    """Run experiment for a single seed."""
    # Split data
    train_idx, val_idx, test_idx = split_fn(df, seed=seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"      Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Create dataloaders
    train_loader = create_matrix_dataloader(
        train_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=True,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )
    val_loader = create_matrix_dataloader(
        val_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=False,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )
    test_loader = create_matrix_dataloader(
        test_df, protein_matrix_dir, ligand_matrix_dir,
        batch_size=config.batch_size, shuffle=False,
        label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
    )

    # Create and train model
    model = FlexibleCrossAttentionModel(
        protein_dim=config.protein_dim,
        ligand_dim=config.ligand_dim,
        hidden_dim=config.hidden_dim,
        encoder_type=encoder_type,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout
    )

    if n_params is None:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      Parameters: {n_params:,}")

    print(f"      Training for {config.num_epochs} epochs...")
    model, history = train_model(model, train_loader, val_loader, config, device)

    # Evaluate
    test_metrics = evaluate(model, test_loader, device)
    print(f"      Test: MCC={test_metrics['mcc']:.4f}, AUC={test_metrics['auc']:.4f}, "
          f"F1={test_metrics['f1']:.4f}, Acc={test_metrics['accuracy']:.4f}")

    return test_metrics, n_params


def _is_seed_completed(
    encoder_type: str,
    scenario_name: str,
    seed: int,
    all_results: Dict
) -> bool:
    """Check if a seed experiment was already completed."""
    return (encoder_type in all_results and
            scenario_name in all_results[encoder_type] and
            'seed_results' in all_results[encoder_type][scenario_name] and
            seed in all_results[encoder_type][scenario_name]['seed_results'])


def _calculate_statistics(scenario_seed_results: Dict) -> Dict:
    """Calculate mean and std across seeds."""
    return {
        'seed_results': scenario_seed_results,
        'mean': {
            'accuracy': np.mean([r['accuracy'] for r in scenario_seed_results.values()]),
            'mcc': np.mean([r['mcc'] for r in scenario_seed_results.values()]),
            'auc': np.mean([r['auc'] for r in scenario_seed_results.values()]),
            'f1': np.mean([r['f1'] for r in scenario_seed_results.values()]),
        },
        'std': {
            'accuracy': np.std([r['accuracy'] for r in scenario_seed_results.values()]),
            'mcc': np.std([r['mcc'] for r in scenario_seed_results.values()]),
            'auc': np.std([r['auc'] for r in scenario_seed_results.values()]),
            'f1': np.std([r['f1'] for r in scenario_seed_results.values()]),
        },
        'n_params': list(scenario_seed_results.values())[0]['n_params']
    }


def _save_intermediate_checkpoint(
    checkpoint_path: str,
    encoder_type: str,
    scenario_name: str,
    scenario_seed_results: Dict,
    all_results: Dict,
    n_params: int
) -> None:
    """Save intermediate checkpoint after each seed."""
    encoder_results = all_results.get(encoder_type, {})
    encoder_results[scenario_name] = _calculate_statistics(scenario_seed_results)
    all_results[encoder_type] = encoder_results

    save_checkpoint(checkpoint_path, encoder_type, scenario_name, None,
                   None, all_results, completed=False)
    print(f"      💾 Checkpoint saved")


def _print_scenario_summary(scenario_name: str, scenario_results: Dict) -> None:
    """Print summary for a scenario."""
    print(f"\n    Summary for {scenario_name}:")
    print(f"      MCC: {scenario_results['mean']['mcc']:.4f} ± {scenario_results['std']['mcc']:.4f}")
    print(f"      AUC: {scenario_results['mean']['auc']:.4f} ± {scenario_results['std']['auc']:.4f}")
