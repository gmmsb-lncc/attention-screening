"""Main experiment runner for encoder comparison."""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from src.classifier.utils.matrix_dataloader import create_matrix_dataloader

from .config import TrainingConfig, ENCODER_TYPES, EMBEDDING_BASE_PATH, DATASET_PATHS, DEFAULT_AFFINITY_THRESHOLD
from .models.flexible_model import FlexibleCrossAttentionModel
from .data import get_scenarios
from .training import evaluate, EvaluationError
from .training.stable_trainer import train_model
from .utils import get_device, set_seed
from .utils.checkpoints import save_checkpoint, load_checkpoint, get_checkpoint_path


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
        embedding_name, dataset_type, config
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
            all_results, checkpoint_path,
            embedding_name  # Pass the embedding_name parameter
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


def _load_data(embedding_name: str, dataset_type: str, config: TrainingConfig = None) -> tuple:
    """
    Load dataset and validate paths.

    Args:
        embedding_name: Name of embedding model
        dataset_type: Type of dataset
        config: Training configuration (contains affinity threshold)

    Returns:
        Tuple of (dataframe, protein_matrix_dir, ligand_matrix_dir)
    """
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

    # Get threshold configuration
    threshold_config = (
        config.affinity_threshold if config else DEFAULT_AFFINITY_THRESHOLD
    )

    # Apply threshold to generate binary labels
    threshold = threshold_config.threshold_nm
    column = threshold_config.column_name
    label_col = threshold_config.label_column

    df[label_col] = (df[column] <= threshold).astype(int)
    df['seq_id'] = df['seq_id'].astype(str)

    # Report class distribution
    n_active = df[label_col].sum()
    n_inactive = len(df) - n_active
    print(f"  Total: {len(df)} samples")
    print(f"  Affinity threshold: {threshold} nM ({threshold/1000} μM)")
    print(f"  Class distribution: {n_active} active ({100*n_active/len(df):.1f}%), "
          f"{n_inactive} inactive ({100*n_inactive/len(df):.1f}%)")

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
    checkpoint_path: str,
    embedding_name: str = "unknown"  # Added embedding_name parameter
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
            all_results, checkpoint_path,
            embedding_name  # Pass the embedding_name parameter
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
    checkpoint_path: str,
    embedding_name: str = "unknown"  # Added embedding_name parameter
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
            config, device, n_params,
            embedding_name, scenario_name  # Pass the required parameters
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
    n_params: Optional[int],
    embedding_name: str = "unknown",  # Added embedding_name parameter
    scenario_name: str = "unknown"    # Added scenario_name parameter
) -> tuple:
    """Run experiment for a single seed."""
    # Split data
    train_idx, val_idx, test_idx = split_fn(df, seed=seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"      Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Set all random seeds for full reproducibility
    set_seed(seed, deterministic=True)

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
    # Generate a unique checkpoint path for this specific training run
    exp_id = f"{embedding_name}_{encoder_type}_{scenario_name}_{seed}"
    # Create a checkpoint path specific to this training run
    checkpoint_dir = './checkpoints'  # Use a default checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"train_{exp_id}_checkpoint.pt")

    model, history = train_model(model, train_loader, val_loader, config, device,
                                checkpoint_path=checkpoint_path, experiment_id=exp_id)

    # Evaluate with explicit error handling
    test_result = evaluate(model, test_loader, device, raise_on_invalid=False)

    if not test_result.is_valid:
        raise EvaluationError(
            f"Test evaluation failed for seed {seed}: {test_result.failure_reason}\n"
            f"This run cannot be included in results. "
            f"Consider lowering learning rate or checking data quality."
        )

    test_metrics = test_result.metrics
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
    """
    Calculate mean and sample standard deviation across seeds.

    Uses ddof=1 for sample standard deviation (Bessel's correction),
    which is appropriate when seeds represent a sample from the
    population of possible random initializations.

    Also computes 95% confidence interval when n >= 2.
    """
    n_seeds = len(scenario_seed_results)

    # Extract metrics arrays
    accuracies = np.array([r['accuracy'] for r in scenario_seed_results.values()])
    mccs = np.array([r['mcc'] for r in scenario_seed_results.values()])
    aucs = np.array([r['auc'] for r in scenario_seed_results.values()])
    f1s = np.array([r['f1'] for r in scenario_seed_results.values()])

    # Use ddof=1 for sample standard deviation (Bessel's correction)
    # This provides unbiased estimate of population variance
    ddof = 1 if n_seeds > 1 else 0

    stats = {
        'seed_results': scenario_seed_results,
        'n_seeds': n_seeds,
        'mean': {
            'accuracy': float(np.mean(accuracies)),
            'mcc': float(np.mean(mccs)),
            'auc': float(np.mean(aucs)),
            'f1': float(np.mean(f1s)),
        },
        'std': {
            'accuracy': float(np.std(accuracies, ddof=ddof)),
            'mcc': float(np.std(mccs, ddof=ddof)),
            'auc': float(np.std(aucs, ddof=ddof)),
            'f1': float(np.std(f1s, ddof=ddof)),
        },
        'n_params': list(scenario_seed_results.values())[0]['n_params']
    }

    # Add 95% CI if n >= 2 (requires scipy for t-distribution)
    if n_seeds >= 2:
        from scipy import stats as scipy_stats
        t_critical = scipy_stats.t.ppf(0.975, df=n_seeds - 1)
        sem = lambda arr: np.std(arr, ddof=1) / np.sqrt(n_seeds)

        stats['ci_95'] = {
            'accuracy': float(t_critical * sem(accuracies)),
            'mcc': float(t_critical * sem(mccs)),
            'auc': float(t_critical * sem(aucs)),
            'f1': float(t_critical * sem(f1s)),
        }
    else:
        # CI undefined for n=1
        stats['ci_95'] = {
            'accuracy': np.nan,
            'mcc': np.nan,
            'auc': np.nan,
            'f1': np.nan,
        }
        stats['_warning'] = "CI undefined: only 1 seed. Use >= 3 seeds for valid statistics."

    return stats


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
