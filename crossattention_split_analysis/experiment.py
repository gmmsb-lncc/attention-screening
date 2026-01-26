"""Main experiment runner for CrossAttention split analysis."""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.classifier.models.cross_attention_model import (
    CrossAttentionAffinityModel,
    MultiTaskLoss
)
from src.classifier.utils.matrix_dataloader import create_matrix_dataloader

from .config import (
    TrainingConfig,
    DATASET_PATHS,
    EMBEDDING_BASE_PATH,
    EMBEDDING_BASE_PATHS_ALL,
    PROTEIN_DIMS,
    MAX_SEQ_LEN,
    DEFAULT_AFFINITY_THRESHOLD,
    MIN_SEEDS_FOR_STATISTICS,
    DEFAULT_SEEDS,
    LIGAND_MATRIX_DIRS,
    MOLFORMER_DIM
)
from .data import (
    create_attention_dataloader,
    split_random,
    split_by_compound,
    split_new_compound_new_kinase,
    get_scenarios
)
from .training import train_model, evaluate, EvaluationError
from .utils import get_device, set_seed, get_checkpoint_path
from .visualization import plot_results, save_results, print_summary


def run_scenario(
    scenario_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_matrix_dirs: List[str],
    ligand_matrix_dirs: List[str],
    config: TrainingConfig,
    device,
    seed: int,
    checkpoint_path: Optional[str] = None,
    use_attention: bool = False
) -> Dict:
    """
    Run training and evaluation for one scenario with one seed.

    Args:
        scenario_name: Name of the scenario
        train_df, val_df, test_df: DataFrames for each split
        protein_matrix_dirs: List of paths to protein embeddings/attention matrices
        ligand_matrix_dirs: List of paths to ligand embeddings
        config: Training configuration
        device: Training device
        seed: Random seed
        checkpoint_path: Path for checkpointing
        use_attention: Use attention matrices instead of embeddings

    Returns:
        Dictionary with test metrics
    """
    # Set seed for reproducibility
    set_seed(seed, deterministic=True)

    print(f"\n  Creating data loaders...")
    print(f"  Input type: {'Attention Matrices' if use_attention else 'Per-token Embeddings'}")

    # Create data loaders (pass list of directories for multi-source datasets)
    if use_attention:
        train_loader = create_attention_dataloader(
            train_df, protein_matrix_dirs, ligand_matrix_dirs,
            max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=True,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        val_loader = create_attention_dataloader(
            val_df, protein_matrix_dirs, ligand_matrix_dirs,
            max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=False,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        test_loader = create_attention_dataloader(
            test_df, protein_matrix_dirs, ligand_matrix_dirs,
            max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=False,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
    else:
        train_loader = create_matrix_dataloader(
            train_df, protein_matrix_dirs, ligand_matrix_dirs,
            batch_size=config.batch_size, shuffle=True,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        val_loader = create_matrix_dataloader(
            val_df, protein_matrix_dirs, ligand_matrix_dirs,
            batch_size=config.batch_size, shuffle=False,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        test_loader = create_matrix_dataloader(
            test_df, protein_matrix_dirs, ligand_matrix_dirs,
            batch_size=config.batch_size, shuffle=False,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )

    # Create model
    print(f"  Creating model (protein_dim={config.protein_dim}, ligand_dim={config.ligand_dim})...")

    model = CrossAttentionAffinityModel(
        protein_dim=config.protein_dim,
        ligand_dim=config.ligand_dim,
        hidden_dim=config.hidden_dim,
        num_cnn_layers=config.num_cnn_layers,
        num_cross_attn_layers=config.num_cross_attn_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Loss function
    loss_fn = MultiTaskLoss(
        classification_weight=1.0,
        regression_weight=0.5
    )

    # Train
    model, history = train_model(
        model, train_loader, val_loader, config, device, loss_fn,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10
    )

    # Evaluate on test set
    print(f"  Evaluating on test set...")
    test_result = evaluate(model, test_loader, device, raise_on_invalid=False)

    if not test_result.is_valid:
        raise EvaluationError(f"Test evaluation failed: {test_result.failure_reason}")

    test_metrics = test_result.metrics
    test_metrics['n_params'] = n_params

    # Loss is now calculated as log_loss on test set (from evaluate())
    # This is consistent with baseline models (KNN, MLP) and is the standard practice

    print(f"  Results: Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, "
          f"AUC={test_metrics['auc']:.4f}, Loss={test_metrics['loss']:.4f}")

    return test_metrics


def run_crossattention_analysis(
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    config: TrainingConfig,
    seeds: List[int] = None,
    prefix: str = "",
    use_attention: bool = False,
    scenarios: List[str] = None,
    use_molformer_ligand: bool = False
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Run complete CrossAttention analysis for one embedding + dataset.

    Args:
        embedding_name: Name of the embedding model
        dataset_type: Type of dataset (human, non_human)
        output_dir: Directory to save results
        config: Training configuration
        seeds: List of random seeds
        prefix: Prefix for output files
        use_attention: Use attention matrices instead of embeddings
        scenarios: List of scenario keys to run (None = all)
        use_molformer_ligand: Use MoLFormer matrices instead of SMI-TED for ligands

    Returns:
        Tuple of (all_results, split_stats) or (None, None) on failure
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # Validate paths
    dataset_path = DATASET_PATHS.get(dataset_type)

    # For dataset 'all', use multiple directories (human + non_human)
    if dataset_type == 'all':
        embedding_dirs = [
            os.path.join(base_path, embedding_name, 'build')
            for base_path in EMBEDDING_BASE_PATHS_ALL
        ]
    else:
        embedding_dirs = [os.path.join(
            EMBEDDING_BASE_PATH.format(dataset_type=dataset_type),
            embedding_name,
            'build'
        )]

    if use_attention:
        protein_matrix_dirs = [os.path.join(d, 'attention_matrices') for d in embedding_dirs]
        input_type = "Attention Matrices"
    else:
        protein_matrix_dirs = [os.path.join(d, 'protein_matrices') for d in embedding_dirs]
        input_type = "Per-token Embeddings"

    # Select ligand matrix directory
    if use_molformer_ligand:
        ligand_dir_name = LIGAND_MATRIX_DIRS['molformer']
        ligand_type = "Per-token Embeddings (MoLFormer)"
    else:
        ligand_dir_name = LIGAND_MATRIX_DIRS['smited']
        ligand_type = "Per-token Embeddings (SMI-TED)"

    ligand_matrix_dirs = [os.path.join(d, ligand_dir_name) for d in embedding_dirs]

    # For backward compatibility, keep single dir references
    protein_matrix_dir = protein_matrix_dirs[0]
    ligand_matrix_dir = ligand_matrix_dirs[0]

    print(f"\n  Protein input: {input_type}")
    if len(protein_matrix_dirs) > 1:
        print(f"  Protein dirs: {protein_matrix_dirs}")
    else:
        print(f"  Protein dir: {protein_matrix_dir}")
    print(f"  Ligand input: {ligand_type}")
    if len(ligand_matrix_dirs) > 1:
        print(f"  Ligand dirs: {ligand_matrix_dirs}")
    else:
        print(f"  Ligand dir: {ligand_matrix_dir}")

    # Check that at least one directory exists
    protein_dirs_exist = [os.path.exists(d) for d in protein_matrix_dirs]
    ligand_dirs_exist = [os.path.exists(d) for d in ligand_matrix_dirs]

    if not any(protein_dirs_exist):
        print(f"ERROR: No protein matrices found in: {protein_matrix_dirs}")
        return None, None

    if not any(ligand_dirs_exist):
        print(f"ERROR: No ligand matrices found in: {ligand_matrix_dirs}")
        return None, None

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, sep='\t')

    # Apply threshold using pChEMBL values (logarithmic scale)
    # pChEMBL >= threshold means active (higher pChEMBL = higher affinity = lower IC50/Kd)
    threshold = config.affinity_threshold.threshold_pchembl
    threshold_nm_equiv = 10 ** (9 - threshold)

    # Calculate pChEMBL from standard_value when missing
    # pChEMBL = -log10(Molar) = -log10(nM / 1e9) = 9 - log10(nM)
    import numpy as np
    n_missing = df['pchembl_value'].isna().sum()
    if n_missing > 0:
        print(f"  Calculating pChEMBL for {n_missing} samples with missing values...")
        mask_missing = df['pchembl_value'].isna()
        df.loc[mask_missing, 'pchembl_value'] = 9 - np.log10(df.loc[mask_missing, 'standard_value'])

    df['label'] = (df['pchembl_value'] >= threshold).astype(int)
    df['seq_id'] = df['seq_id'].astype(str)

    # Report class distribution
    n_active = df['label'].sum()
    n_inactive = len(df) - n_active
    print(f"  Total: {len(df)} samples, {df['chembl_id'].nunique()} compounds, "
          f"{df['target_kinase'].nunique()} kinases")
    print(f"  Affinity threshold: pChEMBL >= {threshold:.1f} (equivalent to <= {threshold_nm_equiv:.0f} nM / {threshold_nm_equiv/1000:.1f} μM)")
    print(f"  Class distribution: {n_active} active ({100*n_active/len(df):.1f}%), "
          f"{n_inactive} inactive ({100*n_inactive/len(df):.1f}%)")

    device = get_device()
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds} (n={len(seeds)})")

    all_results = {}
    split_stats = {}

    # Define scenarios in order (hardest to easiest)
    # Map scenario keys to (display_name, plot_key, split_function)
    all_scenarios_config = {
        'new_compound_new_kinase': ('New Compound + New Kinase', 'New Comp.\n+ New Kinase', split_new_compound_new_kinase),
        'compound': ('Split by Compound', 'Split by\nCompound', split_by_compound),
        'random': ('Random Split', 'Random Split\n(Original)', split_random),
    }

    # Filter scenarios if specified
    if scenarios is None:
        scenarios = ['new_compound_new_kinase', 'compound', 'random']

    scenarios_config = [
        all_scenarios_config[s] for s in scenarios if s in all_scenarios_config
    ]

    for scenario_name, scenario_key, split_fn in scenarios_config:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")

        scenario_seed_results = {}

        for seed in seeds:
            print(f"\n  Seed: {seed}")

            # Split data
            train_idx, val_idx, test_idx = split_fn(df, seed=seed)

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Record stats (only for first seed)
            if seed == seeds[0]:
                split_stats[scenario_key] = {
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'test_size': len(test_idx),
                    'test_compounds': test_df['chembl_id'].nunique(),
                }
                if 'target_kinase' in test_df.columns:
                    split_stats[scenario_key]['test_kinases'] = test_df['target_kinase'].nunique()

            print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            # Checkpoint path includes seed
            checkpoint_path = get_checkpoint_path(
                output_dir, f"{prefix}seed{seed}_", scenario_name
            )
            print(f"  Checkpoint: {checkpoint_path}")

            # Run scenario
            try:
                metrics = run_scenario(
                    scenario_name, train_df, val_df, test_df,
                    protein_matrix_dirs, ligand_matrix_dirs,
                    config, device, seed,
                    checkpoint_path=checkpoint_path,
                    use_attention=use_attention
                )
                scenario_seed_results[seed] = metrics
            except EvaluationError as e:
                print(f"  ERROR: {e}")
                continue

        # Aggregate results across seeds
        if scenario_seed_results:
            # For single seed, just use the metrics directly
            if len(scenario_seed_results) == 1:
                seed_metrics = list(scenario_seed_results.values())[0]
                all_results[scenario_key] = {'CNN+CrossAttn': seed_metrics}
            else:
                # Calculate mean and std across seeds
                import numpy as np

                aggregated = {}
                for metric in ['accuracy', 'mcc', 'auc', 'f1', 'loss']:
                    values = [r[metric] for r in scenario_seed_results.values() if metric in r]
                    if values:
                        aggregated[metric] = float(np.mean(values))
                        aggregated[f'{metric}_std'] = float(np.std(values, ddof=1))

                aggregated['seed_results'] = scenario_seed_results
                aggregated['n_seeds'] = len(scenario_seed_results)
                all_results[scenario_key] = {'CNN+CrossAttn': aggregated}

    return all_results, split_stats


def run_single_analysis(
    embedding_name: str,
    dataset_type: str,
    output_dir: str,
    seeds: List[int] = None,
    force: bool = False,
    use_attention: bool = False,
    scenarios: List[str] = None,
    num_epochs: int = 500,
    patience: Optional[int] = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    use_molformer_ligand: bool = False
) -> Optional[Dict]:
    """
    Run analysis for a single embedding + dataset combination.

    Args:
        embedding_name: Short name (8M, 150M, 650M) or full name
        dataset_type: Dataset type
        output_dir: Output directory
        seeds: List of random seeds
        force: Force recalculation even if results exist
        use_attention: Use attention matrices
        scenarios: List of scenario keys to run (None = all)
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        batch_size: Training batch size
        learning_rate: Learning rate
        use_molformer_ligand: Use MoLFormer matrices instead of SMI-TED for ligands

    Returns:
        Results dictionary or None
    """
    from .config import SUPPORTED_EMBEDDINGS, DEFAULT_SCENARIOS

    if seeds is None:
        seeds = DEFAULT_SEEDS

    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    # Resolve embedding name
    if embedding_name in SUPPORTED_EMBEDDINGS:
        embedding_name = SUPPORTED_EMBEDDINGS[embedding_name]

    # Generate prefix (include molformer tag to avoid overwriting existing results)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    attn_prefix = 'attn_' if use_attention else ''
    molformer_prefix = 'molformer_' if use_molformer_ligand else ''
    prefix = f"{dataset_type}_{attn_prefix}{molformer_prefix}{short_name}_"

    # Check cache
    json_file = os.path.join(output_dir, f'{prefix}crossattention_analysis_results.json')

    if os.path.exists(json_file) and not force:
        print(f"\n[CACHE] Results already exist: {json_file}")
        print(f"        Use --force to recalculate.")
        return None

    protein_input_type = "ATTENTION MATRICES" if use_attention else "PER-TOKEN EMBEDDINGS"
    ligand_input_type = "PER-TOKEN EMBEDDINGS (MoLFormer)" if use_molformer_ligand else "PER-TOKEN EMBEDDINGS (SMI-TED)"
    print("\n" + "=" * 70)
    print(f"CNN+CROSSATTENTION ANALYSIS: {embedding_name} + {dataset_type}")
    print(f"PROTEIN INPUT: {protein_input_type}")
    print(f"LIGAND INPUT: {ligand_input_type}")
    print(f"SEEDS: {seeds}")
    print(f"SCENARIOS: {scenarios}")
    if patience is None:
        print(f"EARLY STOPPING: DISABLED (training for {num_epochs} epochs)")
    else:
        print(f"EARLY STOPPING: patience={patience}")
    print("=" * 70)

    # Create config
    if use_attention:
        protein_dim = MAX_SEQ_LEN
    else:
        protein_dim = PROTEIN_DIMS.get(embedding_name, 640)

    # Ligand dimension (same for both SMI-TED and MoLFormer)
    ligand_dim = MOLFORMER_DIM if use_molformer_ligand else 768

    config = TrainingConfig(
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis
    all_results, split_stats = run_crossattention_analysis(
        embedding_name, dataset_type, output_dir, config,
        seeds=seeds, prefix=prefix, use_attention=use_attention,
        scenarios=scenarios, use_molformer_ligand=use_molformer_ligand
    )

    if all_results is None:
        return None

    # Generate plots
    plot_results(all_results, split_stats, embedding_name, output_dir, prefix)

    # Save results
    save_results(
        all_results, split_stats, embedding_name, dataset_type,
        output_dir, prefix, config=config.to_dict()
    )

    # Print summary
    print_summary(all_results)

    return all_results
