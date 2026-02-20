"""Main experiment runner for CrossAttention split analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.classifier.models.cross_attention_model import (
    CrossAttentionAffinityModel,
    MultiTaskLoss
)
from src.classifier.utils.matrix_dataloader import create_matrix_dataloader

from .config import (
    TrainingConfig,
    EMBEDDING_BASE_PATH,
    EMBEDDING_BASE_PATHS_ALL,
    PROTEIN_DIMS,
    MAX_SEQ_LEN,
    DEFAULT_SEEDS,
    LIGAND_MATRIX_DIRS,
    MOLFORMER_DIM
)
from .data import (
    create_attention_dataloader
)
from .training import train_model, evaluate, EvaluationError
from .utils import get_device, set_seed, get_checkpoint_path
from .visualization import plot_results, save_results, print_summary

DEFAULT_SCAFFOLD_SPLIT_DIR = "scaffolds_splits/output"
SCAFFOLD_SCENARIO_CODE = "Sc"
SCAFFOLD_SCENARIO_NAME = "Split by Scaffold"
SCAFFOLD_SCENARIO_KEY = "Split by\nScaffold"


def _ensure_required_columns(df: pd.DataFrame, split_name: str) -> None:
    """Validate minimum schema required for training/evaluation."""
    required = {"chembl_id", "target_kinase", "seq_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{split_name} split is missing required column(s): {missing}")


def _ensure_label_column(df: pd.DataFrame, threshold: float, split_name: str) -> pd.DataFrame:
    """Ensure binary label exists; derive from pChEMBL when needed."""
    out = df.copy()
    if "label" not in out.columns:
        if "pchembl_value" not in out.columns:
            raise ValueError(f"{split_name} split has no 'label' and no 'pchembl_value' to derive labels")
        out["label"] = (out["pchembl_value"] >= threshold).astype(int)
    else:
        out["label"] = out["label"].astype(int)

    out["seq_id"] = out["seq_id"].astype(str)
    return out


def _read_split_tsv(path: Path, split_name: str) -> pd.DataFrame:
    """Read one TSV split file with clear error messaging."""
    if not path.exists():
        raise FileNotFoundError(f"Required precomputed split file not found: {path}")
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {split_name} split file {path}: {exc}") from exc


def _load_precomputed_scaffold_splits(
    dataset_type: str,
    scaffold_split_dir: str,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load fixed scaffold splits produced by scaffold_split.py.

    Expected files:
      - {dir}/scenarios/Sc/{dataset}_train.tsv
      - {dir}/scenarios/Sc/{dataset}_val.tsv
      - {dir}/{dataset}_test.tsv
    where dataset is human/non_human, or both concatenated when dataset_type=all.
    """
    base = Path(scaffold_split_dir)
    scenario_dir = base / "scenarios" / SCAFFOLD_SCENARIO_CODE

    def _load_one(ds: str):
        train_path = scenario_dir / f"{ds}_train.tsv"
        val_path = scenario_dir / f"{ds}_val.tsv"
        test_path = base / f"{ds}_test.tsv"

        train_df = _read_split_tsv(train_path, f"{ds} train")
        val_df = _read_split_tsv(val_path, f"{ds} val")
        test_df = _read_split_tsv(test_path, f"{ds} test")

        train_df = _ensure_label_column(train_df, threshold, f"{ds} train")
        val_df = _ensure_label_column(val_df, threshold, f"{ds} val")
        test_df = _ensure_label_column(test_df, threshold, f"{ds} test")

        _ensure_required_columns(train_df, f"{ds} train")
        _ensure_required_columns(val_df, f"{ds} val")
        _ensure_required_columns(test_df, f"{ds} test")

        return train_df, val_df, test_df, {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "test_path": str(test_path),
        }

    if dataset_type in {"human", "non_human"}:
        train_df, val_df, test_df, paths = _load_one(dataset_type)
        metadata = {"dataset_type": dataset_type, "split_source": "precomputed_scaffold_split", "paths": paths}
        return train_df, val_df, test_df, metadata

    if dataset_type == "all":
        h_train, h_val, h_test, h_paths = _load_one("human")
        n_train, n_val, n_test, n_paths = _load_one("non_human")

        for frame, source in (
            (h_train, "human"), (h_val, "human"), (h_test, "human"),
            (n_train, "non_human"), (n_val, "non_human"), (n_test, "non_human"),
        ):
            if "dataset_source" not in frame.columns:
                frame["dataset_source"] = source

        train_df = pd.concat([h_train, n_train], axis=0, ignore_index=True)
        val_df = pd.concat([h_val, n_val], axis=0, ignore_index=True)
        test_df = pd.concat([h_test, n_test], axis=0, ignore_index=True)

        metadata = {
            "dataset_type": dataset_type,
            "split_source": "precomputed_scaffold_split",
            "paths": {"human": h_paths, "non_human": n_paths},
        }
        return train_df, val_df, test_df, metadata

    raise ValueError(f"Unsupported dataset_type='{dataset_type}'. Expected human, non_human, or all.")


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
    use_molformer_ligand: bool = False,
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR,
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
        scenarios: Scenario list (only scaffold is supported)
        use_molformer_ligand: Use MoLFormer matrices instead of SMI-TED for ligands
        scaffold_split_dir: Directory generated by scaffold_split.py

    Returns:
        Tuple of (all_results, split_stats) or (None, None) on failure
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

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

    # Load fixed scaffold split produced by scaffold_split.py
    threshold = config.affinity_threshold.threshold_pchembl
    threshold_nm_equiv = 10 ** (9 - threshold)
    print(f"\nLoading precomputed scaffold splits from: {scaffold_split_dir}")
    try:
        train_df, val_df, test_df, split_source_metadata = _load_precomputed_scaffold_splits(
            dataset_type=dataset_type,
            scaffold_split_dir=scaffold_split_dir,
            threshold=threshold,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return None, None

    total_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    n_active = int(total_df['label'].sum())
    n_inactive = int(len(total_df) - n_active)

    print(
        f"  Loaded fixed splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(total={len(total_df)})"
    )
    print(f"  Total compounds: {total_df['chembl_id'].nunique()}, kinases: {total_df['target_kinase'].nunique()}")
    print(f"  Affinity threshold: pChEMBL >= {threshold:.1f} (equivalent to <= {threshold_nm_equiv:.0f} nM / {threshold_nm_equiv/1000:.1f} μM)")
    print(f"  Class distribution: {n_active} active ({100*n_active/len(total_df):.1f}%), "
          f"{n_inactive} inactive ({100*n_inactive/len(total_df):.1f}%)")

    device = get_device()
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds} (n={len(seeds)})")

    all_results = {}
    split_stats = {}

    # Only scaffold scenario is supported in this pipeline.
    if scenarios is None:
        scenarios = ['scaffold']
    normalized = {s.strip().lower() for s in scenarios}
    if normalized not in ({"scaffold"}, {"sc"}, {"split_by_scaffold"}):
        print(
            "ERROR: This CrossAttention pipeline now supports only the scaffold scenario "
            "and consumes precomputed Sc train/val + fixed test splits."
        )
        return None, None

    print(f"\n{'='*60}")
    print(f"SCENARIO: {SCAFFOLD_SCENARIO_NAME}")
    print(f"{'='*60}")
    print(f"  Split source: {split_source_metadata}")

    split_stats[SCAFFOLD_SCENARIO_KEY] = {
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(test_df)),
        'test_compounds': int(test_df['chembl_id'].nunique()),
        'test_kinases': int(test_df['target_kinase'].nunique()) if 'target_kinase' in test_df.columns else 0,
        'split_source': split_source_metadata,
    }

    scenario_seed_results = {}
    for seed in seeds:
        print(f"\n  Seed: {seed}")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        checkpoint_path = get_checkpoint_path(
            output_dir, f"{prefix}seed{seed}_", SCAFFOLD_SCENARIO_NAME
        )
        print(f"  Checkpoint: {checkpoint_path}")

        try:
            metrics = run_scenario(
                SCAFFOLD_SCENARIO_NAME,
                train_df,
                val_df,
                test_df,
                protein_matrix_dirs,
                ligand_matrix_dirs,
                config,
                device,
                seed,
                checkpoint_path=checkpoint_path,
                use_attention=use_attention,
            )
            scenario_seed_results[seed] = metrics
        except EvaluationError as e:
            print(f"  ERROR: {e}")
            continue

    if scenario_seed_results:
        if len(scenario_seed_results) == 1:
            seed_metrics = list(scenario_seed_results.values())[0]
            all_results[SCAFFOLD_SCENARIO_KEY] = {'CNN+CrossAttn': seed_metrics}
        else:
            aggregated = {}
            for metric in ['accuracy', 'mcc', 'auc', 'f1', 'loss']:
                values = [r[metric] for r in scenario_seed_results.values() if metric in r]
                if values:
                    aggregated[metric] = float(np.mean(values))
                    aggregated[f'{metric}_std'] = float(np.std(values, ddof=1))

            aggregated['seed_results'] = scenario_seed_results
            aggregated['n_seeds'] = len(scenario_seed_results)
            all_results[SCAFFOLD_SCENARIO_KEY] = {'CNN+CrossAttn': aggregated}

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
    use_molformer_ligand: bool = False,
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR,
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
        scaffold_split_dir: Directory generated by scaffold_split.py

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
    print(f"SCAFFOLD SPLIT DIR: {scaffold_split_dir}")
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
        scenarios=scenarios, use_molformer_ligand=use_molformer_ligand,
        scaffold_split_dir=scaffold_split_dir
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
