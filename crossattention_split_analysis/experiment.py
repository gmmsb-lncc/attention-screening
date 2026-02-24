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
from src.classifier.models.diffusion_model import DiffusionAffinityModel
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
from .training import (
    train_model,
    evaluate,
    optimize_decision_threshold,
    EvaluationError,
)
from .utils import get_device, set_seed, get_checkpoint_path
from .visualization import plot_results, save_results, print_summary

DEFAULT_SCAFFOLD_SPLIT_DIR = "scaffolds_splits/output"
SCAFFOLD_SCENARIO_CODE = "Sc"
SCAFFOLD_SCENARIO_NAME = "Split by Scaffold"
SCAFFOLD_SCENARIO_KEY = "Split by\nScaffold"


def _compute_class_pos_weight(train_df: pd.DataFrame) -> float:
    """
    Compute BCE positive-class weight from the training split.

    pos_weight = n_negative / n_positive
    """
    n_pos = int(train_df["label"].sum())
    n_total = int(len(train_df))
    n_neg = n_total - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return 1.0
    return float(n_neg / n_pos)


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


def _read_split_tsv(path: Path, split_name: str) -> Tuple[pd.DataFrame, Path]:
    """Read one split file from .tsv or .tsv.gz with clear error messaging."""
    candidates = [path, path.with_suffix(path.suffix + ".gz")]
    selected = next((p for p in candidates if p.exists()), None)
    if selected is None:
        raise FileNotFoundError(
            f"Required precomputed split file not found: {path} "
            f"(also checked: {path.with_suffix(path.suffix + '.gz')})"
        )
    try:
        return pd.read_csv(selected, sep="\t"), selected
    except Exception as exc:
        raise RuntimeError(f"Failed to read {split_name} split file {selected}: {exc}") from exc


def _load_precomputed_scaffold_splits(
    dataset_type: str,
    scaffold_split_dir: str,
    threshold: float,
    include_test: bool = True,
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

        train_df, train_used = _read_split_tsv(train_path, f"{ds} train")
        val_df, val_used = _read_split_tsv(val_path, f"{ds} val")
        if include_test:
            test_df, test_used = _read_split_tsv(test_path, f"{ds} test")
        else:
            test_df = val_df.iloc[0:0].copy()
            test_used = None

        train_df = _ensure_label_column(train_df, threshold, f"{ds} train")
        val_df = _ensure_label_column(val_df, threshold, f"{ds} val")
        if include_test:
            test_df = _ensure_label_column(test_df, threshold, f"{ds} test")

        _ensure_required_columns(train_df, f"{ds} train")
        _ensure_required_columns(val_df, f"{ds} val")
        if include_test:
            _ensure_required_columns(test_df, f"{ds} test")

        return train_df, val_df, test_df, {
            "train_path": str(train_used),
            "val_path": str(val_used),
            "test_path": str(test_used) if test_used else None,
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

MODEL_VARIANT_TO_ENCODER = {
    'cnn_crossattn': 'cnn',
    'cross_attention_lite': 'linear',
    'diffusion': 'diffusion',
}

MODEL_VARIANT_TO_LABEL = {
    'cnn_crossattn': 'CNN+CrossAttn',
    'cross_attention_lite': 'CrossAttnLite',
    'diffusion': 'Diffusion',
}


def _resolve_model_variant(model_variant: str) -> Tuple[str, str]:
    if model_variant not in MODEL_VARIANT_TO_ENCODER:
        supported = ", ".join(MODEL_VARIANT_TO_ENCODER.keys())
        raise ValueError(f"Unsupported model_variant={model_variant!r}. Supported: {supported}")
    return MODEL_VARIANT_TO_ENCODER[model_variant], MODEL_VARIANT_TO_LABEL[model_variant]


def run_scenario(
    scenario_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    protein_matrix_dirs: List[str],
    ligand_matrix_dirs: List[str],
    ligand_vector_dirs: Optional[List[str]],
    config: TrainingConfig,
    device,
    seed: int,
    checkpoint_path: Optional[str] = None,
    use_attention: bool = False,
    use_ligand_vectors: bool = False,
    external_test_df: Optional[pd.DataFrame] = None,
    evaluation_split: str = "test",
) -> Dict:
    """
    Run training and evaluation for one scenario with one seed.

    Args:
        scenario_name: Name of the scenario
        train_df, val_df, test_df: DataFrames for each split (test_df can be None)
        protein_matrix_dirs: List of paths to protein embeddings/attention matrices
        ligand_matrix_dirs: List of paths to ligand embeddings
        ligand_vector_dirs: List of paths to ligand vectors (optional)
        config: Training configuration
        device: Training device
        seed: Random seed
        checkpoint_path: Path for checkpointing
        use_attention: Use attention matrices instead of embeddings
        use_ligand_vectors: Use ligand vectors instead of per-token matrices
        external_test_df: Optional external test DataFrame (never used for training)
        evaluation_split: Which split to evaluate after training ("test" or "val")

    Returns:
        Dictionary with test metrics
    """
    # Set seed for reproducibility
    set_seed(seed, deterministic=True)

    print(f"\n  Creating data loaders...")
    print(f"  Input type: {'Attention Matrices' if use_attention else 'Per-token Embeddings'}")

    if evaluation_split not in {"test", "val"}:
        raise ValueError(f"evaluation_split must be 'test' or 'val', got {evaluation_split!r}")

    # Create data loaders (pass list of directories for multi-source datasets)
    if use_attention:
        train_loader = create_attention_dataloader(
            train_df, protein_matrix_dirs, ligand_matrix_dirs,
            max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=True,
            num_workers=config.dataloader_num_workers,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        val_loader = create_attention_dataloader(
            val_df, protein_matrix_dirs, ligand_matrix_dirs,
            max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=False,
            num_workers=config.dataloader_num_workers,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        test_loader = None
        if test_df is not None and len(test_df) > 0:
            test_loader = create_attention_dataloader(
                test_df, protein_matrix_dirs, ligand_matrix_dirs,
                max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=False,
                num_workers=config.dataloader_num_workers,
                label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
            )
    else:
        train_loader = create_matrix_dataloader(
            train_df, protein_matrix_dirs, ligand_matrix_dirs,
            ligand_vector_dir=ligand_vector_dirs if use_ligand_vectors else None,
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            prefetch_factor=config.dataloader_prefetch_factor,
            persistent_workers=config.dataloader_persistent_workers,
            cache_in_memory=config.dataloader_cache_in_memory,
            use_ligand_matrices=not use_ligand_vectors,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        val_loader = create_matrix_dataloader(
            val_df, protein_matrix_dirs, ligand_matrix_dirs,
            ligand_vector_dir=ligand_vector_dirs if use_ligand_vectors else None,
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            prefetch_factor=config.dataloader_prefetch_factor,
            persistent_workers=config.dataloader_persistent_workers,
            cache_in_memory=config.dataloader_cache_in_memory,
            use_ligand_matrices=not use_ligand_vectors,
            label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
        )
        test_loader = None
        if test_df is not None and len(test_df) > 0:
            test_loader = create_matrix_dataloader(
                test_df, protein_matrix_dirs, ligand_matrix_dirs,
                ligand_vector_dir=ligand_vector_dirs if use_ligand_vectors else None,
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.dataloader_pin_memory,
                prefetch_factor=config.dataloader_prefetch_factor,
                persistent_workers=config.dataloader_persistent_workers,
                cache_in_memory=config.dataloader_cache_in_memory,
                use_ligand_matrices=not use_ligand_vectors,
                label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
            )

    encoder_type, model_label = _resolve_model_variant(config.model_variant)
    print(
        f"  Creating model ({model_label}; protein_dim={config.protein_dim}, "
        f"ligand_dim={config.ligand_dim})..."
    )

    if encoder_type == "diffusion":
        model = DiffusionAffinityModel(
            protein_dim=config.protein_dim,
            ligand_dim=config.ligand_dim,
            hidden_dim=config.hidden_dim,
            num_diffusion_layers=config.diffusion_layers,
            num_cross_attn_layers=config.diffusion_cross_attn_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            pool_num_queries=config.diffusion_pool_queries,
            diffusion_steps=config.diffusion_steps,
            diffusion_beta_start=config.diffusion_beta_start,
            diffusion_beta_end=config.diffusion_beta_end,
            diffusion_loss_weight=config.diffusion_loss_weight,
            snr_sampling_gamma=config.diffusion_snr_sampling_gamma,
            snr_sampling_mix=config.diffusion_snr_sampling_mix,
            joint_denoise=config.diffusion_joint_denoise,
            classification_only=config.classification_only,
        )
    else:
        model = CrossAttentionAffinityModel(
            protein_dim=config.protein_dim,
            ligand_dim=config.ligand_dim,
            hidden_dim=config.hidden_dim,
            encoder_type=encoder_type,
            num_cnn_layers=config.num_cnn_layers,
            num_cross_attn_layers=config.num_cross_attn_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Class imbalance correction (computed only from training labels).
    class_pos_weight = _compute_class_pos_weight(train_df)
    print(f"  Class weighting: pos_weight={class_pos_weight:.4f} (computed from train split)")

    # Loss function
    loss_fn = MultiTaskLoss(
        classification_weight=config.classification_weight,
        regression_weight=config.regression_weight,
        classification_pos_weight=class_pos_weight,
    )

    # Train
    model, history = train_model(
        model, train_loader, val_loader, config, device, loss_fn,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10
    )

    # Calibrate decision threshold on validation (no test leakage).
    if config.optimize_threshold:
        print(f"  Optimizing decision threshold on validation ({config.threshold_metric})...")
        threshold_result = optimize_decision_threshold(
            model,
            val_loader,
            device,
            metric=config.threshold_metric,
            raise_on_invalid=False,
        )
        if not threshold_result.is_valid:
            raise EvaluationError(f"Threshold optimization failed: {threshold_result.failure_reason}")
        decision_threshold = float(threshold_result.metrics["decision_threshold"])
        threshold_source = f"validation_{config.threshold_metric}"
        print(
            f"  Selected threshold={decision_threshold:.4f} "
            f"(best {config.threshold_metric}={threshold_result.metrics.get('threshold_optimized_score', np.nan):.4f})"
        )
    else:
        decision_threshold = float(config.fixed_threshold)
        threshold_source = "fixed"
        print(f"  Using fixed decision threshold={decision_threshold:.4f}")

    # Evaluate on requested split (external-test mode evaluates on validation only).
    if evaluation_split == "test" and test_loader is None:
        raise EvaluationError("Test evaluation requested, but test split is unavailable")
    eval_loader = test_loader if evaluation_split == "test" else val_loader
    eval_name = "test" if evaluation_split == "test" else "validation"
    print(f"  Evaluating on {eval_name} set...")
    eval_result = evaluate(
        model,
        eval_loader,
        device,
        raise_on_invalid=False,
        decision_threshold=decision_threshold,
    )

    if not eval_result.is_valid:
        raise EvaluationError(f"{eval_name.capitalize()} evaluation failed: {eval_result.failure_reason}")

    test_metrics = eval_result.metrics
    test_metrics['n_params'] = n_params
    test_metrics['evaluation_split'] = eval_name
    test_metrics['threshold_source'] = threshold_source
    test_metrics['class_pos_weight'] = class_pos_weight

    if config.optimize_threshold:
        test_metrics['threshold_optimized_metric'] = config.threshold_metric
        test_metrics['threshold_optimized_score'] = float(
            threshold_result.metrics.get('threshold_optimized_score', np.nan)
        )
        test_metrics['threshold_candidates'] = int(
            threshold_result.metrics.get('threshold_candidates', 0)
        )

    # Loss is now calculated as log_loss on test set (from evaluate())
    # This is consistent with baseline models (KNN, MLP) and is the standard practice

    print(
        f"  Results ({eval_name}): Acc={test_metrics['accuracy']:.4f}, MCC={test_metrics['mcc']:.4f}, "
        f"AUC={test_metrics['auc']:.4f}, Loss={test_metrics['loss']:.4f}, "
        f"Threshold={test_metrics['decision_threshold']:.4f}"
    )

    # Optional external test evaluation (never used for threshold calibration)
    if external_test_df is not None and len(external_test_df) > 0:
        print("  Evaluating on external test set...")
        if use_attention:
            external_loader = create_attention_dataloader(
                external_test_df, protein_matrix_dirs, ligand_matrix_dirs,
                max_seq_len=MAX_SEQ_LEN, batch_size=config.batch_size, shuffle=False,
                num_workers=config.dataloader_num_workers,
                label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
            )
        else:
            external_loader = create_matrix_dataloader(
                external_test_df, protein_matrix_dirs, ligand_matrix_dirs,
                ligand_vector_dir=ligand_vector_dirs if use_ligand_vectors else None,
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.dataloader_num_workers,
                pin_memory=config.dataloader_pin_memory,
                prefetch_factor=config.dataloader_prefetch_factor,
                persistent_workers=config.dataloader_persistent_workers,
                cache_in_memory=config.dataloader_cache_in_memory,
                use_ligand_matrices=not use_ligand_vectors,
                label_column='label', protein_id_column='seq_id', ligand_id_column='chembl_id'
            )

        external_result = evaluate(
            model,
            external_loader,
            device,
            raise_on_invalid=False,
            decision_threshold=decision_threshold,
        )

        if not external_result.is_valid:
            raise EvaluationError(
                f"External test evaluation failed: {external_result.failure_reason}"
            )

        ext_metrics = external_result.metrics
        test_metrics['external_accuracy'] = ext_metrics['accuracy']
        test_metrics['external_mcc'] = ext_metrics['mcc']
        test_metrics['external_auc'] = ext_metrics['auc']
        test_metrics['external_loss'] = ext_metrics['loss']
        test_metrics['external_evaluation_split'] = "external_test"
        print(
            f"  Results (external): Acc={ext_metrics['accuracy']:.4f}, "
            f"MCC={ext_metrics['mcc']:.4f}, AUC={ext_metrics['auc']:.4f}, "
            f"Loss={ext_metrics['loss']:.4f}"
        )

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
    use_molformer_ligand: bool = True,
    use_ligand_vectors: bool = False,
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR,
    external_test_mode: bool = False,
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
        use_ligand_vectors: Use ligand vectors instead of per-token matrices
        scaffold_split_dir: Directory generated by scaffold_split.py
        external_test_mode: If True, use only precomputed scaffold train/val and skip internal test

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

    if use_attention and use_ligand_vectors:
        print("ERROR: Ligand vectors are not supported with attention matrices.")
        return None, None

    if use_attention:
        protein_matrix_dirs = [os.path.join(d, 'attention_matrices') for d in embedding_dirs]
        input_type = "Attention Matrices"
    else:
        protein_matrix_dirs = [os.path.join(d, 'protein_matrices') for d in embedding_dirs]
        input_type = "Per-token Embeddings"

    # Select ligand input
    ligand_vector_dirs = []
    if use_ligand_vectors:
        ligand_dir_name = LIGAND_VECTOR_DIR
        ligand_type = "Vector Embeddings"
        ligand_vector_dirs = [os.path.join(d, ligand_dir_name) for d in embedding_dirs]
        # Keep matrix dirs for backward compatibility (not used in vector mode)
        ligand_matrix_dirs = [os.path.join(d, LIGAND_MATRIX_DIRS['molformer']) for d in embedding_dirs]
    else:
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
    if use_ligand_vectors:
        if len(ligand_vector_dirs) > 1:
            print(f"  Ligand dirs: {ligand_vector_dirs}")
        else:
            print(f"  Ligand dir: {ligand_vector_dirs[0]}")
    else:
        if len(ligand_matrix_dirs) > 1:
            print(f"  Ligand dirs: {ligand_matrix_dirs}")
        else:
            print(f"  Ligand dir: {ligand_matrix_dir}")

    # Check that at least one directory exists
    protein_dirs_exist = [os.path.exists(d) for d in protein_matrix_dirs]
    if use_ligand_vectors:
        ligand_dirs_exist = [os.path.exists(d) for d in ligand_vector_dirs]
    else:
        ligand_dirs_exist = [os.path.exists(d) for d in ligand_matrix_dirs]

    if not any(protein_dirs_exist):
        print(f"ERROR: No protein matrices found in: {protein_matrix_dirs}")
        return None, None

    if not any(ligand_dirs_exist):
        if use_ligand_vectors:
            print(f"ERROR: No ligand vectors found in: {ligand_vector_dirs}")
        else:
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
            include_test=not external_test_mode,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return None, None

    external_test_df = None
    if external_test_mode:
        test_path = os.path.join(scaffold_split_dir, f"{dataset_type}_test.tsv.gz")
        if dataset_type == "all":
            h_path = os.path.join(scaffold_split_dir, "human_test.tsv.gz")
            n_path = os.path.join(scaffold_split_dir, "non_human_test.tsv.gz")
            frames = []
            for path, source in ((h_path, "human"), (n_path, "non_human")):
                if os.path.exists(path):
                    df = pd.read_csv(path, sep="\t")
                    df["dataset_source"] = source
                    frames.append(df)
            if frames:
                external_test_df = pd.concat(frames, axis=0, ignore_index=True)
        elif os.path.exists(test_path):
            external_test_df = pd.read_csv(test_path, sep="\t")
        if external_test_df is None or len(external_test_df) == 0:
            print("WARNING: External test mode enabled but no external test file found.")

    if external_test_mode:
        test_df = None
        split_source_metadata = {
            **split_source_metadata,
            "external_test_mode": True,
            "train_val_protocol": "precomputed_scaffold_split",
        }
    else:
        split_source_metadata = {
            **split_source_metadata,
            "external_test_mode": False,
        }

    total_frames = [train_df, val_df] + ([test_df] if test_df is not None else [])
    total_df = pd.concat(total_frames, axis=0, ignore_index=True)
    n_active = int(total_df['label'].sum())
    n_inactive = int(len(total_df) - n_active)

    if external_test_mode:
        print(
            f"  Loaded precomputed scaffold train/val: train={len(train_df)}, val={len(val_df)} "
            f"(total={len(total_df)})"
        )
    else:
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
    _, model_label = _resolve_model_variant(config.model_variant)
    print(f"  Model variant: {config.model_variant} ({model_label})")

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

    eval_df = val_df if external_test_mode else test_df
    eval_split_name = "validation" if external_test_mode else "test"
    train_pos = int(train_df["label"].sum())
    val_pos = int(val_df["label"].sum())
    eval_pos = int(eval_df["label"].sum())
    split_stats[SCAFFOLD_SCENARIO_KEY] = {
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(eval_df)),
        'train_positive_rate': float(train_pos / len(train_df)) if len(train_df) > 0 else np.nan,
        'val_positive_rate': float(val_pos / len(val_df)) if len(val_df) > 0 else np.nan,
        'test_positive_rate': float(eval_pos / len(eval_df)) if len(eval_df) > 0 else np.nan,
        'test_compounds': int(eval_df['chembl_id'].nunique()),
        'test_kinases': int(eval_df['target_kinase'].nunique()) if 'target_kinase' in eval_df.columns else 0,
        'evaluation_split': eval_split_name,
        'split_source': split_source_metadata,
    }
    print(
        "  Split class distribution: "
        f"train={100*split_stats[SCAFFOLD_SCENARIO_KEY]['train_positive_rate']:.1f}% pos, "
        f"val={100*split_stats[SCAFFOLD_SCENARIO_KEY]['val_positive_rate']:.1f}% pos, "
        f"{eval_split_name}={100*split_stats[SCAFFOLD_SCENARIO_KEY]['test_positive_rate']:.1f}% pos"
    )

    scenario_seed_results = {}
    for seed in seeds:
        print(f"\n  Seed: {seed}")
        if external_test_mode:
            print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: external (unused)")
        else:
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
                test_df if not external_test_mode else None,
                protein_matrix_dirs,
                ligand_matrix_dirs,
                ligand_vector_dirs if use_ligand_vectors else None,
                config,
                device,
                seed,
                checkpoint_path=checkpoint_path,
                use_attention=use_attention,
                use_ligand_vectors=use_ligand_vectors,
                external_test_df=external_test_df,
                evaluation_split="val" if external_test_mode else "test",
            )
            scenario_seed_results[seed] = metrics
        except EvaluationError as e:
            print(f"  ERROR: {e}")
            continue

    if scenario_seed_results:
        if len(scenario_seed_results) == 1:
            seed_metrics = list(scenario_seed_results.values())[0]
            all_results[SCAFFOLD_SCENARIO_KEY] = {model_label: seed_metrics}
        else:
            aggregated = {}
            for metric in [
                'accuracy',
                'mcc',
                'auc',
                'f1',
                'loss',
                'decision_threshold',
                'threshold_optimized_score',
                'class_pos_weight',
            ]:
                values = [r[metric] for r in scenario_seed_results.values() if metric in r]
                if values:
                    aggregated[metric] = float(np.mean(values))
                    aggregated[f'{metric}_std'] = float(np.std(values, ddof=1))

            threshold_sources = {r.get('threshold_source') for r in scenario_seed_results.values() if 'threshold_source' in r}
            if len(threshold_sources) == 1:
                aggregated['threshold_source'] = threshold_sources.pop()

            threshold_metric_names = {
                r.get('threshold_optimized_metric')
                for r in scenario_seed_results.values()
                if 'threshold_optimized_metric' in r
            }
            if len(threshold_metric_names) == 1:
                aggregated['threshold_optimized_metric'] = threshold_metric_names.pop()

            aggregated['seed_results'] = scenario_seed_results
            aggregated['n_seeds'] = len(scenario_seed_results)
            all_results[SCAFFOLD_SCENARIO_KEY] = {model_label: aggregated}

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
    weight_decay: float = 0.01,
    hidden_dim: int = 256,
    num_cnn_layers: int = 3,
    num_cross_attn_layers: int = 2,
    num_heads: int = 8,
    ff_dim: int = 1024,
    dropout: float = 0.1,
    diffusion_steps: int = 200,
    diffusion_beta_start: float = 1e-4,
    diffusion_beta_end: float = 0.02,
    diffusion_layers: int = 4,
    diffusion_cross_attn_layers: int = 1,
    diffusion_loss_weight: float = 0.1,
    diffusion_loss_anneal: str = "none",
    classification_only: bool = False,
    diffusion_pool_queries: int = 4,
    diffusion_snr_sampling_gamma: float = 0.5,
    diffusion_snr_sampling_mix: float = 0.2,
    diffusion_joint_denoise: bool = False,
    dataloader_num_workers: int = 0,
    dataloader_cache_in_memory: bool = False,
    dataloader_pin_memory: bool = True,
    dataloader_prefetch_factor: int = 2,
    dataloader_persistent_workers: bool = True,
    max_grad_norm: float = 1.0,
    classification_weight: float = 1.0,
    regression_weight: float = 0.5,
    optimize_threshold: bool = True,
    threshold_metric: str = "mcc",
    fixed_threshold: float = 0.5,
    use_molformer_ligand: bool = True,
    use_ligand_vectors: bool = False,
    scaffold_split_dir: str = DEFAULT_SCAFFOLD_SPLIT_DIR,
    external_test_mode: bool = False,
    model_variant: str = 'cnn_crossattn',
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
        weight_decay: AdamW weight decay
        hidden_dim: Hidden dimension used by encoders and heads
        num_cnn_layers: Number of CNN encoder layers
        num_cross_attn_layers: Number of cross-attention blocks
        num_heads: Number of attention heads per block
        ff_dim: Feed-forward hidden size in attention blocks
        dropout: Dropout applied throughout model
        diffusion_steps: Number of diffusion timesteps
        diffusion_beta_start: Diffusion beta start value
        diffusion_beta_end: Diffusion beta end value
        diffusion_layers: Number of diffusion denoiser layers
        diffusion_cross_attn_layers: Number of cross-attention blocks after diffusion
        diffusion_loss_weight: Weight for diffusion auxiliary loss
        diffusion_loss_anneal: Anneal schedule for diffusion loss weight
        classification_only: If True, optimize classification head only
        diffusion_pool_queries: Number of attention pooling queries per modality
        diffusion_snr_sampling_gamma: Exponent for SNR-based timestep sampling
        diffusion_snr_sampling_mix: Mix ratio with uniform sampling (0-1)
        diffusion_joint_denoise: Joint denoising over concatenated protein+ligand tokens
        dataloader_num_workers: DataLoader worker processes
        dataloader_cache_in_memory: Cache matrices in RAM
        dataloader_pin_memory: Enable pinned memory
        dataloader_prefetch_factor: Prefetch factor when num_workers > 0
        dataloader_persistent_workers: Keep DataLoader workers alive
        max_grad_norm: Gradient clipping max norm
        classification_weight: Weight for classification loss term
        regression_weight: Weight for regression loss term
        optimize_threshold: If True, calibrate threshold on validation
        threshold_metric: Metric optimized during threshold calibration
        fixed_threshold: Threshold used when optimize_threshold=False
        use_molformer_ligand: Use MoLFormer matrices instead of SMI-TED for ligands
        use_ligand_vectors: Use ligand vectors instead of per-token matrices
        scaffold_split_dir: Directory generated by scaffold_split.py
        external_test_mode: Train/val-only mode using precomputed scaffold split; test is external
        model_variant: 'cnn_crossattn', 'cross_attention_lite', or 'diffusion'

    Returns:
        Results dictionary or None
    """
    from .config import SUPPORTED_EMBEDDINGS, DEFAULT_SCENARIOS, LIGAND_VECTOR_DIR

    if seeds is None:
        seeds = DEFAULT_SEEDS

    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    supported_threshold_metrics = {"mcc", "f1", "balanced_accuracy"}
    if optimize_threshold and threshold_metric not in supported_threshold_metrics:
        raise ValueError(
            f"Unsupported threshold_metric={threshold_metric!r}. "
            f"Expected one of {sorted(supported_threshold_metrics)}"
        )
    if not (0.0 <= fixed_threshold <= 1.0):
        raise ValueError(f"fixed_threshold must be in [0, 1], got {fixed_threshold}")
    if diffusion_snr_sampling_gamma <= 0:
        raise ValueError(
            f"diffusion_snr_sampling_gamma must be > 0, got {diffusion_snr_sampling_gamma}"
        )
    if not (0.0 <= diffusion_snr_sampling_mix <= 1.0):
        raise ValueError(
            f"diffusion_snr_sampling_mix must be in [0, 1], got {diffusion_snr_sampling_mix}"
        )

    # Resolve embedding name
    if embedding_name in SUPPORTED_EMBEDDINGS:
        embedding_name = SUPPORTED_EMBEDDINGS[embedding_name]

    # Generate prefix (include molformer tag to avoid overwriting existing results)
    short_name = embedding_name.replace('esm2_', '').replace('_UR50D', '')
    attn_prefix = 'attn_' if use_attention else ''
    molformer_prefix = 'molformer_' if (use_molformer_ligand and not use_ligand_vectors) else ''
    ligvec_prefix = 'ligvec_' if use_ligand_vectors else ''
    if model_variant == 'cross_attention_lite':
        variant_prefix = 'lite_'
    elif model_variant == 'diffusion':
        variant_prefix = 'diffusion_'
    else:
        variant_prefix = ''
    prefix = f"{dataset_type}_{variant_prefix}{attn_prefix}{molformer_prefix}{ligvec_prefix}{short_name}_"

    # Check cache
    json_file = os.path.join(output_dir, f'{prefix}crossattention_analysis_results.json')

    if os.path.exists(json_file) and not force:
        print(f"\n[CACHE] Results already exist: {json_file}")
        print(f"        Use --force to recalculate.")
        return None

    _, model_label = _resolve_model_variant(model_variant)
    protein_input_type = "ATTENTION MATRICES" if use_attention else "PER-TOKEN EMBEDDINGS"
    if use_ligand_vectors:
        ligand_input_type = "VECTOR EMBEDDINGS"
    else:
        ligand_input_type = "PER-TOKEN EMBEDDINGS (MoLFormer)" if use_molformer_ligand else "PER-TOKEN EMBEDDINGS (SMI-TED)"
    print("\n" + "=" * 70)
    print(f"{model_label.upper()} ANALYSIS: {embedding_name} + {dataset_type}")
    print(f"MODEL VARIANT: {model_variant}")
    print(f"PROTEIN INPUT: {protein_input_type}")
    print(f"LIGAND INPUT: {ligand_input_type}")
    print(f"SEEDS: {seeds}")
    print(f"SCENARIOS: {scenarios}")
    print(f"SCAFFOLD SPLIT DIR: {scaffold_split_dir}")
    if external_test_mode:
        print(f"EVALUATION MODE: EXTERNAL TEST (precomputed scaffold train/val)")
    else:
        print(f"EVALUATION MODE: INTERNAL TEST (precomputed train/val/test)")
    if optimize_threshold:
        print(f"DECISION THRESHOLD: OPTIMIZED ON VALIDATION ({threshold_metric})")
    else:
        print(f"DECISION THRESHOLD: FIXED ({fixed_threshold})")
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
        hidden_dim=hidden_dim,
        num_cnn_layers=num_cnn_layers,
        num_cross_attn_layers=num_cross_attn_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        diffusion_steps=diffusion_steps,
        diffusion_beta_start=diffusion_beta_start,
        diffusion_beta_end=diffusion_beta_end,
        diffusion_layers=diffusion_layers,
        diffusion_cross_attn_layers=diffusion_cross_attn_layers,
        diffusion_loss_weight=diffusion_loss_weight,
        diffusion_loss_anneal=diffusion_loss_anneal,
        classification_only=classification_only,
        diffusion_pool_queries=diffusion_pool_queries,
        diffusion_snr_sampling_gamma=diffusion_snr_sampling_gamma,
        diffusion_snr_sampling_mix=diffusion_snr_sampling_mix,
        diffusion_joint_denoise=diffusion_joint_denoise,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_cache_in_memory=dataloader_cache_in_memory,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        model_variant=model_variant,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        classification_weight=classification_weight,
        regression_weight=regression_weight,
        optimize_threshold=optimize_threshold,
        threshold_metric=threshold_metric,
        fixed_threshold=fixed_threshold,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis
    all_results, split_stats = run_crossattention_analysis(
        embedding_name, dataset_type, output_dir, config,
        seeds=seeds, prefix=prefix, use_attention=use_attention,
        scenarios=scenarios, use_molformer_ligand=use_molformer_ligand,
        use_ligand_vectors=use_ligand_vectors,
        scaffold_split_dir=scaffold_split_dir,
        external_test_mode=external_test_mode,
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
