"""ESM-2 and MolFormer fine-tuning orchestration.

Handles the optional fine-tuning step that trains PLMs on kinase data
and regenerates embeddings with the fine-tuned weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

from benchmark.config import BenchmarkConfig


# ---------------------------------------------------------------------------
# ESM-2 fine-tuning
# ---------------------------------------------------------------------------

def finetune_esm(
    config: BenchmarkConfig,
    *,
    dataset: str,
    train_tsv: str,
    val_tsv: str,
    output_dir: str,
) -> Optional[str]:
    """Fine-tune ESM-2 on kinase training sequences.

    Returns the path to the best checkpoint, or ``None`` on failure.
    """
    import pandas as pd
    import torch
    from src.finetuning.esm_finetuner import ESMFinetuner

    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"ESM-2 Fine-tuning on {dataset} training set")
    tqdm.write("=" * 70)

    ft_dir = Path(output_dir) / f"level4_finetuned_{config.embedding_name}"
    ft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ft_dir / "best_model.pt"

    if checkpoint_path.exists() and not config.force:
        tqdm.write(f"  [OK] Fine-tuned model already exists: {checkpoint_path}")
        return str(checkpoint_path)

    # Load training data
    try:
        df_train = _read_tsv(train_tsv)
    except Exception as exc:
        tqdm.write(f"  ERROR loading training data: {exc}")
        return None

    if "seq" not in df_train.columns or "seq_id" not in df_train.columns:
        tqdm.write("  ERROR: Training TSV must have 'seq' and 'seq_id' columns")
        return None

    df_unique = df_train[["seq_id", "seq"]].drop_duplicates(subset=["seq_id"])
    tqdm.write(f"  Training proteins: {len(df_unique)} unique sequences")
    tqdm.write(f"  Model: {config.embedding_name}")
    tqdm.write(f"  Epochs: {config.finetune_epochs}, BS: {config.finetune_batch_size}, LR: {config.finetune_lr}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        finetuner = ESMFinetuner(model_name=config.embedding_name, device=device, mask_prob=0.15)
    except Exception as exc:
        tqdm.write(f"  ERROR initializing fine-tuner: {exc}")
        return None

    try:
        train_loader, val_loader = finetuner.prepare_data(
            train_tsv=train_tsv,
            val_tsv=val_tsv,
            batch_size=config.finetune_batch_size,
            max_length=1024,
        )
    except Exception as exc:
        tqdm.write(f"  ERROR preparing data: {exc}")
        return None

    try:
        history = finetuner.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.finetune_epochs,
            learning_rate=config.finetune_lr,
            warmup_steps=100,
            gradient_accumulation_steps=4,
            save_path=str(checkpoint_path),
            patience=config.resolved_patience or 3,
        )
        tqdm.write(f"  Fine-tuning completed! Epochs: {len(history['train_loss'])}")
        if history["val_loss"]:
            tqdm.write(f"  Best val loss: {min(history['val_loss']):.4f}")
        return str(checkpoint_path)
    except Exception as exc:
        tqdm.write(f"  ERROR during fine-tuning: {exc}")
        return None


# ---------------------------------------------------------------------------
# MolFormer fine-tuning
# ---------------------------------------------------------------------------

def finetune_molformer(
    config: BenchmarkConfig,
    *,
    dataset: str,
    train_tsv: str,
    val_tsv: str,
    output_dir: str,
) -> Optional[str]:
    """Fine-tune MolFormer on kinase ligand training data.

    Returns the path to the best checkpoint, or ``None`` on failure.
    """
    import pandas as pd
    import torch
    from src.finetuning.molformer_finetuner import MolFormerFinetuner

    tqdm.write("\n" + "-" * 70)
    tqdm.write(f"MolFormer Fine-tuning on {dataset} training set")
    tqdm.write("-" * 70)

    ft_dir = Path(output_dir) / "level4_finetuned_molformer"
    ft_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ft_dir / "best_model"

    if checkpoint_path.exists() and not config.force:
        tqdm.write(f"  [OK] Fine-tuned MolFormer already exists: {checkpoint_path}")
        return str(checkpoint_path)

    try:
        df_train = _read_tsv(train_tsv)
    except Exception as exc:
        tqdm.write(f"  ERROR loading training data: {exc}")
        return None

    if "smiles" not in df_train.columns or "chembl_id" not in df_train.columns:
        tqdm.write("  ERROR: Training TSV must have 'smiles' and 'chembl_id' columns")
        return None

    df_unique = df_train[["chembl_id", "smiles"]].drop_duplicates(subset=["chembl_id"])
    batch_size = config.finetune_batch_size * 2
    learning_rate = config.finetune_lr * 2

    tqdm.write(f"  Training ligands: {len(df_unique)} unique molecules")
    tqdm.write(f"  Epochs: {config.finetune_epochs}, BS: {batch_size}, LR: {learning_rate}")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        finetuner = MolFormerFinetuner(
            model_path="ibm/MoLFormer-XL-both-10pct",
            device=device,
            mask_prob=0.15,
            use_amp=True,
        )
    except Exception as exc:
        tqdm.write(f"  ERROR initializing MolFormer fine-tuner: {exc}")
        return None

    try:
        train_loader, val_loader = finetuner.prepare_data(
            train_tsv=train_tsv,
            val_tsv=val_tsv,
            batch_size=batch_size,
            max_length=202,
        )
    except Exception as exc:
        tqdm.write(f"  ERROR preparing data: {exc}")
        return None

    try:
        history = finetuner.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.finetune_epochs,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            gradient_accumulation_steps=4,
            save_path=str(checkpoint_path),
            patience=config.resolved_patience or 3,
        )
        tqdm.write(f"  MolFormer fine-tuning completed! Epochs: {len(history['train_loss'])}")
        if history["val_loss"]:
            tqdm.write(f"  Best val loss: {min(history['val_loss']):.4f}")
        return str(checkpoint_path)
    except Exception as exc:
        tqdm.write(f"  ERROR during MolFormer fine-tuning: {exc}")
        return None


# ---------------------------------------------------------------------------
# Embedding regeneration
# ---------------------------------------------------------------------------

def regenerate_protein_embeddings(
    config: BenchmarkConfig,
    *,
    dataset: str,
    checkpoint: str,
    output_dir: str,
) -> str:
    """Regenerate protein embeddings using a fine-tuned ESM-2 checkpoint.

    Returns path to the fine-tuned embedding base directory.
    """
    import torch
    from src.finetuning.esm_finetuner import ESMFinetuner

    tqdm.write("    Regenerating protein embeddings with fine-tuned model...")
    finetuner = ESMFinetuner(
        model_name=config.embedding_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_prob=0.15,
    )
    finetuner.load_model(checkpoint)

    finetuned_base = Path(output_dir) / "finetuned_embeddings" / dataset
    finetuned_base.mkdir(parents=True, exist_ok=True)

    for split, subdir in _split_paths(config.scaffold_split_dir, dataset):
        if subdir.exists():
            tqdm.write(f"    Extracting embeddings for {split} split...")
            finetuner.extract_embeddings(
                tsv_file=str(subdir),
                output_dir=str(finetuned_base),
                batch_size=8,
                repr_layer=-1,
                save_matrices=True,
                save_vectors=True,
            )
        else:
            tqdm.write(f"    WARNING: {split} split not found: {subdir}")

    return str(finetuned_base)


def regenerate_ligand_embeddings(
    config: BenchmarkConfig,
    *,
    dataset: str,
    checkpoint: str,
    output_dir: str,
) -> str:
    """Regenerate ligand embeddings using a fine-tuned MolFormer checkpoint.

    Returns path to the fine-tuned embedding base directory.
    """
    import torch
    from src.finetuning.molformer_finetuner import MolFormerFinetuner

    tqdm.write("    Regenerating ligand embeddings with fine-tuned MolFormer...")
    finetuner = MolFormerFinetuner(
        model_path=checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mask_prob=0.15,
    )

    finetuned_base = Path(output_dir) / "finetuned_embeddings" / dataset
    finetuned_base.mkdir(parents=True, exist_ok=True)

    for split, subdir in _split_paths(config.scaffold_split_dir, dataset):
        if subdir.exists():
            tqdm.write(f"    Extracting ligand embeddings for {split} split...")
            finetuner.extract_embeddings(
                tsv_file=str(subdir),
                output_dir=str(finetuned_base),
                batch_size=32,
                save_matrices=True,
                save_vectors=True,
            )
        else:
            tqdm.write(f"    WARNING: {split} split not found: {subdir}")

    return str(finetuned_base)


# ---------------------------------------------------------------------------
# Full fine-tuning pipeline
# ---------------------------------------------------------------------------

def run_finetuning_pipeline(
    config: BenchmarkConfig,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Run the full ESM-2 + MolFormer fine-tuning pipeline.

    Returns a nested dict:
    ``{dataset: {"esm_checkpoint": str|None, "molformer_checkpoint": str|None}}``
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}

    for ds in config.datasets_to_process():
        tqdm.write(f"\n  {'=' * 60}")
        tqdm.write(f"  Fine-tuning for dataset: {ds}")
        tqdm.write(f"  {'=' * 60}")

        train_tsv, val_tsv = _resolve_split_tsvs(config.scaffold_split_dir, ds)
        if not train_tsv or not val_tsv:
            continue

        ds_output = _dataset_output_dir(config, ds)
        ds_result: Dict[str, Optional[str]] = {"esm_checkpoint": None, "molformer_checkpoint": None}

        # ESM-2
        esm_ckpt = finetune_esm(config, dataset=ds, train_tsv=train_tsv, val_tsv=val_tsv, output_dir=ds_output)
        if esm_ckpt:
            ds_result["esm_checkpoint"] = esm_ckpt
            try:
                regenerate_protein_embeddings(config, dataset=ds, checkpoint=esm_ckpt, output_dir=ds_output)
            except Exception as exc:
                tqdm.write(f"    ERROR regenerating protein embeddings: {exc}")

        # MolFormer
        mol_ckpt = finetune_molformer(config, dataset=ds, train_tsv=train_tsv, val_tsv=val_tsv, output_dir=ds_output)
        if mol_ckpt:
            ds_result["molformer_checkpoint"] = mol_ckpt
            try:
                regenerate_ligand_embeddings(config, dataset=ds, checkpoint=mol_ckpt, output_dir=ds_output)
            except Exception as exc:
                tqdm.write(f"    ERROR regenerating ligand embeddings: {exc}")

        results[ds] = ds_result

    # Print summary
    tqdm.write(f"\n  {'=' * 60}")
    tqdm.write("  Fine-tuning Summary:")
    tqdm.write(f"  {'=' * 60}")
    for ds, ckpts in results.items():
        esm_ok = "OK" if ckpts["esm_checkpoint"] else "FAIL"
        mol_ok = "OK" if ckpts["molformer_checkpoint"] else "FAIL"
        tqdm.write(f"    {ds}: ESM-2 [{esm_ok}] | MolFormer [{mol_ok}]")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_tsv(path: str) -> "pd.DataFrame":
    """Read a TSV file, handling ``.gz`` transparently."""
    import pandas as pd

    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt") as fh:
            return pd.read_csv(fh, sep="\t")
    return pd.read_csv(path, sep="\t")


def _resolve_split_tsvs(
    scaffold_split_dir: str,
    dataset: str,
) -> tuple[Optional[str], Optional[str]]:
    """Find universal train and val TSV paths, optionally filtered by dataset.

    Always routes through ``universal_train.tsv`` / ``universal_val.tsv``.
    When *dataset* is ``'human'`` or ``'non_human'``, writes a filtered
    copy so downstream finetuning code can consume a plain TSV path.
    """
    import tempfile

    base = os.path.join(scaffold_split_dir, "scenarios", "Sc")

    train_tsv = _find_file(base, "universal_train.tsv")
    val_tsv = _find_file(base, "universal_val.tsv")

    if not train_tsv:
        tqdm.write("    ERROR: Universal training TSV not found")
    if not val_tsv:
        tqdm.write("    ERROR: Universal validation TSV not found")

    if not train_tsv or not val_tsv:
        return None, None

    # For 'all', the full universal file is used as-is
    if dataset == "all":
        return train_tsv, val_tsv

    # For a specific corpus, filter and write to a temp file
    import pandas as pd

    for label, src_path in [("train", train_tsv), ("val", val_tsv)]:
        df = pd.read_csv(src_path, sep="\t")
        if "dataset_source" in df.columns:
            filtered = df[df["dataset_source"] == dataset]
            filtered_path = os.path.join(base, f"_ft_{dataset}_{label}.tsv")
            filtered.to_csv(filtered_path, sep="\t", index=False)
            if label == "train":
                train_tsv = filtered_path
            else:
                val_tsv = filtered_path

    return train_tsv, val_tsv


def _find_file(directory: str, basename: str) -> Optional[str]:
    """Find a file trying both plain and ``.gz`` variants."""
    path = os.path.join(directory, basename)
    if os.path.exists(path):
        return path
    gz = path + ".gz"
    if os.path.exists(gz):
        return gz
    return None


def _split_paths(
    scaffold_split_dir: str,
    dataset: str,
) -> list[tuple[str, Path]]:
    """Return (split_name, path) tuples for train/val/test."""
    paths = []
    for split in ("train", "val"):
        p = Path(scaffold_split_dir) / "scenarios" / "Sc" / f"{dataset}_{split}.tsv.gz"
        if not p.exists():
            p = p.with_suffix("")
        paths.append((split, p))

    test_p = Path(scaffold_split_dir) / f"{dataset}_test.tsv.gz"
    if not test_p.exists():
        test_p = test_p.with_suffix("")
    paths.append(("test", test_p))

    return paths


def _dataset_output_dir(config: BenchmarkConfig, dataset: str) -> str:
    """Compute per-dataset output directory for fine-tuning."""
    base = config.resolved_output_dir
    if config.dataset == "all":
        return base.replace("benchmark_all", f"benchmark_{dataset}")
    return base
