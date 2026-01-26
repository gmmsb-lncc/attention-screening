#!/usr/bin/env python3
"""
Download and cache MoLFormer model.

This script downloads the DeepChem/MoLFormer-c3-1.1B model from Hugging Face
and caches it locally for offline use.

Usage:
    python download_model.py [--cache_dir PATH]
"""

import argparse
import os
import sys
from pathlib import Path


def download_molformer(cache_dir: str = None, force: bool = False):
    """
    Download MoLFormer model and tokenizer.

    Args:
        cache_dir: Directory to cache the model (default: ./models_cache)
        force: Force re-download even if model exists
    """
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError:
        print("ERROR: transformers library not installed.")
        print("Please install: pip install transformers torch")
        sys.exit(1)

    model_name = "DeepChem/MoLFormer-c3-1.1B"

    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "models_cache" / "molformer"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MoLFormer Model Download")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("-" * 60)

    # Check if already downloaded
    tokenizer_path = cache_dir / "tokenizer"
    model_path = cache_dir / "model"

    if tokenizer_path.exists() and model_path.exists() and not force:
        print("Model already downloaded. Use --force to re-download.")
        return str(model_path), str(tokenizer_path)

    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(str(tokenizer_path))
    print(f"  Saved to: {tokenizer_path}")

    print("\nDownloading model...")
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(str(model_path))
    print(f"  Saved to: {model_path}")

    # Print model info
    print("\n" + "-" * 60)
    print("Model Information:")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Max position embeddings: {model.config.max_position_embeddings}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print("-" * 60)

    print("\nDownload complete!")
    return str(model_path), str(tokenizer_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download MoLFormer model for ligand embeddings"
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory to cache the model'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if model exists'
    )

    args = parser.parse_args()

    download_molformer(cache_dir=args.cache_dir, force=args.force)


if __name__ == "__main__":
    main()
