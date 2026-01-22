#!/usr/bin/env python3
"""
Extract ligand embeddings using MoLFormer.

This script extracts both matrix and vector embeddings from SMILES using
the DeepChem/MoLFormer-c3-1.1B model.

Key advantage over SMI-TED:
- MoLFormer provides MATRIX embeddings [seq_len, 768] (per-token)
- SMI-TED only provides VECTOR embeddings [1, 768] (global)

Usage:
    # Extract embeddings for a single SMILES
    python extract_embeddings.py --smiles "CCO"

    # Extract embeddings from a file (one SMILES per line)
    python extract_embeddings.py --input smiles.txt --output embeddings.pt

    # Extract from TSV with chembl_id column
    python extract_embeddings.py --input compounds.tsv --smiles_column canonical_smiles --output_dir ./embeddings/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


class MoLFormerEmbedder:
    """
    MoLFormer-based embedder for SMILES molecules.

    Extracts both matrix embeddings (per-token) and vector embeddings (pooled).
    """

    MODEL_NAME = "DeepChem/MoLFormer-c3-1.1B"
    HIDDEN_DIM = 768
    MAX_LENGTH = 512

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize MoLFormer embedder.

        Args:
            model_path: Path to cached model (None = download from HuggingFace)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_cache: Whether to use local cache if available
        """
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading MoLFormer on {self.device}...")

        # Check for local cache
        cache_dir = Path(__file__).parent.parent / "models_cache" / "molformer"
        tokenizer_path = cache_dir / "tokenizer"
        local_model_path = cache_dir / "model"

        if use_cache and tokenizer_path.exists() and local_model_path.exists():
            print(f"  Using cached model from: {cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            self.model = AutoModelForMaskedLM.from_pretrained(str(local_model_path))
        elif model_path is not None:
            print(f"  Loading from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        else:
            print(f"  Downloading from HuggingFace: {self.MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForMaskedLM.from_pretrained(self.MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        print(f"  Hidden dimension: {self.model.config.hidden_size}")
        print(f"  Max sequence length: {self.model.config.max_position_embeddings}")

    def extract_matrix_embedding(
        self,
        smiles: str,
        include_special_tokens: bool = False
    ) -> torch.Tensor:
        """
        Extract matrix embedding for a single SMILES.

        Args:
            smiles: SMILES string
            include_special_tokens: Whether to include [CLS]/[SEP] tokens in output

        Returns:
            Matrix embedding of shape [seq_len, hidden_dim]
        """
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.MAX_LENGTH
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get last hidden state: [1, seq_len, hidden_dim]
        hidden_state = outputs.hidden_states[-1][0]

        if not include_special_tokens:
            # Remove [CLS] (first) and [SEP] (last) tokens
            hidden_state = hidden_state[1:-1]

        return hidden_state.cpu()

    def extract_vector_embedding(
        self,
        smiles: str,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Extract vector embedding for a single SMILES.

        Args:
            smiles: SMILES string
            pooling: Pooling strategy ('mean', 'max', 'cls')

        Returns:
            Vector embedding of shape [hidden_dim]
        """
        matrix = self.extract_matrix_embedding(smiles, include_special_tokens=(pooling == "cls"))

        if pooling == "mean":
            return matrix.mean(dim=0)
        elif pooling == "max":
            return matrix.max(dim=0)[0]
        elif pooling == "cls":
            return matrix[0]  # First token (CLS)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def extract_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        return_matrix: bool = True,
        pooling: str = "mean",
        show_progress: bool = True
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Extract embeddings for a batch of SMILES.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            return_matrix: If True, return matrix embeddings; if False, return vectors
            pooling: Pooling strategy for vector embeddings
            show_progress: Show progress bar

        Returns:
            If return_matrix: List of matrix embeddings (variable length)
            If not return_matrix: Tensor of shape [n_molecules, hidden_dim]
        """
        results = []

        iterator = range(0, len(smiles_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings")

        for i in iterator:
            batch = smiles_list[i:i + batch_size]

            if return_matrix:
                # Matrix embeddings (variable length, process one by one)
                for smi in batch:
                    try:
                        emb = self.extract_matrix_embedding(smi)
                        results.append(emb)
                    except Exception as e:
                        print(f"Warning: Failed to process '{smi}': {e}")
                        results.append(None)
            else:
                # Vector embeddings (batch processing)
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()

                if pooling == "mean":
                    # Masked mean pooling
                    embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                elif pooling == "max":
                    # Masked max pooling
                    hidden_states[attention_mask.squeeze(-1) == 0] = float('-inf')
                    embeddings = hidden_states.max(dim=1)[0]
                elif pooling == "cls":
                    embeddings = hidden_states[:, 0, :]

                results.extend(embeddings.cpu())

        if not return_matrix:
            results = torch.stack(results)

        return results


def save_embeddings_to_directory(
    embedder: MoLFormerEmbedder,
    smiles_dict: Dict[str, str],
    output_dir: str,
    save_matrix: bool = True,
    save_vector: bool = True,
    batch_size: int = 32
):
    """
    Save embeddings to directory (one file per molecule).

    Args:
        embedder: MoLFormerEmbedder instance
        smiles_dict: Dictionary mapping molecule IDs to SMILES
        output_dir: Output directory
        save_matrix: Whether to save matrix embeddings
        save_vector: Whether to save vector embeddings
        batch_size: Batch size for processing
    """
    output_dir = Path(output_dir)

    if save_matrix:
        matrix_dir = output_dir / "ligand_matrices_molformer"
        matrix_dir.mkdir(parents=True, exist_ok=True)

    if save_vector:
        vector_dir = output_dir / "ligand_vectors_molformer"
        vector_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting embeddings for {len(smiles_dict)} molecules...")

    for mol_id, smiles in tqdm(smiles_dict.items(), desc="Processing molecules"):
        try:
            if save_matrix:
                matrix_emb = embedder.extract_matrix_embedding(smiles)
                np.save(matrix_dir / f"{mol_id}.npy", matrix_emb.numpy())

            if save_vector:
                vector_emb = embedder.extract_vector_embedding(smiles)
                np.save(vector_dir / f"{mol_id}.npy", vector_emb.numpy())

        except Exception as e:
            print(f"Warning: Failed to process {mol_id}: {e}")

    print(f"\nEmbeddings saved to: {output_dir}")
    if save_matrix:
        print(f"  Matrix embeddings: {matrix_dir}")
    if save_vector:
        print(f"  Vector embeddings: {vector_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ligand embeddings using MoLFormer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single SMILES
  python extract_embeddings.py --smiles "CCO"

  # From file
  python extract_embeddings.py --input smiles.txt --output embeddings.pt

  # From TSV with specific columns
  python extract_embeddings.py --input data.tsv --id_column chembl_id --smiles_column canonical_smiles --output_dir ./embeddings/
        """
    )

    parser.add_argument(
        '--smiles', '-s',
        type=str,
        help='Single SMILES string to process'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file (txt with one SMILES per line, or TSV)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for embeddings (.pt or .npy)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for per-molecule embeddings'
    )

    parser.add_argument(
        '--id_column',
        type=str,
        default='chembl_id',
        help='Column name for molecule IDs in TSV (default: chembl_id)'
    )

    parser.add_argument(
        '--smiles_column',
        type=str,
        default='canonical_smiles',
        help='Column name for SMILES in TSV (default: canonical_smiles)'
    )

    parser.add_argument(
        '--matrix',
        action='store_true',
        help='Extract matrix embeddings (per-token)'
    )

    parser.add_argument(
        '--vector',
        action='store_true',
        help='Extract vector embeddings (pooled)'
    )

    parser.add_argument(
        '--pooling',
        choices=['mean', 'max', 'cls'],
        default='mean',
        help='Pooling strategy for vector embeddings (default: mean)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or auto)'
    )

    args = parser.parse_args()

    # Default to both matrix and vector if neither specified
    if not args.matrix and not args.vector:
        args.matrix = True
        args.vector = True

    # Initialize embedder
    embedder = MoLFormerEmbedder(device=args.device)

    # Process single SMILES
    if args.smiles:
        print(f"\nProcessing SMILES: {args.smiles}")

        if args.matrix:
            matrix = embedder.extract_matrix_embedding(args.smiles)
            print(f"Matrix embedding shape: {matrix.shape}")
            print(f"Matrix embedding (first 3 tokens):\n{matrix[:3]}")

        if args.vector:
            vector = embedder.extract_vector_embedding(args.smiles, pooling=args.pooling)
            print(f"Vector embedding shape: {vector.shape}")
            print(f"Vector embedding (first 10 dims): {vector[:10]}")

        return

    # Process file
    if args.input:
        import pandas as pd

        input_path = Path(args.input)

        if input_path.suffix in ['.tsv', '.csv']:
            sep = '\t' if input_path.suffix == '.tsv' else ','
            df = pd.read_csv(input_path, sep=sep)

            if args.id_column not in df.columns:
                print(f"ERROR: Column '{args.id_column}' not found in {input_path}")
                print(f"Available columns: {list(df.columns)}")
                sys.exit(1)

            if args.smiles_column not in df.columns:
                print(f"ERROR: Column '{args.smiles_column}' not found in {input_path}")
                print(f"Available columns: {list(df.columns)}")
                sys.exit(1)

            smiles_dict = dict(zip(df[args.id_column].astype(str), df[args.smiles_column]))

        else:
            # Assume one SMILES per line
            with open(input_path) as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            smiles_dict = {f"mol_{i}": smi for i, smi in enumerate(smiles_list)}

        print(f"Loaded {len(smiles_dict)} molecules from {input_path}")

        # Save to directory
        if args.output_dir:
            save_embeddings_to_directory(
                embedder,
                smiles_dict,
                args.output_dir,
                save_matrix=args.matrix,
                save_vector=args.vector,
                batch_size=args.batch_size
            )
        elif args.output:
            # Save to single file
            smiles_list = list(smiles_dict.values())
            embeddings = embedder.extract_batch(
                smiles_list,
                batch_size=args.batch_size,
                return_matrix=args.matrix,
                pooling=args.pooling
            )

            output_path = Path(args.output)
            if output_path.suffix == '.pt':
                torch.save(embeddings, output_path)
            else:
                if isinstance(embeddings, torch.Tensor):
                    np.save(output_path, embeddings.numpy())
                else:
                    np.save(output_path, embeddings, allow_pickle=True)

            print(f"Embeddings saved to: {output_path}")
        else:
            print("ERROR: Please specify --output or --output_dir")
            sys.exit(1)


if __name__ == "__main__":
    main()
