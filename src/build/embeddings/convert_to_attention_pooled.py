"""
Convert per-token embedding matrices to attention-pooled vectors.

This script reads protein_matrices/ and ligand_matrices/ (or molformer_matrix/)
and generates attention-pooled vectors in protein_embeddings/ and ligand_embeddings/.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def attention_pooling(matrix: np.ndarray) -> np.ndarray:
    """
    Apply attention pooling to a per-token embedding matrix.
    
    Args:
        matrix: [seq_len, hidden_dim] numpy array
        
    Returns:
        vector: [hidden_dim] numpy array
    """
    # Convert to torch
    tensor = torch.from_numpy(matrix).float()  # [seq_len, hidden_dim]
    
    # Use mean vector as attention query
    query = tensor.mean(dim=0)  # [hidden_dim]
    
    # Compute attention scores
    attention_scores = torch.matmul(tensor, query)  # [seq_len]
    attention_weights = F.softmax(attention_scores, dim=0)  # [seq_len]
    
    # Weighted sum
    pooled = torch.sum(tensor * attention_weights.unsqueeze(-1), dim=0)  # [hidden_dim]
    
    return pooled.numpy()


def convert_directory(
    matrix_dir: str,
    output_dir: str,
    suffix: str = "_matrix.npy"
):
    """
    Convert all matrices in a directory to attention-pooled vectors.
    
    Args:
        matrix_dir: Directory containing *_matrix.npy files
        output_dir: Directory to save *_vector.npy files
        suffix: File suffix for matrix files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    matrix_files = sorted([f for f in os.listdir(matrix_dir) if f.endswith(suffix)])
    
    if not matrix_files:
        print(f"  No files found in {matrix_dir}")
        return 0
    
    print(f"  Converting {len(matrix_files)} files from {matrix_dir}")
    
    for filename in tqdm(matrix_files, desc="  Pooling"):
        matrix_path = os.path.join(matrix_dir, filename)
        
        # Load matrix
        matrix = np.load(matrix_path)
        
        # Apply attention pooling
        vector = attention_pooling(matrix)
        
        # Save vector with appropriate naming
        output_filename = filename.replace(suffix, "_vector.npy")
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, vector)
    
    return len(matrix_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert per-token matrices to attention-pooled vectors"
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory (e.g., results/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build/)"
    )
    parser.add_argument(
        "--protein",
        action="store_true",
        help="Convert protein matrices"
    )
    parser.add_argument(
        "--ligand",
        action="store_true",
        help="Convert ligand matrices (SMI-TED)"
    )
    parser.add_argument(
        "--molformer",
        action="store_true",
        help="Convert MolFormer matrices"
    )
    parser.add_argument(
        "--chemberta",
        action="store_true",
        help="Convert ChemBERTa matrices"
    )
    
    args = parser.parse_args()
    
    if not (args.protein or args.ligand or args.molformer or args.chemberta):
        print("Error: Specify at least one of --protein, --ligand, --molformer, --chemberta")
        sys.exit(1)
    
    base_dir = Path(args.base_dir)
    
    total_converted = 0
    
    if args.protein:
        print("\n[Protein] Converting matrices to attention-pooled vectors...")
        matrix_dir = base_dir / "protein_matrices"
        output_dir = base_dir / "protein_embeddings"
        count = convert_directory(str(matrix_dir), str(output_dir), suffix="_matrix.npy")
        total_converted += count
        print(f"  ✓ Converted {count} protein matrices\n")
    
    if args.ligand:
        print("\n[Ligand SMI-TED] Converting matrices to attention-pooled vectors...")
        matrix_dir = base_dir / "ligand_matrices"
        output_dir = base_dir / "ligand_embeddings"
        count = convert_directory(str(matrix_dir), str(output_dir), suffix="_matrix.npy")
        total_converted += count
        print(f"  ✓ Converted {count} ligand matrices\n")
    
    if args.molformer:
        print("\n[Ligand MolFormer] Converting matrices to attention-pooled vectors...")
        matrix_dir = base_dir / "molformer_matrix"
        output_dir = base_dir / "ligand_embeddings"
        count = convert_directory(str(matrix_dir), str(output_dir), suffix="_molformer_matrix.npy")
        total_converted += count
        print(f"  ✓ Converted {count} MolFormer matrices\n")
    
    if args.chemberta:
        print("\n[Ligand ChemBERTa] Converting matrices to attention-pooled vectors...")
        matrix_dir = base_dir / "chemberta_matrix"
        output_dir = base_dir / "ligand_embeddings"
        count = convert_directory(str(matrix_dir), str(output_dir), suffix="_chemberta_matrix.npy")
        total_converted += count
        print(f"  ✓ Converted {count} ChemBERTa matrices\n")
    
    print(f"[DONE] Total converted: {total_converted} files")


if __name__ == "__main__":
    main()
