#!/usr/bin/env python3
"""
Script to run the complete pipeline with MoLFormer to generate ligand embedding matrices
for the non-human kinase compounds dataset.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_pipeline_with_molformer():
    """Run the complete pipeline with MoLFormer to generate ligand embedding matrices."""
    
    # Define the input dataset (first 100 rows of non-human compounds)
    input_dataset = Path("tests/datasets/kinase_non_human_compounds.tsv")
    temp_input = Path("temp_first_100.tsv")
    
    # Extract first 100 rows from the dataset
    print(f"🔍 Extracting first 100 rows from {input_dataset}")
    with open(input_dataset, 'r') as infile, open(temp_input, 'w') as outfile:
        for i, line in enumerate(infile):
            if i < 100:  # Header + 99 data rows
                outfile.write(line)
    
    print(f"✅ Created temporary dataset with first 100 rows: {temp_input}")
    
    # Define output directories for different protein models
    protein_models = [
        'esm2_t6_8M_UR50D',
        'esm2_t30_150M_UR50D', 
        'esm2_t33_650M_UR50D'
    ]
    
    # Run the pipeline for each protein model with MoLFormer
    for model in protein_models:
        output_dir = f"results/molformer_matrices_{model}"
        
        print(f"\n🚀 Running pipeline for {model} with MoLFormer...")
        print(f"   Input: {temp_input}")
        print(f"   Output: {output_dir}")
        
        cmd = [
            "python", "scripts/run_complete_pipeline.py",
            "--input", str(temp_input),
            "--output", output_dir,
            "--ligand-model", "MOLFORMER",  # Use MoLFormer
            "--protein-model", model,
            "--save-matrices",  # Enable saving of embedding matrices
            "--device", "cuda",  # Use GPU
            "--no-regression"  # Skip regression to speed up, keep classification
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Pipeline completed successfully for {model}")
            print(f"   Output saved to: {output_dir}")
            
            # Check if ligand matrices were generated
            ligand_matrix_dir = Path(output_dir) / "build" / "ligand_matrix_embeddings"
            if ligand_matrix_dir.exists():
                matrix_files = list(ligand_matrix_dir.glob("*.npy"))
                print(f"   Generated {len(matrix_files)} ligand embedding matrices")
            else:
                print(f"   ❌ Ligand matrix directory not found: {ligand_matrix_dir}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed for {model}")
            print(f"   Error: {e.stderr}")
            continue
    
    # Cleanup temporary file
    if temp_input.exists():
        temp_input.unlink()
        print(f"\n🧹 Cleaned up temporary file: {temp_input}")

def verify_matrices_exist():
    """Verify that the ligand embedding matrices were generated correctly."""
    print("\n🔍 Verifying generated ligand embedding matrices...")
    
    protein_models = [
        'esm2_t6_8M_UR50D',
        'esm2_t30_150M_UR50D', 
        'esm2_t33_650M_UR50D'
    ]
    
    for model in protein_models:
        matrix_dir = Path(f"results/molformer_matrices_{model}") / "build" / "ligand_matrix_embeddings"
        
        if matrix_dir.exists():
            matrix_files = list(matrix_dir.glob("*.npy"))
            print(f"   {model}: {len(matrix_files)} matrix files found")
            
            # Show info about first few matrices if they exist
            for i, matrix_file in enumerate(matrix_files[:3]):
                import numpy as np
                matrix = np.load(matrix_file)
                print(f"     {matrix_file.name}: shape {matrix.shape}")
        else:
            print(f"   {model}: ❌ Matrix directory not found")

if __name__ == "__main__":
    print("🧬 Starting pipeline with MoLFormer for ligand embedding matrices...")
    
    # Change to the project directory
    os.chdir("/media/storage/leon/attention-screening")
    
    # Activate environment and run pipeline
    run_pipeline_with_molformer()
    verify_matrices_exist()
    
    print("\n🎉 Pipeline execution completed!")
    print("The ligand embedding matrices (per-token representations) have been generated using MoLFormer.")
    print("These can now be used for cross-attention mechanisms alongside the protein matrices.")