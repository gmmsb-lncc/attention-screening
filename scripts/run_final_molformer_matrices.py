#!/usr/bin/env python3
"""
Script final para gerar matrizes de embedding de ligantes usando MoLFormer
para os modelos de proteínas 8M, 150M e 650M no dataset de compostos não humanos.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_molformer_for_protein_models():
    """Executar o pipeline com MoLFormer para diferentes modelos de proteínas."""
    
    print("🔄 Iniciando pipeline para gerar matrizes de ligantes com MoLFormer...")
    
    # Caminho para o dataset de não humanos
    dataset_path = "tests/datasets/kinase_non_human_compounds.tsv"
    
    # Verificar se o dataset existe
    if not Path(dataset_path).exists():
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False
    
    print(f"📁 Dataset encontrado: {dataset_path}")
    
    # Modelos de proteína para testar (8M, 150M, 650M como solicitado)
    protein_models = [
        'esm2_t6_8M_UR50D',      # 8M parameters
        'esm2_t30_150M_UR50D',   # 150M parameters  
        'esm2_t33_650M_UR50D'    # 650M parameters
    ]
    
    # Executar o pipeline para cada modelo de proteína
    for model in protein_models:
        output_dir = f"results/protein_model_benchmark_non_human_v2_with_molformer_{model}"
        
        print(f"\n🚀 Executando pipeline para {model} com MoLFormer...")
        print(f"   Modelo de proteína: {model}")
        print(f"   Modelo de ligante: MOLFORMER")
        print(f"   Saída: {output_dir}")
        
        # Comando para executar o pipeline
        cmd = [
            "python", "run_complete_pipeline.py",
            "--input", dataset_path,
            "--output", output_dir,
            "--ligand-model", "MOLFORMER",      # Usar MoLFormer
            "--protein-model", model,
            "--save-matrices",                 # Salvar matrizes
            "--device", "cuda",                # Usar GPU
            "--no-regression",                 # Apenas classificação para acelerar
            "--no-classification"              # Pular classificação para acelerar
        ]
        
        try:
            print(f"   Executando comando...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Pipeline completado com sucesso para {model}")
            
            # Verificar se as matrizes de ligantes foram geradas
            ligand_matrix_dir = Path(output_dir) / "build" / "ligand_molformer_matrices"
            if ligand_matrix_dir.exists():
                matrix_files = list(ligand_matrix_dir.glob("*.npy"))
                print(f"   ✅ {len(matrix_files)} matrizes de ligantes geradas em: {ligand_matrix_dir}")
                
                # Mostrar informações sobre algumas matrizes
                for i, matrix_file in enumerate(matrix_files[:3]):  # Mostrar as 3 primeiras
                    import numpy as np
                    matrix = np.load(matrix_file)
                    print(f"     {matrix_file.name}: shape {matrix.shape}")
            else:
                print(f"   ❌ Diretório de matrizes de ligantes não encontrado: {ligand_matrix_dir}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline falhou para {model}")
            print(f"   Erro: {e.stderr}")
            continue
    
    print(f"\n🎉 Pipeline concluído!")
    print("As matrizes de ligantes (representações por token) foram geradas usando MoLFormer.")
    print("Estas podem ser usadas para mecanismos de atenção cruzada juntamente com as matrizes de proteínas.")
    return True

if __name__ == "__main__":
    print("🧬 Gerando matrizes de ligantes com MoLFormer para o benchmark de proteínas...")
    
    # Mudar para o diretório do projeto
    os.chdir("/media/storage/leon/semantic-screening")
    
    success = run_molformer_for_protein_models()
    
    if success:
        print("\n✅ Execução completada com sucesso!")
        print("As matrizes de ligantes estão disponíveis nos diretórios especificados.")
        print("\n📁 Estrutura de saída:")
        print("   results/protein_model_benchmark_non_human_v2_with_molformer_[model]/")
        print("   └── build/")
        print("       ├── ligand_molformer_matrices/  ← Matrizes de ligantes por token")
        print("       ├── ligands/                    ← Embeddings vetoriais de ligantes")
        print("       ├── proteins/                   ← Embeddings de proteínas")
        print("       └── embedding_matrix.npy        ← Matriz concatenada")
    else:
        print("\n❌ Erro durante a execução.")
        sys.exit(1)