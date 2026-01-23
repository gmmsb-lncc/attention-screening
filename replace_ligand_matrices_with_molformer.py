#!/usr/bin/env python3
"""
Script para substituir as matrizes de ligantes existentes pelas matrizes MoLFormer (representações por token).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def replace_ligand_matrices_with_molformer():
    """Substitui as matrizes de ligantes existentes pelas matrizes MoLFormer (representações por token)."""
    
    print("🔄 Substituindo matrizes de ligantes por matrizes MoLFormer (representações por token)...")
    
    # Carregar dataset para obter SMILES
    dataset_path = "tests/datasets/kinase_non_human_compounds.tsv"
    print(f"📁 Carregando dataset: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path, sep='\t')
    smiles_map = dict(zip(df['chembl_id'], df['canonical_smiles']))
    print(f"✅ Dataset carregado: {len(df)} entradas, {len(smiles_map)} SMILES únicos")
    
    # Diretórios de modelos de proteínas já existentes
    base_dir = "results/protein_model_benchmark_non_human_v2"
    protein_models = [
        "esm2_t6_8M_UR50D",      # 8M
        "esm2_t30_150M_UR50D",   # 150M  
        "esm2_t33_650M_UR50D"    # 650M
    ]
    
    # Carregar modelo MoLFormer
    print("🤖 Carregando modelo MoLFormer...")
    try:
        model_path = "llm/models_cache/molformer/model"
        tokenizer_path = "llm/models_cache/molformer/tokenizer"
        
        if not (Path(model_path).exists() and Path(tokenizer_path).exists()):
            print(f"❌ Modelos MoLFormer não encontrados em: {model_path} ou {tokenizer_path}")
            return False
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        print("✅ Modelo MoLFormer carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    # Processar cada modelo de proteína
    for model_name in protein_models:
        model_dir = Path(base_dir) / model_name
        if not model_dir.exists():
            print(f"⚠️  Diretório não encontrado: {model_dir}")
            continue
        
        print(f"\n--- Processando {model_name} ---")
        
        # Diretório de matrizes de ligantes existentes
        ligand_matrix_dir = model_dir / "build" / "ligand_matrices"
        if not ligand_matrix_dir.exists():
            print(f"⚠️  Diretório de matrizes de ligantes não encontrado: {ligand_matrix_dir}")
            continue
        
        # Diretório para novas matrizes MoLFormer
        molformer_matrix_dir = model_dir / "build" / "molformer_matrix"
        molformer_matrix_dir.mkdir(exist_ok=True)

        # Obter lista de arquivos de matrizes existentes
        matrix_files = list(ligand_matrix_dir.glob("*_matrix.npy"))
        print(f"🔍 Processando {len(matrix_files)} matrizes existentes...")

        processed = 0
        for matrix_file in matrix_files:
            # Extrair ID do arquivo (ex: CHEMBL1234567_matrix.npy -> CHEMBL1234567)
            chembl_id = matrix_file.name.replace("_matrix.npy", "")

            if chembl_id in smiles_map:
                smiles = smiles_map[chembl_id]

                try:
                    # Tokenizar SMILES
                    inputs = tokenizer(
                        smiles,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=512
                    )

                    # Gerar matriz por token
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)

                    # Obter representações por token (última camada de estados ocultos)
                    last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]
                    matrix = last_hidden_state.cpu().numpy()

                    # Salvar nova matriz MoLFormer
                    new_matrix_file = molformer_matrix_dir / f"{chembl_id}_molformer_matrix.npy"
                    np.save(new_matrix_file, matrix)

                    print(f"   ✅ {chembl_id}: shape {matrix.shape}")
                    processed += 1
                    
                except Exception as e:
                    print(f"   ❌ Erro ao processar {chembl_id}: {e}")
            else:
                print(f"   ⚠️  SMILES não encontrado para {chembl_id}")
        
        print(f"✅ {model_name}: {processed} matrizes MoLFormer geradas em {molformer_matrix_dir}")
    
    print(f"\n🎉 Processamento concluído!")
    print("As matrizes de ligantes MoLFormer (representações por token) foram geradas.")
    print("Cada matriz tem formato [n_tokens, 768] com representações por átomo/token.")
    return True

if __name__ == "__main__":
    os.chdir("/media/storage/leon/semantic-screening")
    success = replace_ligand_matrices_with_molformer()
    
    if success:
        print("\n✅ Sucesso! Matrizes de ligantes MoLFormer geradas com sucesso.")
        print("\n📁 As matrizes estão disponíveis em:")
        print("   results/protein_model_benchmark_non_human_v2/[model]/build/molformer_matrix/")
        print("\n💡 As matrizes estão prontas para mecanismos de atenção cruzada!")
    else:
        print("\n❌ Erro durante o processamento.")
        sys.exit(1)