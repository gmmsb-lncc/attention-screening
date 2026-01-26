#!/usr/bin/env python3
"""
Script otimizado para gerar APENAS as matrizes de embedding de ligantes (MoLFormer) 
nos diretórios de resultados já existentes para não humanos.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def generate_molformer_matrices_only():
    """Gera APENAS as matrizes de ligantes (MoLFormer) para os diretórios existentes."""
    
    print("🔄 Gerando APENAS matrizes de ligantes MoLFormer para resultados existentes...")
    
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
        "esm2_t33_650M_UR50D"    # 650M (560M como mencionado)
    ]
    
    # Carregar modelo MoLFormer
    print("🤖 Carregando modelo MoLFormer...")
    try:
        model_path = "llm/models_cache/molformer/model"
        tokenizer_path = "llm/models_cache/molformer/tokenizer"
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        print("✅ Modelo MoLFormer carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    # Processar cada modelo de proteína
    for model in protein_models:
        model_dir = Path(base_dir) / model
        if not model_dir.exists():
            print(f"⚠️  Diretório não encontrado: {model_dir}")
            continue
        
        print(f"\n--- Processando {model} ---")
        
        # Criar diretório específico para matrizes MoLFormer
        molformer_matrix_dir = model_dir / "build" / "molformer_matrix"
        molformer_matrix_dir.mkdir(exist_ok=True)
        
        # Obter lista de arquivos de ligantes existentes para saber quais IDs processar
        ligand_dir = model_dir / "build" / "ligands"
        if not ligand_dir.exists():
            print(f"⚠️  Diretório de ligantes não encontrado: {ligand_dir}")
            continue
        
        embedding_files = list(ligand_dir.glob("*_embedding.npy"))
        print(f"🔍 Processando {len(embedding_files)} ligantes existentes...")
        
        processed = 0
        for embedding_file in embedding_files:
            # Extrair ID do arquivo (ex: CHEMBL1234567_embedding.npy -> CHEMBL1234567)
            chembl_id = embedding_file.name.replace("_embedding.npy", "")
            
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
                    
                    # Obter representações por token
                    last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]
                    matrix = last_hidden_state.cpu().numpy()
                    
                    # Salvar matriz
                    matrix_file = molformer_matrix_dir / f"{chembl_id}_molformer_matrix.npy"
                    np.save(matrix_file, matrix)
                    
                    print(f"   ✅ {chembl_id}: {matrix.shape}")
                    processed += 1
                    
                except Exception as e:
                    print(f"   ❌ Erro com {chembl_id}: {e}")
            else:
                print(f"   ⚠️  SMILES não encontrado para {chembl_id}")
        
        print(f"✅ {model}: {processed} matrizes geradas em {molformer_matrix_dir}")
    
    print(f"\n🎉 Processamento concluído!")
    print("As matrizes de ligantes MoLFormer foram geradas nos diretórios especificados.")
    return True

if __name__ == "__main__":
    os.chdir("/media/storage/leon/semantic-screening")
    success = generate_molformer_matrices_only()
    
    if success:
        print("\n✅ Sucesso! Matrizes de ligantes MoLFormer geradas com sucesso.")
    else:
        print("\n❌ Erro durante o processamento.")
        sys.exit(1)