#!/usr/bin/env python3
"""
Script para gerar matrizes de embedding de ligantes (MoLFormer) para um diretório de output existente.
Este script extrai os SMILES do dataset original e gera as matrizes de ligantes por token.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def generate_ligand_matrices_from_existing_output(output_dir, dataset_path="tests/datasets/kinase_non_human_compounds.tsv"):
    """
    Gera matrizes de ligantes para um diretório de output existente.
    
    Args:
        output_dir: Diretório de output existente do pipeline
        dataset_path: Caminho para o dataset original com SMILES
    """
    print(f"🔄 Gerando matrizes de ligantes para: {output_dir}")
    
    # Verificar se o diretório de output existe
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ Diretório de output não encontrado: {output_dir}")
        return False
    
    # Carregar o dataset original para obter os SMILES
    print(f"📁 Carregando dataset: {dataset_path}")
    if not Path(dataset_path).exists():
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path, sep='\t')
    print(f"📊 Dataset carregado com {len(df)} entradas")
    
    # Extrair mapeamento de chembl_id para SMILES
    smiles_map = dict(zip(df['chembl_id'], df['canonical_smiles']))
    print(f"🔍 Encontrados SMILES para {len(smiles_map)} compostos únicos")
    
    # Criar diretório para matrizes de ligantes
    build_dir = output_path / "build"
    if not build_dir.exists():
        print(f"❌ Diretório build não encontrado em: {build_dir}")
        return False
    
    matrix_dir = build_dir / "ligand_molformer_matrices"
    matrix_dir.mkdir(exist_ok=True)
    print(f"📁 Diretório de matrizes criado: {matrix_dir}")
    
    # Carregar o modelo MoLFormer
    print("🤖 Carregando modelo MoLFormer...")
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        
        # Caminho para o modelo MoLFormer
        model_cache_path = Path(__file__).parent / "llm" / "models_cache" / "molformer"
        model_path = model_cache_path / "model"
        tokenizer_path = model_cache_path / "tokenizer"
        
        if not (model_path.exists() and tokenizer_path.exists()):
            print(f"❌ Modelos MoLFormer não encontrados em: {model_cache_path}")
            print("💡 Execute: python llm/MoLFormer/download_model.py")
            return False
        
        # Carregar tokenizer e modelo
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        
        print("✅ Modelo MoLFormer carregado com sucesso")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo MoLFormer: {e}")
        return False
    
    # Processar cada arquivo de embedding de ligante existente para extrair os IDs
    ligand_dir = build_dir / "ligands"
    if ligand_dir.exists():
        embedding_files = list(ligand_dir.glob("*_embedding.npy"))
        print(f"🔍 Processando {len(embedding_files)} arquivos de embedding de ligantes existentes")
        
        processed = 0
        for embedding_file in embedding_files:
            # Extrair ID do arquivo (ex: CHEMBL1234567_embedding.npy -> CHEMBL1234567)
            chembl_id = embedding_file.name.replace("_embedding.npy", "")
            
            if chembl_id in smiles_map:
                smiles = smiles_map[chembl_id]
                
                # Gerar matriz de embedding por token
                try:
                    inputs = tokenizer(
                        smiles,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                    
                    # Obter última camada de estados ocultos (representações por token)
                    last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]
                    
                    # Converter para numpy array
                    matrix = last_hidden_state.cpu().numpy()
                    
                    # Salvar matriz
                    matrix_file = matrix_dir / f"{chembl_id}_matrix.npy"
                    np.save(matrix_file, matrix)
                    
                    print(f"   ✅ {chembl_id}: shape {matrix.shape}")
                    processed += 1
                    
                except Exception as e:
                    print(f"   ❌ Erro ao processar {chembl_id}: {e}")
            else:
                print(f"   ⚠️  SMILES não encontrado para {chembl_id}")
        
        print(f"✅ Processados {processed} matrizes de ligantes")
    else:
        print(f"⚠️  Diretório de ligantes não encontrado: {ligand_dir}")
        # Tentar obter os IDs de outra forma
        print("💡 Dica: Verifique se o pipeline foi executado com sucesso antes")
        return False
    
    print(f"\n🎉 Matrizes de ligantes geradas com sucesso!")
    print(f"📁 Local: {matrix_dir}")
    print(f"📊 Total de matrizes geradas: {len(list(matrix_dir.glob('*.npy')))}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Gerar matrizes de ligantes para diretório de output existente')
    parser.add_argument('--output-dir', required=True, help='Diretório de output existente do pipeline')
    parser.add_argument('--dataset', default='tests/datasets/kinase_non_human_compounds.tsv', 
                       help='Caminho para o dataset original (padrão: tests/datasets/kinase_non_human_compounds.tsv)')
    
    args = parser.parse_args()
    
    # Mudar para o diretório do projeto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = generate_ligand_matrices_from_existing_output(args.output_dir, args.dataset)
    
    if success:
        print(f"\n✅ Execução completada com sucesso!")
        print(f"As matrizes de ligantes estão prontas para uso em mecanismos de atenção cruzada.")
    else:
        print(f"\n❌ Erro durante a execução.")
        sys.exit(1)

if __name__ == "__main__":
    main()