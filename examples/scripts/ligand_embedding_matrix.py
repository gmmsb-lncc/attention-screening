#!/usr/bin/env python3
"""
Script para extrair embeddings de ligantes de um diretório de output existente.
Este script extrai apenas os embeddings vetoriais de ligantes (não as matrizes por token).
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
import shutil


def extract_ligand_embeddings(output_dir, output_path=None):
    """
    Extrai embeddings de ligantes de um diretório de output existente.
    
    Args:
        output_dir: Diretório de output existente do pipeline
        output_path: Caminho para salvar os embeddings extraídos (opcional)
    """
    print(f"🔍 Extraindo embeddings de ligantes de: {output_dir}")
    
    # Verificar se o diretório de output existe
    output_path_obj = Path(output_dir)
    if not output_path_obj.exists():
        print(f"❌ Diretório de output não encontrado: {output_dir}")
        return False
    
    # Localizar diretório de ligantes
    build_dir = output_path_obj / "build"
    if not build_dir.exists():
        print(f"❌ Diretório build não encontrado em: {build_dir}")
        return False
    
    ligand_dir = build_dir / "ligands"
    if not ligand_dir.exists():
        print(f"❌ Diretório de ligantes não encontrado: {ligand_dir}")
        return False
    
    print(f"📁 Diretório de ligantes encontrado: {ligand_dir}")
    
    # Encontrar todos os arquivos de embedding de ligantes
    embedding_files = list(ligand_dir.glob("*_embedding.npy"))
    print(f"📊 Encontrados {len(embedding_files)} arquivos de embedding de ligantes")
    
    if len(embedding_files) == 0:
        print("❌ Nenhum arquivo de embedding de ligantes encontrado")
        return False
    
    # Criar diretório de saída se não for especificado
    if output_path is None:
        output_path = output_path_obj / "extracted_ligand_embeddings"
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(exist_ok=True)
    print(f"📁 Diretório de saída: {output_path}")
    
    # Copiar ou extrair os embeddings
    extracted_count = 0
    for embedding_file in embedding_files:
        try:
            # Copiar o arquivo para o diretório de saída
            dest_file = output_path / embedding_file.name
            shutil.copy2(embedding_file, dest_file)
            
            # Carregar e verificar o embedding
            embedding = np.load(embedding_file)
            print(f"   ✅ {embedding_file.name}: shape {embedding.shape}")
            
            extracted_count += 1
            
        except Exception as e:
            print(f"   ❌ Erro ao processar {embedding_file.name}: {e}")
    
    print(f"\n✅ Extraídos {extracted_count} embeddings de ligantes")
    
    # Criar um arquivo com informações resumo
    summary_file = output_path / "ligand_embeddings_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Diretório de origem: {ligand_dir}\n")
        f.write(f"Total de embeddings extraídos: {extracted_count}\n")
        f.write(f"Formato: Vetorial (1D) - dimensão fixa de 768 para MoLFormer\n")
        f.write(f"\nLista de arquivos:\n")
        
        for embedding_file in sorted(embedding_files):
            try:
                embedding = np.load(embedding_file)
                f.write(f"  {embedding_file.name}: shape {embedding.shape}\n")
            except:
                f.write(f"  {embedding_file.name}: ERRO AO CARREGAR\n")
    
    print(f"📋 Resumo salvo em: {summary_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extrair embeddings de ligantes de diretório de output existente')
    parser.add_argument('--output-dir', required=True, 
                       help='Diretório de output existente do pipeline')
    parser.add_argument('--output-path', 
                       help='Caminho para salvar os embeddings extraídos (padrão: [output_dir]/extracted_ligand_embeddings)')
    
    args = parser.parse_args()
    
    # Mudar para o diretório do projeto se necessário
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = extract_ligand_embeddings(args.output_dir, args.output_path)
    
    if success:
        print(f"\n🎉 Extração de embeddings de ligantes completada com sucesso!")
        print(f"Os embeddings vetoriais de ligantes (768-dim) estão disponíveis em:")
        output_path = args.output_path or f"{args.output_dir}/extracted_ligand_embeddings"
        print(f"📁 {output_path}")
        print(f"\n💡 Dica: Estes embeddings podem ser usados diretamente para modelos de ML")
        print(f"   ou concatenados com embeddings de proteínas para modelos multimodais.")
    else:
        print(f"\n❌ Erro durante a extração.")
        sys.exit(1)


if __name__ == "__main__":
    main()