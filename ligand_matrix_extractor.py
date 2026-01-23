#!/usr/bin/env python3
"""
Script para extrair matrizes de embedding de ligantes (representações por token) de um diretório de output existente.
Este script extrai as matrizes de ligantes que contêm representações por token/átomo.
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
import shutil


def extract_ligand_matrices(output_dir, output_path=None):
    """
    Extrai matrizes de ligantes (representações por token) de um diretório de output existente.
    
    Args:
        output_dir: Diretório de output existente do pipeline
        output_path: Caminho para salvar as matrizes extraídas (opcional)
    """
    print(f"🔍 Extraindo matrizes de ligantes (representações por token) de: {output_dir}")
    
    # Verificar se o diretório de output existe
    output_path_obj = Path(output_dir)
    if not output_path_obj.exists():
        print(f"❌ Diretório de output não encontrado: {output_dir}")
        return False
    
    # Localizar diretório de matrizes de ligantes
    build_dir = output_path_obj / "build"
    if not build_dir.exists():
        print(f"❌ Diretório build não encontrado em: {build_dir}")
        return False
    
    # Verificar se o diretório de matrizes de ligantes existe
    ligand_matrix_dir = build_dir / "ligand_molformer_matrices"
    if not ligand_matrix_dir.exists():
        print(f"❌ Diretório de matrizes de ligantes não encontrado: {ligand_matrix_dir}")
        print("💡 Dica: Execute o pipeline com --save-matrices para gerar as matrizes de ligantes")
        return False
    
    print(f"📁 Diretório de matrizes de ligantes encontrado: {ligand_matrix_dir}")
    
    # Encontrar todos os arquivos de matriz de ligantes
    matrix_files = list(ligand_matrix_dir.glob("*_matrix.npy"))
    print(f"📊 Encontrados {len(matrix_files)} arquivos de matriz de ligantes")
    
    if len(matrix_files) == 0:
        print("❌ Nenhuma matriz de ligantes encontrada")
        return False
    
    # Criar diretório de saída
    if output_path is None:
        output_path = output_path_obj / "extracted_ligand_matrices"
    else:
        output_path = Path(output_path)
    
    output_path.mkdir(exist_ok=True)
    print(f"📁 Diretório de saída: {output_path}")
    
    # Copiar as matrizes
    extracted_count = 0
    total_tokens = 0
    
    for matrix_file in matrix_files:
        try:
            # Copiar o arquivo para o diretório de saída
            dest_file = output_path / matrix_file.name
            shutil.copy2(matrix_file, dest_file)
            
            # Carregar e verificar a matriz
            matrix = np.load(matrix_file)
            print(f"   ✅ {matrix_file.name}: shape {matrix.shape}")
            
            # Contar tokens totais
            total_tokens += matrix.shape[0]
            extracted_count += 1
            
        except Exception as e:
            print(f"   ❌ Erro ao processar {matrix_file.name}: {e}")
    
    print(f"\n✅ Extraídas {extracted_count} matrizes de ligantes")
    print(f"📊 Total de tokens processados: {total_tokens}")
    
    # Criar um arquivo com informações resumo
    summary_file = output_path / "ligand_matrices_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Diretório de origem: {ligand_matrix_dir}\n")
        f.write(f"Total de matrizes extraídas: {extracted_count}\n")
        f.write(f"Total de tokens: {total_tokens}\n")
        f.write(f"Formato: Matricial (2D) - [n_tokens, 768] para MoLFormer\n")
        f.write(f"Dimensão do embedding: 768 (fixa para MoLFormer)\n")
        f.write(f"\nDescrição: Cada matriz contém representações por token/átomo do SMILES,\n")
        f.write(f"onde cada linha representa um token e cada coluna uma dimensão do embedding.\n")
        f.write(f"Estas matrizes podem ser usadas para mecanismos de atenção cruzada.\n")
        f.write(f"\nLista de arquivos:\n")
        
        for matrix_file in sorted(matrix_files):
            try:
                matrix = np.load(matrix_file)
                f.write(f"  {matrix_file.name}: shape {matrix.shape} ({matrix.shape[0]} tokens x {matrix.shape[1]} dim)\n")
            except:
                f.write(f"  {matrix_file.name}: ERRO AO CARREGAR\n")
    
    print(f"📋 Resumo salvo em: {summary_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extrair matrizes de ligantes (representações por token) de diretório de output existente')
    parser.add_argument('--output-dir', required=True, 
                       help='Diretório de output existente do pipeline')
    parser.add_argument('--output-path', 
                       help='Caminho para salvar as matrizes extraídas (padrão: [output_dir]/extracted_ligand_matrices)')
    
    args = parser.parse_args()
    
    # Mudar para o diretório do projeto se necessário
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    success = extract_ligand_matrices(args.output_dir, args.output_path)
    
    if success:
        print(f"\n🎉 Extração de matrizes de ligantes completada com sucesso!")
        print(f"As matrizes de ligantes (representações por token) estão disponíveis em:")
        output_path = args.output_path or f"{args.output_dir}/extracted_ligand_matrices"
        print(f"📁 {output_path}")
        print(f"\n💡 Dica: Estas matrizes contêm representações por token/átomo e podem ser usadas")
        print(f"   em mecanismos de atenção cruzada com as matrizes de proteínas.")
        print(f"   Cada matriz tem formato [n_tokens, 768] onde n_tokens varia por composto.")
    else:
        print(f"\n❌ Erro durante a extração.")
        sys.exit(1)


if __name__ == "__main__":
    main()