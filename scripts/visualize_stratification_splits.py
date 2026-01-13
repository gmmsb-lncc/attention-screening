#!/usr/bin/env python3
"""
Script auxiliar para visualizar splits da estratificação.

Este script facilita a visualização dos resultados da estratificação,
automaticamente localizando e carregando os arquivos gerados pelo pipeline.

Usage:
    # Visualizar splits de uma execução específica
    python scripts/visualize_stratification_splits.py --run-dir results/stratification_20241208/
    
    # Especificar arquivos manualmente
    python scripts/visualize_stratification_splits.py \\
        --embeddings data/embeddings.npy \\
        --train-idx results/train_indices.npy \\
        --val-idx results/val_indices.npy \\
        --test-idx results/test_indices.npy
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import subprocess

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_stratification_files(run_dir: Path) -> dict:
    """
    Procura arquivos de estratificação em um diretório.
    
    Args:
        run_dir: Diretório contendo resultados da estratificação
        
    Returns:
        Dicionário com caminhos dos arquivos encontrados
    """
    files = {
        'embeddings': None,
        'protein_emb': None,
        'ligand_emb': None,
        'labels': None,
        'train_idx': None,
        'val_idx': None,
        'test_idx': None,
        'cluster_labels': None,
    }
    
    # Padrões de busca
    patterns = {
        'embeddings': ['*embeddings*.npy', '*combined*.npy'],
        'protein_emb': ['*protein*emb*.npy', '*prot*.npy'],
        'ligand_emb': ['*ligand*emb*.npy', '*lig*.npy'],
        'labels': ['*labels*.npy', '*activity*.npy'],
        'train_idx': ['*train*idx*.npy', '*train*indices*.npy'],
        'val_idx': ['*val*idx*.npy', '*val*indices*.npy', '*validation*.npy'],
        'test_idx': ['*test*idx*.npy', '*test*indices*.npy'],
        'cluster_labels': ['*cluster*labels*.npy', '*clusters*.npy'],
    }
    
    for file_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = list(run_dir.rglob(pattern))
            if matches:
                # Usar o arquivo mais recente se houver múltiplas correspondências
                files[file_type] = str(max(matches, key=lambda p: p.stat().st_mtime))
                break
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Visualizar splits da estratificação de forma simplificada',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Opção 1: Diretório de execução
    parser.add_argument('--run-dir', '-r', type=str,
                       help='Diretório contendo resultados da estratificação')
    
    # Opção 2: Arquivos individuais
    parser.add_argument('--embeddings', '-e', type=str,
                       help='Caminho para embeddings concatenados')
    parser.add_argument('--protein-emb', '-p', type=str,
                       help='Caminho para embeddings de proteínas')
    parser.add_argument('--ligand-emb', '-l', type=str,
                       help='Caminho para embeddings de ligantes')
    parser.add_argument('--labels', type=str,
                       help='Caminho para labels')
    parser.add_argument('--train-idx', type=str,
                       help='Caminho para índices train')
    parser.add_argument('--val-idx', type=str,
                       help='Caminho para índices val')
    parser.add_argument('--test-idx', type=str,
                       help='Caminho para índices test')
    parser.add_argument('--cluster-labels', type=str,
                       help='Caminho para labels de clusters')
    
    # Opções de visualização
    parser.add_argument('--method', '-m', type=str, default='pca',
                       choices=['pca', 'tsne', 'umap'],
                       help='Método de redução (default: pca)')
    parser.add_argument('--output', '-o', type=str, default='.',
                       help='Diretório de saída (default: .)')
    parser.add_argument('--prefix', type=str, default='stratification_splits',
                       help='Prefixo para arquivos (default: stratification_splits)')
    parser.add_argument('--no-show', action='store_true',
                       help='Não exibir gráficos')
    parser.add_argument('--save-metrics', action='store_true',
                       help='Salvar métricas em JSON')
    
    # Opções de clustering
    parser.add_argument('--similarity-threshold', '-s', type=float, default=0.7,
                       help='Threshold de similaridade (default: 0.7)')
    parser.add_argument('--auto-threshold', action='store_true',
                       help='Usar threshold automático')
    
    args = parser.parse_args()
    
    # Determinar arquivos a usar
    files = {}
    
    if args.run_dir:
        print(f"🔍 Procurando arquivos em: {args.run_dir}")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"❌ Diretório não encontrado: {run_dir}")
            return 1
        
        files = find_stratification_files(run_dir)
        
        # Mostrar arquivos encontrados
        print("\n📂 Arquivos encontrados:")
        for file_type, path in files.items():
            if path:
                print(f"   ✓ {file_type}: {Path(path).name}")
            else:
                print(f"   ✗ {file_type}: não encontrado")
    else:
        # Usar arquivos especificados manualmente
        files['embeddings'] = args.embeddings
        files['protein_emb'] = args.protein_emb
        files['ligand_emb'] = args.ligand_emb
        files['labels'] = args.labels
        files['train_idx'] = args.train_idx
        files['val_idx'] = args.val_idx
        files['test_idx'] = args.test_idx
        files['cluster_labels'] = args.cluster_labels
    
    # Validar que temos arquivos necessários
    if not files['embeddings'] and not (files['protein_emb'] and files['ligand_emb']):
        print("\n❌ ERRO: Forneça --embeddings ou ambos --protein-emb e --ligand-emb")
        print("   Ou use --run-dir para buscar automaticamente")
        return 1
    
    # Verificar se temos índices de splits
    has_splits = (files['train_idx'] and files['val_idx'] and files['test_idx'])
    
    if not has_splits:
        print("\n⚠️  AVISO: Índices de splits não encontrados")
        print("   Apenas visualização de clusters será gerada")
        print("   Para visualizar splits, forneça --train-idx, --val-idx e --test-idx")
    
    # Construir comando para o script principal
    script_path = Path(__file__).parent / 'visualize_cluster_pca.py'
    cmd = ['python', str(script_path)]
    
    # Adicionar argumentos de entrada
    if files['embeddings']:
        cmd.extend(['--embeddings', files['embeddings']])
    elif files['protein_emb'] and files['ligand_emb']:
        cmd.extend(['--protein-emb', files['protein_emb']])
        cmd.extend(['--ligand-emb', files['ligand_emb']])
    
    if files['labels']:
        cmd.extend(['--labels', files['labels']])
    
    if files['cluster_labels']:
        cmd.extend(['--cluster-labels', files['cluster_labels']])
    
    if has_splits:
        cmd.extend(['--train-idx', files['train_idx']])
        cmd.extend(['--val-idx', files['val_idx']])
        cmd.extend(['--test-idx', files['test_idx']])
    
    # Adicionar opções de visualização
    cmd.extend(['--method', args.method])
    cmd.extend(['--output', args.output])
    cmd.extend(['--prefix', args.prefix])
    cmd.extend(['--similarity-threshold', str(args.similarity_threshold)])
    
    if args.no_show:
        cmd.append('--no-show')
    
    if args.save_metrics:
        cmd.append('--save-metrics')
    
    if args.auto_threshold:
        cmd.append('--auto-threshold')
    
    # Executar comando
    print("\n" + "="*60)
    print("🚀 Executando visualização...")
    print("="*60)
    print(f"\nComando: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
