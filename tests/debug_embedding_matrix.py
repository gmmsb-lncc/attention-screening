#!/usr/bin/env python3
"""
Debug: Encontrar o embedding problemático que causa erro de concatenação
"""

import sys
import numpy as np
from pathlib import Path

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd


def debug_embedding_matrix():
    """Debugar matriz de embeddings."""
    
    print('=' * 80)
    print('🔍 DEBUG: Matriz de Embeddings')
    print('=' * 80)
    print()
    
    # Carregar TSV
    tsv_path = 'tests/datasets/kinase_non_human_compounds.tsv'
    df = pd.read_csv(tsv_path, sep='\t')
    df['molregno'] = df['molregno'].astype(str)
    df['seq_id'] = df['seq_id'].astype(str)
    
    print(f'📊 Total de pares: {len(df)}')
    print()
    
    # Diretório de embeddings
    build_dir = Path('results/kinase_non_human_multi_model/build')
    
    # Verificar cada par
    print('🔎 Verificando dimensões...')
    print()
    
    problematic = []
    
    for idx, row in df.iterrows():
        molregno = str(row['molregno'])
        seq_id = str(row['seq_id'])
        
        # Buscar arquivos
        ligand_files = list(build_dir.glob(f'{molregno}*embedding.npy'))
        protein_files = list(build_dir.glob(f'{seq_id}*embedding.npy'))
        
        # Carregar embeddings
        lig_emb = None
        prot_emb = None
        
        if ligand_files:
            lig_emb = np.load(ligand_files[0])
        
        if protein_files:
            prot_emb = np.load(protein_files[0])
        
        # Verificar dimensões
        if lig_emb is not None and prot_emb is not None:
            lig_dim = lig_emb.shape[0] if len(lig_emb.shape) == 1 else lig_emb.shape
            prot_dim = prot_emb.shape[0] if len(prot_emb.shape) == 1 else prot_emb.shape
            total_dim = lig_dim + prot_dim if isinstance(lig_dim, int) and isinstance(prot_dim, int) else 'INVALID'
            
            # Verificar se há problema
            if total_dim != 1088:
                problematic.append({
                    'index': idx,
                    'molregno': molregno,
                    'seq_id': seq_id,
                    'ligand_dim': lig_dim,
                    'protein_dim': prot_dim,
                    'total': total_dim,
                    'ligand_file': ligand_files[0].name if ligand_files else None,
                    'protein_file': protein_files[0].name if protein_files else None
                })
        elif lig_emb is None or prot_emb is None:
            problematic.append({
                'index': idx,
                'molregno': molregno,
                'seq_id': seq_id,
                'ligand_dim': 'MISSING' if lig_emb is None else lig_emb.shape,
                'protein_dim': 'MISSING' if prot_emb is None else prot_emb.shape,
                'total': 'MISSING',
                'ligand_file': ligand_files[0].name if ligand_files else 'NOT FOUND',
                'protein_file': protein_files[0].name if protein_files else 'NOT FOUND'
            })
        
        # Progress
        if (idx + 1) % 1000 == 0:
            print(f'   Processado: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)')
    
    print()
    print('=' * 80)
    print('📋 RESULTADOS')
    print('=' * 80)
    print()
    
    if problematic:
        print(f'❌ Encontrados {len(problematic)} problemas:')
        print()
        
        # Mostrar primeiros 20
        for i, prob in enumerate(problematic[:20], 1):
            print(f'{i}. Index {prob["index"]}:')
            print(f'   molregno: {prob["molregno"]} → {prob["ligand_dim"]} (arquivo: {prob["ligand_file"]})')
            print(f'   seq_id: {prob["seq_id"]} → {prob["protein_dim"]} (arquivo: {prob["protein_file"]})')
            print(f'   Total: {prob["total"]} (esperado: 1088)')
            print()
        
        if len(problematic) > 20:
            print(f'... e mais {len(problematic) - 20} problemas')
    else:
        print('✅ Nenhum problema encontrado!')
        print('    Todas as dimensões estão corretas (320 + 768 = 1088)')
    
    return problematic


if __name__ == '__main__':
    problems = debug_embedding_matrix()
    sys.exit(0 if not problems else 1)
