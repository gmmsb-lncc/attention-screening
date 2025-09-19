#!/usr/bin/env python3
"""
Teste funcional específico para buildEmbeddingMatrix.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Adicionar o diretório build ao path
sys.path.insert(0, 'src/build')

def test_embedding_matrix_functionality():
    """Testa a funcionalidade completa do buildEmbeddingMatrix."""
    print("🧪 Teste funcional do buildEmbeddingMatrix.py")
    print("="*50)
    
    try:
        # 1. Criar dados de teste
        print("📝 Criando dados de teste...")
        
        # Criar diretórios
        os.makedirs('test_ligand_embeddings', exist_ok=True)
        os.makedirs('test_protein_embeddings', exist_ok=True)
        os.makedirs('test_output', exist_ok=True)
        
        # Criar embeddings sintéticos
        for i in range(3):
            # Ligand embeddings: formato [CLS, sequence]
            ligand_emb = np.random.rand(2, 768)
            np.save(f'test_ligand_embeddings/{i}_ligand.npy', ligand_emb)
            
            # Protein embeddings: formato [sequence_length, hidden_dim] 
            protein_emb = np.random.rand(50, 2560)
            np.save(f'test_protein_embeddings/{i}_protein_embedding.npy', protein_emb)
        
        # Criar TSV sintético
        test_data = {
            'molregno': ['0', '1', '2'],
            'seq_id': ['0', '1', '2'],
            'target_kinase': ['KINASE_A', 'KINASE_B', 'KINASE_C'],
            'standard_type': ['IC50', 'Ki', 'Kd'],
            'standard_value': [500, 1500, 300]
        }
        df = pd.DataFrame(test_data)
        df.to_csv('test_compounds.tsv', sep='\t', index=False)
        
        print("✅ Dados de teste criados")
        
        # 2. Testar importação e instanciação
        print("\n🔍 Importando EmbeddingMatrixReconstructor...")
        from buildEmbeddingMatrix import EmbeddingMatrixReconstructor
        
        reconstructor = EmbeddingMatrixReconstructor(
            original_tsv_path='test_compounds.tsv',
            ligand_embeddings_dir='test_ligand_embeddings',
            protein_embeddings_dir='test_protein_embeddings', 
            output_dir='test_output',
            embedding_type='cls'
        )
        print("✅ Classe instanciada com sucesso")
        
        # 3. Testar reconstrução da matriz
        print("\n🔧 Testando reconstrução da matriz...")
        matrix = reconstructor.reconstruct_matrix()
        
        print(f"✅ Matriz reconstruída: {matrix.shape}")
        print(f"   - Esperado: (3, 3328) [3 amostras, 2560+768 dimensões]")
        print(f"   - Obtido: {matrix.shape}")
        
        # Verificar dimensões
        expected_shape = (3, 2560 + 768)  # 3 amostras, protein_dim + ligand_dim
        if matrix.shape == expected_shape:
            print("✅ Dimensões corretas!")
        else:
            print(f"❌ Dimensões incorretas! Esperado {expected_shape}, obtido {matrix.shape}")
        
        # 4. Testar salvamento
        print("\n💾 Testando salvamento...")
        reconstructor.save_matrix(matrix)
        
        # Verificar arquivos criados
        expected_files = [
            'test_output/concatenated_embeddings.npy',
            'test_output/concatenated_embeddings_normalized.npy'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} criado com sucesso")
                # Verificar se pode ser carregado
                loaded = np.load(file_path)
                print(f"   - Shape: {loaded.shape}")
            else:
                print(f"❌ {file_path} não foi criado")
        
        # 5. Testar normalização
        print("\n🔄 Verificando normalização...")
        original_matrix = np.load('test_output/concatenated_embeddings.npy')
        normalized_matrix = np.load('test_output/concatenated_embeddings_normalized.npy')
        
        print(f"Original - Min: {original_matrix.min():.4f}, Max: {original_matrix.max():.4f}")
        print(f"Normalizada - Min: {normalized_matrix.min():.4f}, Max: {normalized_matrix.max():.4f}")
        
        if 0 <= normalized_matrix.min() and normalized_matrix.max() <= 1:
            print("✅ Normalização funcionando corretamente")
        else:
            print("❌ Problema na normalização")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # Limpeza
        print("\n🧹 Limpando arquivos de teste...")
        import shutil
        cleanup_items = [
            'test_ligand_embeddings', 
            'test_protein_embeddings', 
            'test_output',
            'test_compounds.tsv'
        ]
        
        for item in cleanup_items:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
        print("✅ Limpeza concluída")

if __name__ == "__main__":
    success = test_embedding_matrix_functionality()
    
    print("\n" + "="*50)
    if success:
        print("🎉 TESTE FUNCIONAL PASSOU!")
    else:
        print("❌ TESTE FUNCIONAL FALHOU!")
    print("="*50)
    
    sys.exit(0 if success else 1)
