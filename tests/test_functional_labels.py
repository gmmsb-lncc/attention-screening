#!/usr/bin/env python3
"""
Teste funcional para o pipeline de labels
"""

import os
import sys
import numpy as np
import pandas as pd

# Adicionar o diretório build ao path
sys.path.insert(0, 'src/build')

def test_labels_pipeline():
    """Testa o pipeline completo de geração de labels."""
    print("🧪 Teste funcional do pipeline de labels")
    print("="*50)
    
    try:
        # 1. Criar dados de teste
        print("📝 Criando dados de teste...")
        
        os.makedirs('test_concatenated_embeddings', exist_ok=True)
        
        # Criar TSV sintético com dados realistas
        test_data = {
            'molregno': ['mol001', 'mol002', 'mol003', 'mol004'],
            'seq_id': ['seq001', 'seq002', 'seq001', 'seq003'],
            'target_kinase': ['KINASE_A', 'KINASE_B', 'KINASE_A', 'KINASE_C'],
            'standard_type': ['IC50', 'Ki', 'Kd', 'IC50'],
            'standard_value': [500, 1500, 300, 2000]  # Valores em nM
        }
        df = pd.DataFrame(test_data)
        df.to_csv('nr_kinase_all_compounds.tsv', sep='\t', index=False)
        
        print("✅ TSV de teste criado")
        
        # 2. Testar buildInteractionLabels
        print("\n🔍 Testando buildInteractionLabels...")
        from buildInteractionLabels import InteractionLabels
        
        label_extractor = InteractionLabels('nr_kinase_all_compounds.tsv', 
                                          output_dir='test_concatenated_embeddings')
        label_extractor.extract_labels()
        label_extractor.stop_spark()
        
        # Verificar se interaction_labels.npy foi criado
        interaction_labels_path = 'test_concatenated_embeddings/interaction_labels.npy'
        if os.path.exists(interaction_labels_path):
            labels = np.load(interaction_labels_path, allow_pickle=True)
            print(f"✅ interaction_labels.npy criado: {labels.shape}")
            print(f"   - Primeiras linhas: {labels[:2]}")
        else:
            print("❌ interaction_labels.npy não foi criado")
            return False
        
        # 3. Testar buildbinaryLabels
        print("\n🔍 Testando buildbinaryLabels...")
        from buildbinaryLabels import BinaryLabelGenerator
        
        # Usar o arquivo criado na etapa anterior
        generator = BinaryLabelGenerator('test_concatenated_embeddings/interaction_labels.npy',
                                       output_dir='test_concatenated_embeddings')
        generator.generate_binary_labels()
        
        # Verificar se binary_labels.npy foi criado
        binary_labels_path = 'test_concatenated_embeddings/binary_labels.npy'
        if os.path.exists(binary_labels_path):
            binary_labels = np.load(binary_labels_path)
            print(f"✅ binary_labels.npy criado: {binary_labels.shape}")
            print(f"   - Labels: {binary_labels}")
            
            # Verificar lógica do threshold (≤1000 nM = 1, >1000 nM = 0)
            expected_labels = [1, 0, 1, 0]  # 500≤1000=1, 1500>1000=0, 300≤1000=1, 2000>1000=0
            print(f"   - Esperado: {expected_labels}")
            print(f"   - Obtido: {binary_labels.tolist()}")
            
            if np.array_equal(binary_labels, expected_labels):
                print("✅ Lógica de threshold funcionando corretamente!")
            else:
                print("❌ Problema na lógica de threshold")
                return False
        else:
            print("❌ binary_labels.npy não foi criado")
            return False
        
        # 4. Testar checkConcatenate com dados simulados
        print("\n🔍 Testando checkConcatenate...")
        
        # Criar matriz simulada de embeddings concatenados
        fake_embeddings = np.random.rand(4, 3328)  # 4 amostras, 3328 dimensões
        fake_embeddings_normalized = (fake_embeddings - fake_embeddings.min()) / (fake_embeddings.max() - fake_embeddings.min())
        
        np.save('test_concatenated_embeddings/concatenated_embeddings_normalized.npy', fake_embeddings_normalized)
        
        from checkConcatenate import EmbeddingCheck
        checker = EmbeddingCheck(matrix_dir='test_concatenated_embeddings',
                               original_tsv_path='nr_kinase_all_compounds.tsv')
        
        # Fazer verificação manual
        concatenated_matrix = checker.load_matrix(checker.concatenated_path)
        labels_matrix = checker.load_matrix(checker.labels_path)
        
        if concatenated_matrix is not None and labels_matrix is not None:
            print(f"✅ Matrizes carregadas com sucesso")
            print(f"   - Embeddings shape: {concatenated_matrix.shape}")
            print(f"   - Labels shape: {labels_matrix.shape}")
            
            # Verificar alinhamento
            if concatenated_matrix.shape[0] == labels_matrix.shape[0]:
                print("✅ Alinhamento das matrizes OK")
            else:
                print("❌ Problema no alinhamento das matrizes")
                return False
        else:
            print("❌ Erro ao carregar matrizes")
            return False
        
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
            'test_concatenated_embeddings',
            'nr_kinase_all_compounds.tsv'
        ]
        
        for item in cleanup_items:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
        print("✅ Limpeza concluída")

if __name__ == "__main__":
    success = test_labels_pipeline()
    
    print("\n" + "="*50)
    if success:
        print("🎉 TESTE DO PIPELINE DE LABELS PASSOU!")
    else:
        print("❌ TESTE DO PIPELINE DE LABELS FALHOU!")
    print("="*50)
    
    sys.exit(0 if success else 1)
