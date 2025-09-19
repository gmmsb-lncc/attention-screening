#!/usr/bin/env python3
"""
Script de teste para validar todos os scripts do diretório src/build
"""

import os
import sys
import importlib.util
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

def test_script_import(script_path, script_name):
    """Testa se um script pode ser importado sem erros."""
    print(f"\n🔍 Testando import de {script_name}...")
    try:
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        if spec is None:
            print(f"❌ Erro: Não foi possível criar spec para {script_name}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {script_name} importado com sucesso")
        return True
    except Exception as e:
        print(f"❌ Erro ao importar {script_name}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_class_instantiation(script_path, script_name, class_name, *args, **kwargs):
    """Testa se uma classe pode ser instanciada."""
    print(f"\n🏗️ Testando instanciação da classe {class_name} em {script_name}...")
    try:
        spec = importlib.util.spec_from_file_location(script_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        print(f"✅ Classe {class_name} instanciada com sucesso")
        return True, instance
    except Exception as e:
        print(f"❌ Erro ao instanciar {class_name}: {str(e)}")
        return False, None

def create_test_data():
    """Cria dados de teste simulados."""
    print("\n📝 Criando dados de teste...")
    
    # Criar diretórios de teste
    test_dirs = ['test_ligand_embeddings', 'test_protein_embeddings', 'test_concatenated_embeddings']
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Criar embeddings de teste
    # Embeddings de ligantes (768 dimensões)
    for i in range(3):
        ligand_emb = np.random.rand(2, 768)  # [CLS, sequence]
        np.save(f'test_ligand_embeddings/{i}_ligand.npy', ligand_emb)
    
    # Embeddings de proteínas (2560 dimensões) 
    for i in range(3):
        protein_emb = np.random.rand(50, 2560)  # [sequence_length, hidden_dim]
        np.save(f'test_protein_embeddings/{i}_protein_embedding.npy', protein_emb)
    
    # Criar TSV de teste
    test_data = {
        'molregno': ['0', '1', '2'],
        'seq_id': ['0', '1', '2'],
        'target_kinase': ['KINASE_A', 'KINASE_B', 'KINASE_C'],
        'standard_type': ['IC50', 'Ki', 'Kd'],
        'standard_value': [500, 1500, 300],
        'canonical_smiles': ['CC', 'CCO', 'CCC']
    }
    df = pd.DataFrame(test_data)
    df.to_csv('test_nr_kinase_all_compounds.tsv', sep='\t', index=False)
    
    print("✅ Dados de teste criados com sucesso")

def cleanup_test_data():
    """Remove dados de teste."""
    print("\n🧹 Limpando dados de teste...")
    
    # Remover diretórios de teste
    import shutil
    test_dirs = ['test_ligand_embeddings', 'test_protein_embeddings', 'test_concatenated_embeddings', 
                 'ligand_embeddings', 'protein_embeddings', 'concatenated_embeddings', 'matrix_embedding']
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    # Remover arquivos de teste
    test_files = ['test_nr_kinase_all_compounds.tsv', 'nr_kinase_all_compounds.tsv',
                  'processed_files.log', 'preparation_checkpoint.txt', 'embedding_checkpoint.txt']
    for file_name in test_files:
        if os.path.exists(file_name):
            os.remove(file_name)
    
    print("✅ Dados de teste removidos")

def main():
    """Executa todos os testes dos scripts de build."""
    print("🚀 Iniciando testes dos scripts de build...")
    
    # Definir caminhos
    build_dir = Path('src/build')
    scripts_to_test = [
        ('buildbinaryLabels.py', 'BinaryLabelGenerator'),
        ('buildInteractionLabels.py', 'InteractionLabels'),
        ('buildEmbeddingMatrix.py', 'EmbeddingMatrixReconstructor'),
        ('buildKinaseMatrix.py', 'EmbeddingMatrixReconstructor'),
        ('checkConcatenate.py', 'EmbeddingCheck'),
        ('checkEmbedding.py', 'EmbeddingCheck'),
        ('embeddingIBM.py', 'EmbeddingIBM'),
        ('embeddingMeta.py', 'EmbeddingMeta'),
    ]
    
    results = []
    create_test_data()
    
    try:
        for script_name, class_name in scripts_to_test:
            script_path = build_dir / script_name
            
            # Teste 1: Import
            import_success = test_script_import(str(script_path), script_name)
            
            # Teste 2: Instanciação de classe
            if import_success:
                if script_name == 'buildbinaryLabels.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name, 
                                                        'test_concatenated_embeddings/interaction_labels.npy')
                elif script_name == 'buildInteractionLabels.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name,
                                                        'test_nr_kinase_all_compounds.tsv')
                elif script_name in ['buildEmbeddingMatrix.py', 'buildKinaseMatrix.py']:
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name,
                                                        'test_nr_kinase_all_compounds.tsv')
                elif script_name == 'checkConcatenate.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name,
                                                        original_tsv_path='test_nr_kinase_all_compounds.tsv')
                elif script_name == 'checkEmbedding.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name)
                elif script_name == 'embeddingIBM.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name)
                elif script_name == 'embeddingMeta.py':
                    success, _ = test_class_instantiation(str(script_path), script_name, class_name,
                                                        seq_input_dir='test_protein_embeddings')
                else:
                    success = True
            else:
                success = False
            
            results.append((script_name, import_success, success))
    
    finally:
        cleanup_test_data()
    
    # Relatório final
    print("\n" + "="*60)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("="*60)
    
    all_passed = True
    for script_name, import_ok, class_ok in results:
        status_import = "✅" if import_ok else "❌"
        status_class = "✅" if class_ok else "❌"
        print(f"{script_name:25} | Import: {status_import} | Classe: {status_class}")
        
        if not (import_ok and class_ok):
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print("⚠️ ALGUNS TESTES FALHARAM - Verifique os erros acima")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
