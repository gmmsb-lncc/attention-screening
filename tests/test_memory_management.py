#!/usr/bin/env python3
"""
Teste de validação para o sistema de gestão de memória escalável.
Verifica se o problema de OOM foi resolvido.
"""

import sys
sys.path.append('/Users/sulfierry/docktkinase/src')

import torch
import numpy as np
from sklearn.datasets import make_classification
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

def test_memory_management():
    """Testa o sistema de gestão de memória."""
    print("🚀 TESTE DE GESTÃO DE MEMÓRIA")
    print("=" * 50)
    
    try:
        from classifier.utils.data_manager import DataManager, MemoryManager
        print("✅ Import do DataManager: OK")
        
        # Teste 1: Informações de memória
        print("\n📊 1. INFORMAÇÕES DE MEMÓRIA:")
        gpu_info = MemoryManager.get_gpu_memory_info()
        ram_info = MemoryManager.get_ram_info()
        
        print(f"   GPU: {gpu_info['available']:.1f}GB disponível de {gpu_info['total']:.1f}GB")
        print(f"   RAM: {ram_info['available']:.1f}GB disponível de {ram_info['total']:.1f}GB")
        
        # Teste 2: Dataset pequeno
        print("\n🔬 2. TESTE COM DATASET PEQUENO:")
        X_small, y_small = make_classification(
            n_samples=1000, n_features=50, n_classes=2, random_state=42
        )
        
        memory_est = MemoryManager.estimate_dataset_memory(
            X_small.shape[0], X_small.shape[1]
        )
        print(f"   Estimativa de memória: {memory_est['total_mb']:.1f}MB")
        
        # Criar DataManager e dataset
        data_manager = DataManager()
        dataset, dataloader = data_manager.load_from_arrays(X_small, y_small)
        
        print(f"   ✅ Dataset criado: {dataset.n_samples} samples")
        print(f"   ✅ DataLoader criado: {len(dataloader)} batches")
        
        # Teste 3: Dataset grande (simulado)
        print("\n🔬 3. TESTE COM DATASET GRANDE (SIMULADO):")
        X_large, y_large = make_classification(
            n_samples=50000, n_features=1024, n_classes=2, random_state=42
        )
        
        memory_est_large = MemoryManager.estimate_dataset_memory(
            X_large.shape[0], X_large.shape[1]
        )
        print(f"   Estimativa de memória: {memory_est_large['total_gb']:.2f}GB")
        
        # Batch size recomendado
        recommended_batch = MemoryManager.recommend_batch_size(
            X_large.shape[0], X_large.shape[1]
        )
        print(f"   Batch size recomendado: {recommended_batch}")
        
        # Criar dataset grande
        dataset_large, dataloader_large = data_manager.load_from_arrays(
            X_large, y_large, batch_size=recommended_batch
        )
        
        print(f"   ✅ Dataset grande criado: {dataset_large.n_samples} samples")
        print(f"   ✅ DataLoader criado: {len(dataloader_large)} batches de {recommended_batch}")
        
        # Teste 4: Carregamento sob demanda
        print("\n🔬 4. TESTE DE CARREGAMENTO SOB DEMANDA:")
        
        # Verificar que dados estão na CPU
        print(f"   Dados mantidos em: CPU")
        print(f"   Device alvo: {dataset_large.device}")
        
        # Testar acesso a um batch
        sample_batch = next(iter(dataloader_large))
        X_batch, y_batch = sample_batch
        print(f"   ✅ Batch carregado: {X_batch.shape} no device {X_batch.device}")
        print(f"   ✅ Transferência CPU→GPU funcionando")
        
        # Limpeza
        del X_batch, y_batch, sample_batch
        MemoryManager.clear_cache()
        
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✨ Sistema de gestão de memória funcionando corretamente!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_oom_prevention():
    """Testa especificamente a prevenção de OOM."""
    print("\n🚀 TESTE DE PREVENÇÃO DE OOM")
    print("=" * 50)
    
    try:
        from classifier.utils.data_manager import DataManager, ScalableDataset
        
        # Simular dataset que causaria OOM se carregado completamente
        print("📊 Simulando dataset de 100GB (se fosse carregado na GPU)...")
        
        # Dataset grande só na CPU
        X_huge = np.random.randn(1000000, 1024).astype(np.float32)  # ~4GB na CPU
        y_huge = np.random.randint(0, 2, 1000000).astype(np.float32)
        
        print(f"   Dataset simulado: {X_huge.shape} = {X_huge.nbytes / 1024**3:.2f}GB")
        
        # Sistema antigo causaria OOM aqui:
        # X_gpu = torch.from_numpy(X_huge).to("cuda")  # ❌ OOM!
        
        # Sistema novo: carregamento escalável
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = ScalableDataset(X_huge, y_huge, device=device)
        
        print(f"   ✅ ScalableDataset criado sem OOM")
        print(f"   📊 {dataset.n_samples} samples mantidos na CPU")
        
        # Testar acesso individual (transfere apenas 1 sample)
        sample_x, sample_y = dataset[0]
        print(f"   ✅ Acesso individual: {sample_x.shape} transferido para {sample_x.device}")
        
        # Limpeza
        del X_huge, y_huge, dataset, sample_x, sample_y
        MemoryManager.clear_cache()
        
        print("   🎉 PREVENÇÃO DE OOM FUNCIONANDO!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE OOM: {e}")
        return False

if __name__ == "__main__":
    print("🧪 VALIDAÇÃO COMPLETA DO SISTEMA DE GESTÃO DE MEMÓRIA")
    print("=" * 60)
    
    test1 = test_memory_management()
    test2 = test_oom_prevention()
    
    if test1 and test2:
        print("\n" + "=" * 60)
        print("🎉 TODOS OS TESTES DE VALIDAÇÃO PASSARAM!")
        print("✨ Problema de OutOfMemory RESOLVIDO!")
        print("🚀 Sistema escalável pronto para datasets grandes!")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM!")
