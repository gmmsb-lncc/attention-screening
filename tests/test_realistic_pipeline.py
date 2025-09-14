#!/usr/bin/env python3
"""
Teste realista do pipeline com auto-adaptação
"""

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src', 'classifier'))

import torch
import numpy as np

def test_realistic_pipeline():
    """Teste realista do pipeline principal com auto-adaptação."""
    print("=== Teste Realista do Pipeline ===")
    
    try:
        from main import MLPPipeline
        print("✅ MLPPipeline importado com sucesso")
        
        # Criar dados sintéticos com diferentes tamanhos
        test_cases = [
            {"n_features": 256, "n_samples": 1000},
            {"n_features": 512, "n_samples": 800},
            {"n_features": 1024, "n_samples": 600},
        ]
        
        for case in test_cases:
            print(f"\n🧪 Testando {case['n_features']} features, {case['n_samples']} amostras")
            
            # Gerar dados sintéticos
            np.random.seed(42)
            X = np.random.randn(case['n_samples'], case['n_features'])
            y = np.random.randint(0, 2, case['n_samples'])  # Labels binários
            
            print(f"   Dados criados: X.shape={X.shape}, y.shape={y.shape}")
            
            # Criar pipeline
            pipeline = MLPPipeline()
            print("   Pipeline criado")
            
            # Carregar dados no pipeline usando o dataset escalável
            print("   Criando dataset...")
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            # Simular ScalableDataset
            from core.data_manager import ScalableDataset
            dataset = ScalableDataset(X, y)  # Usar numpy arrays diretamente
            pipeline.dataset = dataset
            pipeline.X = X_tensor.to(pipeline.device)
            pipeline.y = y_tensor.to(pipeline.device)
            
            # Carregar configuração (cria model_config com auto-detecção)
            pipeline.load_config()
            
            # Executar cross-validation (vai usar auto-detecção)
            print("   Iniciando cross-validation...")
            results = pipeline.run_cross_validation(n_folds=3)  # 3 folds para ser mais rápido
            
            print(f"   ✅ Treinamento concluído!")
            print(f"   - Melhor accuracy: {results.get('best_score', 'N/A'):.4f}")
            print(f"   - Input detectado: {pipeline.model_config.input_size if hasattr(pipeline, 'model_config') else 'N/A'}")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realistic_pipeline()
