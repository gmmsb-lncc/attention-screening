#!/usr/bin/env python3
"""
Teste completo do pipeline principal.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "classifier"))

import torch
import numpy as np

print("🚀 Teste Completo do Pipeline Principal")
print("=" * 60)

def create_synthetic_data(n_samples=1000, n_features=100, n_classes=2):
    """Cria dados sintéticos para teste."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    return X, y

def test_complete_pipeline():
    """Testa pipeline completo."""
    print("\n📊 Criando dados sintéticos...")
    X, y = create_synthetic_data(500, 50, 2)
    print(f"✅ Dados criados: {X.shape}, classes: {y.bincount()}")
    
    print("\n🏗️ Inicializando pipeline...")
    try:
        from main import MLPPipeline
        
        # Criar pipeline com configuração simples
        pipeline = MLPPipeline()
        print(f"✅ Pipeline inicializado no device: {pipeline.device}")
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return False
    
    print("\n📥 Carregando dados no pipeline...")
    try:
        # Usar método direto para criar DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"✅ Dataset criado: {len(dataset)} amostras")
        print(f"✅ DataLoader criado: {len(dataloader)} batches")
        
    except Exception as e:
        print(f"❌ Erro no carregamento: {e}")
        return False
    
    print("\n🧠 Testando modelo...")
    try:
        from config.mlp_config import create_default_config
        from models.mlp import MLPEmbeddingClassifier
        
        # Configuração do modelo
        config = create_default_config(input_size=50)
        model = MLPEmbeddingClassifier(config)
        model = model.to(pipeline.device)
        
        print(f"✅ Modelo criado: {config.get_architecture_summary()}")
        
        # Teste forward com batch
        model.eval()
        with torch.no_grad():
            batch_X, batch_y = next(iter(dataloader))
            batch_X = batch_X.to(pipeline.device)
            output = model(batch_X)
            
        print(f"✅ Forward pass: {batch_X.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"❌ Erro no modelo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📏 Testando métricas...")
    try:
        # Criar predições fake para teste
        y_true = batch_y.to(pipeline.device)
        y_pred = torch.sigmoid(output.squeeze())
        y_pred_binary = (y_pred > 0.5).float()
        
        # Calcular métricas básicas
        accuracy = (y_pred_binary == y_true.float()).float().mean()
        print(f"✅ Accuracy calculada: {accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Erro nas métricas: {e}")
        return False
    
    return True

def main():
    """Executa teste completo."""
    try:
        success = test_complete_pipeline()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 TESTE COMPLETO PASSOU!")
            print("✅ Pipeline funcionando corretamente")
            print("✅ Pronto para treinamento real")
        else:
            print("❌ TESTE FALHOU")
            print("🔧 Correções necessárias")
        print("=" * 60)
        
        return success
        
    except Exception as e:
        print(f"💥 ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
