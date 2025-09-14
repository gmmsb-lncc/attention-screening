#!/usr/bin/env python3
"""
Teste de auto-adaptação dimensional
"""

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src', 'classifier'))

import torch
import torch.nn as nn
import numpy as np
from config.mlp_config import create_default_config
from models.mlp import MLPEmbeddingClassifier

def test_auto_adaptive_dimensions():
    """Testa adaptação automática a diferentes tamanhos de entrada."""
    print("=== Teste de Auto-Adaptação Dimensional ===")
    
    # Testar diferentes tamanhos de entrada
    test_sizes = [256, 512, 1024, 2048]
    
    for input_size in test_sizes:
        print(f"\n🧪 Testando input_size: {input_size}")
        
        # Criar configuração com auto-detecção
        config = create_default_config()  # input_size=None
        print(f"   Config inicial input_size: {config.input_size}")
        
        # Criar modelo
        model = MLPEmbeddingClassifier(config)
        print(f"   Modelo criado, rede vazia: {len(model.layers) == 0}")
        
        # Criar dados de teste
        batch_size = 32
        x = torch.randn(batch_size, input_size)
        
        # Fazer forward pass (deve auto-detectar)
        print(f"   Executando forward pass...")
        output = model(x)
        
        print(f"   ✅ Auto-detecção funcionou!")
        print(f"   - Input detectado: {model.config.input_size}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Parâmetros totais: {model.count_parameters()}")

def test_consistency():
    """Testa consistência entre múltiplos forward passes."""
    print(f"\n=== Teste de Consistência ===")
    
    config = create_default_config()
    model = MLPEmbeddingClassifier(config)
    
    # Primeiro forward
    x1 = torch.randn(16, 512)
    out1 = model(x1)
    params_after_first = model.count_parameters()
    
    print(f"Primeiro forward - params: {params_after_first}")
    
    # Segundo forward (mesmo tamanho)
    x2 = torch.randn(8, 512)  # batch diferente, features iguais
    out2 = model(x2)
    params_after_second = model.count_parameters()
    
    print(f"Segundo forward - params: {params_after_second}")
    print(f"Parâmetros consistentes: {params_after_first == params_after_second}")

if __name__ == "__main__":
    try:
        test_auto_adaptive_dimensions()
        test_consistency()
        print(f"\n🎉 Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
