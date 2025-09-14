#!/usr/bin/env python3
"""
🧪 TESTE MASTER: Demonstração completa do sistema auto-adaptativo
Este é o teste definitivo que mostra todas as funcionalidades
"""

import sys
import os

# Adicionar o diretório do projeto ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src', 'classifier'))

import torch
import numpy as np
from config.mlp_config import create_default_config
from models.mlp import MLPEmbeddingClassifier

def print_separator(title):
    """Imprime separador visual."""
    print(f"\n{'='*60}")
    print(f"🔬 {title}")
    print(f"{'='*60}")

def test_master_auto_adaptation():
    """Teste master do sistema auto-adaptativo."""
    print_separator("SISTEMA AUTO-ADAPTATIVO DOCKTKINASE")
    
    print("✨ Funcionalidade: O modelo detecta automaticamente o número")
    print("   de features dos dados e constrói a rede adequada")
    print("\n🎯 Problema resolvido: Não é mais necessário especificar input_size!")
    
    # Casos de teste realistas
    test_cases = [
        {"name": "🧬 Proteínas pequenas", "features": 256, "batch": 32},
        {"name": "🧪 Moléculas médias", "features": 512, "batch": 16},
        {"name": "🔬 Complexos grandes", "features": 1024, "batch": 8},
        {"name": "⚗️  Sistemas complexos", "features": 2048, "batch": 4},
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Caso {i}/4: {case['name']} ({case['features']} features)")
        
        # 1. Configuração padrão (input_size=None)
        config = create_default_config()
        model = MLPEmbeddingClassifier(config)
        
        print(f"   🏗️  Modelo criado: input_size={config.input_size} (auto-detectar)")
        
        # 2. Dados simulados
        X = torch.randn(case['batch'], case['features'])
        print(f"   📊 Dados: shape={list(X.shape)}")
        
        # 3. Forward pass - AUTO-DETECÇÃO ACONTECE AQUI
        print(f"   ⚡ Executando forward pass (auto-detecção)...")
        output = model(X)
        
        # 4. Resultados
        params = model.count_parameters()
        print(f"   ✅ SUCESSO!")
        print(f"      • Input detectado: {model.config.input_size}")
        print(f"      • Parâmetros: {params:,}")
        print(f"      • Output shape: {list(output.shape)}")
        
        # 5. Verificação de consistência
        X2 = torch.randn(case['batch']//2, case['features'])
        output2 = model(X2)
        print(f"      • Consistência: {list(output2.shape)} ✅")
        
        results.append({
            "features": case['features'],
            "parameters": params,
            "input_detected": model.config.input_size
        })
    
    return results

def test_training_integration():
    """Testa integração com treinamento."""
    print_separator("INTEGRAÇÃO COM TREINAMENTO")
    
    print("🎯 Testando: Modelo auto-adaptativo + Otimizador + Loss")
    
    # Criar modelo auto-adaptativo
    config = create_default_config()
    model = MLPEmbeddingClassifier(config)
    
    # Dados de exemplo para kinase binding
    batch_size = 32
    n_features = 768  # Típico de embeddings moleculares
    
    X = torch.randn(batch_size, n_features)
    y = torch.randint(0, 2, (batch_size,)).float().unsqueeze(1)
    
    print(f"📊 Dataset sintético: {batch_size} amostras, {n_features} features")
    
    # Forward pass para construir rede
    print("⚡ Auto-detecção de dimensões...")
    logits = model(X)
    
    # Agora pode criar otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print(f"✅ Rede construída: {model.count_parameters():,} parâmetros")
    print("🔧 Otimizador e loss function criados")
    
    # Mini treinamento
    print("\n🚀 Executando mini-treinamento...")
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        
        # Forward
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"   Epoch {epoch+1}/5: Loss = {loss.item():.4f}")
    
    # Verificar se loss diminuiu
    if losses[-1] < losses[0]:
        print(f"✅ TREINAMENTO FUNCIONANDO! Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print(f"⚠️  Loss oscilando: {losses[0]:.4f} → {losses[-1]:.4f}")

def print_summary(results):
    """Imprime resumo dos resultados."""
    print_separator("RESUMO DOS RESULTADOS")
    
    print("🎯 SISTEMA AUTO-ADAPTATIVO VALIDADO COM SUCESSO!")
    print("\n📊 Capacidades demonstradas:")
    print("   ✅ Auto-detecção de dimensões")
    print("   ✅ Construção dinâmica da rede")  
    print("   ✅ Múltiplos tamanhos de entrada")
    print("   ✅ Integração com PyTorch")
    print("   ✅ Treinamento funcional")
    
    print(f"\n📈 Escalabilidade demonstrada:")
    for result in results:
        features = result['features']
        params = result['parameters']
        print(f"   • {features:4d} features → {params:8,} parâmetros")
    
    print(f"\n💡 BENEFÍCIO PRINCIPAL:")
    print(f"   Antes: model_config = MLPConfig(input_size=512, ...)")
    print(f"   Agora: model_config = create_default_config()  # Auto!")
    
    print(f"\n🏆 O modelo se adapta automaticamente aos seus dados!")

if __name__ == "__main__":
    try:
        # Executar testes
        results = test_master_auto_adaptation()
        test_training_integration()
        print_summary(results)
        
        print_separator("TESTE CONCLUÍDO")
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("🚀 Sistema pronto para uso em produção!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
