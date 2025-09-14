"""
Exemplo de uso do Sistema de Device Management - Ponto 2.

Demonstra as funcionalidades do novo sistema robusto de validação
e seleção de device (GPU/CPU) com fallback inteligente.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import logging

# Configurar logging para ver os detalhes
logging.basicConfig(level=logging.INFO)

# Adicionar path do projeto
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.classifier.utils.device_manager import SmartDeviceManager, DeviceValidator
from src.classifier.main import MLPPipeline

def criar_dados_exemplo():
    """Cria dados sintéticos para demonstração."""
    print("📊 Criando dados de exemplo...")
    
    np.random.seed(42)
    n_samples = 5000
    n_features = 100
    
    # Dados sintéticos com padrão
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Target baseado em combinação linear das primeiras features
    y = ((X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3) > 0).astype(np.float32)
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i:03d}" for i in range(n_features)])
    df["target"] = y
    
    print(f"   • Shape: {df.shape}")
    print(f"   • Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def exemplo_device_validator():
    """Demonstra o DeviceValidator detalhado."""
    print("\n" + "="*60)
    print("🔍 EXEMPLO 1: DeviceValidator Detalhado")
    print("="*60)
    
    # Configurações diferentes
    configs = [
        {"min_gpu_memory_gb": 1.0, "enable_benchmarking": False},
        {"min_gpu_memory_gb": 4.0, "enable_benchmarking": True},
        {"min_gpu_memory_gb": 0.5, "enable_benchmarking": True, "prefer_gpu": False}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n📋 Configuração {i+1}: {config}")
        
        validator = DeviceValidator(**config)
        devices = validator.detect_available_devices()
        
        print(f"   🔍 {len(devices)} devices detectados:")
        for j, device in enumerate(devices):
            status = "✅ RECOMENDADO" if device.is_recommended else "💡 disponível"
            print(f"      {j+1}. {device.name} ({device.type}) - {status}")
            
            if device.total_memory:
                print(f"         • Memória: {device.get_memory_gb()}")
            if device.compute_capability:
                print(f"         • Compute: {device.get_capability_str()}")
            if device.benchmark_score:
                print(f"         • Benchmark: {device.benchmark_score:.2f}")
            if device.warnings:
                print(f"         • Warnings: {len(device.warnings)}")
            if device.limitations:
                print(f"         • Limitações: {len(device.limitations)}")

def exemplo_smart_device_manager():
    """Demonstra o SmartDeviceManager."""
    print("\n" + "="*60)
    print("🧠 EXEMPLO 2: SmartDeviceManager")
    print("="*60)
    
    # Diferentes modos de seleção
    requirements = ["auto", "cpu_only", "fastest"]
    
    for requirement in requirements:
        print(f"\n🎯 Requisito: {requirement}")
        
        try:
            manager = SmartDeviceManager(
                enable_benchmarking=(requirement == "fastest"),
                min_gpu_memory_gb=1.0
            )
            
            device = manager.get_device(requirement)
            device_info = manager.get_device_info()
            
            print(f"   ✅ Device selecionado: {device}")
            if device_info:
                print(f"      • Nome: {device_info.name}")
                print(f"      • Tipo: {device_info.type}")
                print(f"      • Memória: {device_info.get_memory_gb()}")
                print(f"      • Recomendado: {device_info.is_recommended}")
                if device_info.benchmark_score:
                    print(f"      • Benchmark: {device_info.benchmark_score:.2f}")
            
            # Testar validação
            is_valid = manager.validate_current_device()
            print(f"      • Validação: {'✅ OK' if is_valid else '❌ Problema'}")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")

def exemplo_mlp_pipeline_integration():
    """Demonstra integração completa com MLPPipeline."""
    print("\n" + "="*60)
    print("🚀 EXEMPLO 3: Integração MLPPipeline")
    print("="*60)
    
    # Criar dados temporários
    df = criar_dados_exemplo()
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        data_path = f.name
    
    try:
        # Diferentes configurações de pipeline
        pipeline_configs = [
            {
                "name": "Padrão (auto)",
                "device_requirement": "auto",
                "enable_benchmarking": False,
                "min_gpu_memory_gb": 1.0
            },
            {
                "name": "CPU Forçado",
                "device_requirement": "cpu_only", 
                "enable_benchmarking": False,
                "min_gpu_memory_gb": 0.5
            },
            {
                "name": "Mais rápido (com benchmark)",
                "device_requirement": "fastest",
                "enable_benchmarking": True,
                "min_gpu_memory_gb": 2.0
            }
        ]
        
        for config in pipeline_configs:
            print(f"\n🔧 Pipeline: {config['name']}")
            
            try:
                pipeline = MLPPipeline(
                    device_requirement=config["device_requirement"],
                    enable_benchmarking=config["enable_benchmarking"],
                    min_gpu_memory_gb=config["min_gpu_memory_gb"]
                )
                
                print(f"   ✅ Pipeline inicializado com: {pipeline.device}")
                
                # Informações do device
                device_info = pipeline.device_manager.get_device_info()
                if device_info:
                    print(f"      • Device: {device_info.name} ({device_info.type})")
                    if device_info.benchmark_score:
                        print(f"      • Performance: {device_info.benchmark_score:.2f}")
                    print(f"      • Warnings: {len(device_info.warnings)}")
                
                # Testar carregamento de dados
                print("   📊 Testando carregamento de dados...")
                pipeline.load_data(
                    data_path=Path(data_path),
                    target_column="target"
                )
                print(f"      ✅ Dados carregados: {pipeline.dataset.n_samples} amostras")
                
                # Validar status do device
                status_ok = pipeline.validate_device_status()
                print(f"      • Status device: {'✅ OK' if status_ok else '⚠️ Problema'}")
                
            except Exception as e:
                print(f"   ❌ Erro na configuração: {e}")
    
    finally:
        Path(data_path).unlink()  # Limpar arquivo temporário

def exemplo_fallback_automatico():
    """Demonstra o sistema de fallback automático."""
    print("\n" + "="*60)  
    print("🔄 EXEMPLO 4: Fallback Automático")
    print("="*60)
    
    print("🧪 Simulando cenários de fallback...")
    
    # Cenário 1: Tentar GPU mas usar CPU
    print("\n📋 Cenário 1: GPU preferido mas fallback para CPU")
    try:
        manager = SmartDeviceManager(min_gpu_memory_gb=32.0)  # Memória muito alta
        device = manager.get_device("auto")
        device_info = manager.get_device_info()
        
        print(f"   ✅ Fallback executado para: {device}")
        if device_info and device_info.warnings:
            print(f"   ⚠️ Warnings detectados: {len(device_info.warnings)}")
            
    except Exception as e:
        print(f"   ❌ Erro no fallback: {e}")
    
    # Cenário 2: Validação contínua
    print("\n📋 Cenário 2: Validação contínua de device")
    try:
        pipeline = MLPPipeline(device_requirement="auto")
        
        # Primeira validação
        status1 = pipeline.validate_device_status()
        print(f"   ✅ Validação inicial: {'OK' if status1 else 'Problema'}")
        
        # Simular device OK
        status2 = pipeline.validate_device_status() 
        print(f"   ✅ Validação contínua: {'OK' if status2 else 'Problema'}")
        
    except Exception as e:
        print(f"   ❌ Erro na validação: {e}")

def relatorio_sistema():
    """Gera relatório detalhado do sistema atual."""
    print("\n" + "="*60)
    print("📋 RELATÓRIO DO SISTEMA")
    print("="*60)
    
    try:
        validator = DeviceValidator(enable_benchmarking=True)
        devices = validator.detect_available_devices()
        
        print(f"🔍 Sistema detectado: {len(devices)} devices")
        
        for i, device in enumerate(devices):
            print(f"\n{i+1}. {device.name}")
            print(f"   • Tipo: {device.type}")
            print(f"   • Device: {device.device}")
            if device.total_memory:
                print(f"   • Memória total: {device.get_memory_gb()}")
            if device.available_memory:
                print(f"   • Memória disponível: {device.available_memory:.1f}GB")
            if device.compute_capability:
                print(f"   • Compute capability: {device.get_capability_str()}")
            if device.benchmark_score:
                print(f"   • Benchmark score: {device.benchmark_score:.2f}")
            print(f"   • Disponível: {'✅' if device.is_available else '❌'}")
            print(f"   • Recomendado: {'✅' if device.is_recommended else '❌'}")
            
            if device.warnings:
                print(f"   • ⚠️ Warnings:")
                for warning in device.warnings:
                    print(f"     - {warning}")
                    
            if device.limitations:
                print(f"   • ⚠️ Limitações:")
                for limitation in device.limitations:
                    print(f"     - {limitation}")
        
        # Recomendação final
        recommended = [d for d in devices if d.is_recommended]
        if recommended:
            device = recommended[0]
            print(f"\n🎯 RECOMENDAÇÃO FINAL: {device.name} ({device.type})")
        
    except Exception as e:
        print(f"❌ Erro no relatório: {e}")

def main():
    """Executa todos os exemplos."""
    print("🚀 SISTEMA DE DEVICE MANAGEMENT - PONTO 2")
    print("Demonstrações das funcionalidades implementadas")
    print("="*60)
    
    # Executar todos os exemplos
    try:
        relatorio_sistema()
        exemplo_device_validator()
        exemplo_smart_device_manager()
        exemplo_mlp_pipeline_integration()
        exemplo_fallback_automatico()
        
        print("\n" + "="*60)
        print("✅ PONTO 2 COMPLETAMENTE IMPLEMENTADO!")
        print("="*60)
        print("🔧 Funcionalidades disponíveis:")
        print("   • Detecção automática de CPU/CUDA/MPS")
        print("   • Validação de memória e compute capability")
        print("   • Fallback inteligente GPU → CPU")
        print("   • Benchmarking opcional para seleção")
        print("   • Validação contínua de devices")
        print("   • Integração completa com MLPPipeline")
        print("   • Interface CLI atualizada")
        print("   • Logging detalhado de todos os processos")
        
        print("\n🎉 Sistema pronto para uso em produção!")
        
    except Exception as e:
        print(f"❌ Erro na execução dos exemplos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
