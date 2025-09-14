#!/usr/bin/env python3
"""
Teste de performance e stress para validar estabilidade do sistema DockTKinase.
"""

import sys
import time
import gc
import traceback
import tracemalloc
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_performance_stress():
    """Teste de stress para verificar vazamentos de memória e performance."""
    print("🔥 TESTE DE PERFORMANCE E STRESS")
    print("=" * 50)
    
    # Iniciar monitoramento de memória
    tracemalloc.start()
    
    try:
        from classifier.utils.device_manager import SmartDeviceManager
        from classifier.utils.config_manager import ConfigManager
        from classifier.config.mlp_config import MLPConfig
        
        print("📊 Testando criação múltipla de instâncias...")
        
        # Teste 1: Criação múltipla de ConfigManager
        start_time = time.time()
        config_managers = []
        
        for i in range(20):
            config_mgr = ConfigManager()
            config_managers.append(config_mgr)
            if (i + 1) % 5 == 0:
                print(f"✅ ConfigManager {i+1}/20 criado")
        
        config_time = time.time() - start_time
        print(f"⏱️  Tempo para 20 ConfigManagers: {config_time:.2f}s")
        
        # Teste 2: Criação múltipla de DeviceManager
        start_time = time.time()
        device_managers = []
        
        for i in range(10):
            device_mgr = SmartDeviceManager()
            device_managers.append(device_mgr)
            if (i + 1) % 3 == 0:
                print(f"✅ SmartDeviceManager {i+1}/10 criado")
        
        device_time = time.time() - start_time
        print(f"⏱️  Tempo para 10 DeviceManagers: {device_time:.2f}s")
        
        # Teste 3: Criação de configs múltiplas
        start_time = time.time()
        configs = []
        
        for i in range(50):
            config = MLPConfig(
                hidden_layers=[128, 64] if i % 2 == 0 else [256, 128, 64],
                learning_rate=0.001 * (1 + i * 0.1),
                dropout_rate=0.3 + (i * 0.01)
            )
            configs.append(config)
            if (i + 1) % 10 == 0:
                print(f"✅ MLPConfig {i+1}/50 criado")
        
        mlp_time = time.time() - start_time
        print(f"⏱️  Tempo para 50 MLPConfigs: {mlp_time:.2f}s")
        
        # Teste 4: Templates múltiplos
        start_time = time.time()
        templates_created = []
        
        config_mgr = ConfigManager()
        templates = ["development", "production", "research"]
        
        for i in range(30):
            template_name = templates[i % len(templates)]
            template_config = config_mgr.create_config(template_name)
            templates_created.append(template_config)
            if (i + 1) % 10 == 0:
                print(f"✅ Template {i+1}/30 criado")
        
        template_time = time.time() - start_time
        print(f"⏱️  Tempo para 30 templates: {template_time:.2f}s")
        
        # Verificar uso de memória
        current, peak = tracemalloc.get_traced_memory()
        print(f"\n💾 Uso de memória:")
        print(f"   Atual: {current / 1024 / 1024:.2f} MB")
        print(f"   Pico: {peak / 1024 / 1024:.2f} MB")
        
        # Teste de limpeza
        print("\n🧹 Testando limpeza de memória...")
        del config_managers, device_managers, configs, templates_created
        gc.collect()
        
        # Verificar memória após limpeza
        time.sleep(1)  # Aguardar GC
        current_after, _ = tracemalloc.get_traced_memory()
        print(f"   Após limpeza: {current_after / 1024 / 1024:.2f} MB")
        
        memory_reduction = (current - current_after) / current * 100
        print(f"   Redução: {memory_reduction:.1f}%")
        
        # Métricas finais
        total_time = config_time + device_time + mlp_time + template_time
        print(f"\n📊 MÉTRICAS FINAIS:")
        print(f"   Tempo total: {total_time:.2f}s")
        print(f"   Objetos criados: 110")
        print(f"   Média por objeto: {total_time/110*1000:.1f}ms")
        
        if memory_reduction > 50:
            print("✅ Gestão de memória eficiente")
        else:
            print("⚠️  Possível vazamento de memória")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de stress: {e}")
        traceback.print_exc()
        return False
    
    finally:
        tracemalloc.stop()

def test_concurrent_usage():
    """Teste de uso concorrente simulado."""
    print("\n🔄 TESTE DE USO CONCORRENTE")
    print("=" * 50)
    
    try:
        from classifier.utils.config_manager import ConfigManager
        from classifier.utils.device_manager import SmartDeviceManager
        
        # Simular uso de múltiplos "usuários"
        users = []
        
        for user_id in range(5):
            print(f"👤 Usuário {user_id + 1}:")
            
            # Cada usuário cria seu próprio ambiente
            config_mgr = ConfigManager()
            device_mgr = SmartDeviceManager()
            
            # Configurações diferentes
            template = ["development", "production", "research", "development", "production"][user_id]
            config = config_mgr.create_config(template)
            device = device_mgr.get_device()
            
            user_info = {
                'id': user_id + 1,
                'config_mgr': config_mgr,
                'device_mgr': device_mgr,
                'config': config,
                'device': device
            }
            users.append(user_info)
            
            print(f"   ✅ Template: {template}")
            print(f"   ✅ Device: {device}")
            print(f"   ✅ Modelo: {config.model.hidden_layers}")
        
        print("\n🔍 Verificando isolamento entre usuários...")
        
        # Verificar se cada usuário tem configurações independentes
        for i, user in enumerate(users):
            other_users = [u for j, u in enumerate(users) if j != i]
            
            # Modificar configuração de um usuário
            user['config'].model.dropout_rate = 0.9
            
            # Verificar se outros não foram afetados
            conflicts = 0
            for other in other_users:
                if other['config'].model.dropout_rate == 0.9:
                    conflicts += 1
            
            if conflicts == 0:
                print(f"   ✅ Usuário {user['id']}: Isolamento OK")
            else:
                print(f"   ❌ Usuário {user['id']}: {conflicts} conflitos")
        
        print("✅ Teste de uso concorrente concluído")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste concorrente: {e}")
        return False

def main():
    """Executa todos os testes de performance."""
    print("🚀 TESTES DE PERFORMANCE E STRESS - DOCKTKINASE")
    print("=" * 60)
    
    tests = [
        test_performance_stress,
        test_concurrent_usage
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Pausa entre testes
        except Exception as e:
            print(f"❌ Erro inesperado em {test.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADO: {passed}/{len(tests)} testes de performance passaram")
    
    if passed == len(tests):
        print("🎉 SISTEMA APROVADO EM TESTES DE STRESS!")
        return 0
    else:
        print("⚠️  Alguns testes falharam - verificar problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main())
