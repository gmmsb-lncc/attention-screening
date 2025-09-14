#!/usr/bin/env python3
"""
Validação de tipos e verificação estática de código com mypy simulado.
"""

import sys
import inspect
from pathlib import Path
from typing import get_type_hints, Any, Dict, List, Optional

# Adicionar src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def analyze_type_annotations(module_path: str, class_name: str) -> Dict[str, Any]:
    """Analisa anotações de tipo de uma classe."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            try:
                hints = get_type_hints(method)
                methods[name] = {
                    'annotations': hints,
                    'signature': str(inspect.signature(method)),
                    'has_return_type': 'return' in hints
                }
            except Exception as e:
                methods[name] = {'error': str(e)}
        
        return {
            'class': class_name,
            'methods': methods,
            'class_annotations': get_type_hints(cls) if hasattr(cls, '__annotations__') else {}
        }
    except Exception as e:
        return {'error': str(e)}

def validate_typing_consistency():
    """Valida consistência de tipos em todo o sistema."""
    print("🔍 VALIDAÇÃO DE TIPOS E ANOTAÇÕES")
    print("=" * 50)
    
    # Classes principais para verificar
    classes_to_check = [
        ("classifier.config.mlp_config", "MLPConfig"),
        ("classifier.utils.config_manager", "ConfigManager"),
        ("classifier.utils.device_manager", "SmartDeviceManager"),
        ("classifier.utils.device_manager", "DeviceValidator"),
    ]
    
    results = {}
    issues = []
    
    for module_path, class_name in classes_to_check:
        print(f"🔎 Analisando {class_name}...")
        result = analyze_type_annotations(module_path, class_name)
        results[f"{module_path}.{class_name}"] = result
        
        if 'error' in result:
            issues.append(f"{class_name}: {result['error']}")
            continue
        
        # Verificar problemas específicos
        methods = result.get('methods', {})
        for method_name, method_info in methods.items():
            if 'error' in method_info:
                issues.append(f"{class_name}.{method_name}: {method_info['error']}")
                continue
            
            # Verificar se métodos públicos têm tipos de retorno
            if not method_name.startswith('_') and method_name != '__init__':
                if not method_info.get('has_return_type', False):
                    issues.append(f"{class_name}.{method_name}: Missing return type annotation")
    
    # Relatório de tipos
    print(f"\n📊 ANÁLISE DE TIPOS:")
    total_classes = len(classes_to_check)
    successful = len([r for r in results.values() if 'error' not in r])
    
    print(f"   Classes analisadas: {total_classes}")
    print(f"   Análises bem-sucedidas: {successful}")
    print(f"   Problemas encontrados: {len(issues)}")
    
    if issues:
        print("\n⚠️  PROBLEMAS DE TIPOS:")
        for issue in issues[:10]:  # Mostrar apenas os primeiros 10
            print(f"   • {issue}")
        if len(issues) > 10:
            print(f"   ... e mais {len(issues) - 10} problemas")
    else:
        print("✅ Nenhum problema de tipos encontrado")
    
    return len(issues) == 0, results

def check_import_structure():
    """Verifica estrutura de imports e dependências."""
    print("\n🔗 VERIFICAÇÃO DE ESTRUTURA DE IMPORTS")
    print("=" * 50)
    
    try:
        # Testar imports principais
        imports_to_test = [
            "classifier.config.mlp_config",
            "classifier.utils.config_manager", 
            "classifier.utils.device_manager",
            "classifier.core.data_manager",
            "classifier.core.memory_manager",
            "classifier.main"
        ]
        
        import_results = {}
        for import_name in imports_to_test:
            try:
                module = __import__(import_name, fromlist=[''])
                # Verificar se tem as classes/funções esperadas
                members = dir(module)
                public_members = [m for m in members if not m.startswith('_')]
                
                import_results[import_name] = {
                    'success': True,
                    'public_members': len(public_members),
                    'has_classes': len([m for m in public_members if hasattr(getattr(module, m, None), '__module__')]) > 0
                }
                print(f"✅ {import_name}: {len(public_members)} membros públicos")
                
            except Exception as e:
                import_results[import_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {import_name}: {e}")
        
        # Estatísticas
        successful_imports = len([r for r in import_results.values() if r.get('success', False)])
        total_imports = len(imports_to_test)
        
        print(f"\n📊 RESULTADO DE IMPORTS:")
        print(f"   Imports testados: {total_imports}")
        print(f"   Imports bem-sucedidos: {successful_imports}")
        print(f"   Taxa de sucesso: {successful_imports/total_imports*100:.1f}%")
        
        return successful_imports == total_imports, import_results
        
    except Exception as e:
        print(f"❌ Erro na verificação de imports: {e}")
        return False, {}

def validate_dataclass_integrity():
    """Valida integridade de dataclasses."""
    print("\n📋 VALIDAÇÃO DE DATACLASSES")
    print("=" * 50)
    
    try:
        from classifier.config.mlp_config import MLPConfig
        from dataclasses import fields, is_dataclass
        
        # Verificar se MLPConfig é uma dataclass válida
        if not is_dataclass(MLPConfig):
            print("❌ MLPConfig não é uma dataclass válida")
            return False
        
        print("✅ MLPConfig é uma dataclass válida")
        
        # Analisar campos
        config_fields = fields(MLPConfig)
        print(f"📝 Campos encontrados: {len(config_fields)}")
        
        for field in config_fields:
            has_default = field.default != field.default_factory
            field_type = field.type if hasattr(field, 'type') else 'Unknown'
            print(f"   • {field.name}: {field_type} {'(com padrão)' if has_default else '(obrigatório)'}")
        
        # Testar criação de instância
        try:
            config = MLPConfig()
            print("✅ Instância padrão criada com sucesso")
            
            # Testar serialização
            config_dict = config.to_dict()
            print(f"✅ Serialização: {len(config_dict)} campos")
            
            # Testar desserialização
            config_restored = MLPConfig.from_dict(config_dict)
            print("✅ Desserialização bem-sucedida")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao testar MLPConfig: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Erro na validação de dataclasses: {e}")
        return False

def main():
    """Executa validação completa de tipos."""
    print("🔍 VALIDAÇÃO ESTÁTICA DE CÓDIGO - DOCKTKINASE")
    print("=" * 60)
    
    tests = [
        ("Tipos e Anotações", validate_typing_consistency),
        ("Estrutura de Imports", check_import_structure),
        ("Integridade de Dataclasses", validate_dataclass_integrity)
    ]
    
    passed = 0
    results_summary = {}
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            results_summary[test_name] = {'success': success, 'details': details}
            if success:
                passed += 1
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            results_summary[test_name] = {'success': False, 'error': str(e)}
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADO FINAL: {passed}/{len(tests)} validações passaram")
    
    if passed == len(tests):
        print("🎉 CÓDIGO VALIDADO - TIPOS CONSISTENTES!")
        return 0
    else:
        print("⚠️  Algumas validações falharam")
        return 1

if __name__ == "__main__":
    sys.exit(main())
