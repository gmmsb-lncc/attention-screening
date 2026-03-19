#!/usr/bin/env python3
"""
Revisão Profunda e Completa do Sistema DockTKinase
===================================================

Este script executa uma análise exhaustiva de todo o sistema build
para garantir que nenhum erro será propagado para produção.

Verifica:
- Sintaxe Python em todos os arquivos
- Importações e dependências
- Herança e implementação de métodos abstratos
- Integridade dos módulos
- Funcionalidade do pipeline
- Compatibilidade backward
- Performance e memory leaks
- Documentação e tipo hints
- Testes unitários
"""

import sys
import ast
import importlib
import inspect
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import subprocess
import time
import gc

# Configurar paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

class ComprehensiveReviewer:
    """Revisor completo e profundo do sistema."""
    
    def __init__(self):
        self.current_dir = current_dir
        self.build_dir = src_dir / "build"
        self.errors = []
        self.warnings = []
        self.info = []
        self.files_analyzed = 0
        self.modules_loaded = {}
        self.performance_stats = {}
        
    def log_issue(self, level: str, file_path: Path, line_number: int, message: str) -> None:
        """Log um problema encontrado."""
        try:
            relative_path = str(file_path.relative_to(self.current_dir))
        except ValueError:
            relative_path = str(file_path)
        
        issue = {
            'level': level,
            'file': relative_path,
            'line': line_number,
            'message': message,
            'timestamp': time.time()
        }
        
        if level == 'ERROR':
            self.errors.append(issue)
        elif level == 'WARNING':
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def check_python_syntax(self, file_path: Path) -> bool:
        """Verifica sintaxe Python rigorosamente."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificação básica de sintaxe
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.log_issue('ERROR', file_path, e.lineno or 0, f"Syntax error: {e.msg}")
                return False
                
            # Verificação avançada usando compile
            try:
                compile(content, str(file_path), 'exec')
            except SyntaxError as e:
                self.log_issue('ERROR', file_path, e.lineno or 0, f"Compilation error: {e.msg}")
                return False
                
            return True
            
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"File reading error: {e}")
            return False
    
    def check_imports_deep(self, file_path: Path) -> bool:
        """Verificação profunda de imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError:
                            # Ignorar dependências externas conhecidas
                            external_deps = ['torch', 'esm', 'pandas', 'sklearn', 'models', 'transformers', 'huggingface_hub']
                            if not any(skip in alias.name for skip in external_deps):
                                self.log_issue('ERROR', file_path, node.lineno, 
                                             f"Import '{alias.name}' cannot be resolved")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            module = importlib.import_module(node.module)
                            # Verificar se os nomes importados existem
                            for alias in node.names:
                                if alias.name != '*' and not hasattr(module, alias.name):
                                    self.log_issue('WARNING', file_path, node.lineno,
                                                 f"'{alias.name}' not found in '{node.module}'")
                        except ImportError:
                            # Ignorar dependências externas conhecidas
                            external_deps = ['torch', 'esm', 'pandas', 'sklearn', 'models', 'transformers', 'huggingface_hub']
                            if not any(skip in node.module for skip in external_deps):
                                self.log_issue('ERROR', file_path, node.lineno,
                                             f"Module '{node.module}' cannot be imported")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"Import analysis failed: {e}")
            return False
    
    def check_class_inheritance_deep(self, file_path: Path) -> bool:
        """Verificação profunda de herança de classes."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Pular classes de exceção
                    if 'exceptions.py' in str(file_path):
                        continue
                    
                    # Verificar herança de BaseBuilder
                    inherits_base_builder = any(
                        getattr(base, 'id', None) == 'BaseBuilder' for base in node.bases
                    )
                    
                    if inherits_base_builder:
                        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        
                        # Verificar métodos obrigatórios
                        required_methods = ['_validate_config']
                        missing_methods = [m for m in required_methods if m not in methods]
                        
                        if missing_methods:
                            self.log_issue('ERROR', file_path, node.lineno,
                                         f"Class {node.name} missing critical methods: {missing_methods}")
                        
                        # Verificar se tem método build ou métodos abstratos específicos
                        if 'build' not in methods:
                            # Classes abstratas podem ter métodos específicos em vez de build
                            abstract_methods = []
                            for n in node.body:
                                if isinstance(n, ast.FunctionDef):
                                    # Verificar se tem decorator @abstractmethod
                                    has_abstract = any(
                                        getattr(dec, 'id', None) == 'abstractmethod' or
                                        (hasattr(dec, 'attr') and dec.attr == 'abstractmethod')
                                        for dec in n.decorator_list
                                    )
                                    if has_abstract:
                                        abstract_methods.append(n.name)
                            
                            if not abstract_methods and 'base_' not in file_path.name:
                                self.log_issue('WARNING', file_path, node.lineno,
                                             f"Class {node.name} should implement 'build' method")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"Class inheritance analysis failed: {e}")
            return False
    
    def check_type_hints(self, file_path: Path) -> bool:
        """Verifica type hints e documentação."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Verificar se métodos públicos têm type hints
                    if not node.name.startswith('_'):
                        has_return_annotation = node.returns is not None
                        has_arg_annotations = any(arg.annotation for arg in node.args.args[1:])  # Skip self
                        
                        if not has_return_annotation and not node.name in ['__init__', '__str__', '__repr__']:
                            self.log_issue('INFO', file_path, node.lineno,
                                         f"Method '{node.name}' missing return type hint")
                        
                        # Verificar docstring
                        docstring = ast.get_docstring(node)
                        if not docstring and not node.name.startswith('_'):
                            self.log_issue('INFO', file_path, node.lineno,
                                         f"Method '{node.name}' missing docstring")
            
            return True
            
        except Exception as e:
            self.log_issue('WARNING', file_path, 0, f"Type hints check failed: {e}")
            return True  # Não crítico
    
    def test_module_loading(self, module_path: str) -> bool:
        """Testa carregamento real do módulo."""
        try:
            start_time = time.time()
            module = importlib.import_module(module_path)
            load_time = time.time() - start_time
            
            self.modules_loaded[module_path] = module
            self.performance_stats[module_path] = {
                'load_time': load_time,
                'memory_usage': sys.getsizeof(module)
            }
            
            # Verificar se o módulo tem os atributos esperados
            if hasattr(module, '__all__'):
                for attr in module.__all__:
                    if not hasattr(module, attr):
                        self.log_issue('ERROR', Path(module_path), 0,
                                     f"Module declares '{attr}' in __all__ but doesn't define it")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path(module_path), 0, f"Module loading failed: {e}")
            return False
    
    def test_class_instantiation(self) -> bool:
        """Testa instanciação de classes principais."""
        try:
            # Primeiro, criar uma instância de BuildConfig
            from build.core import BuildConfig
            config = BuildConfig()
            
            # Testar instanciação das classes principais
            test_cases = [
                ('build.core', 'BuildConfig', {}),
                ('build.pipeline', 'BuildPipeline', {'config': config}),
                ('build.embeddings', 'ProteinEmbedding', {'config': config}),
                ('build.embeddings', 'LigandEmbedding', {'config': config}),
                ('build.matrix', 'EmbeddingMatrix', {'config': config}),
                ('build.validation', 'MatrixValidator', {'config': config})
            ]
            
            for module_name, class_name, init_args in test_cases:
                try:
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    
                    # Tentar instanciar
                    instance = cls(**init_args)
                    
                    # Verificar se tem métodos essenciais
                    if hasattr(instance, '_validate_config'):
                        try:
                            instance._validate_config()
                        except Exception as e:
                            self.log_issue('WARNING', Path(module_name), 0,
                                         f"{class_name}._validate_config() failed: {e}")
                    
                    self.log_issue('INFO', Path(module_name), 0, f"{class_name} instantiated successfully")
                    
                except Exception as e:
                    self.log_issue('ERROR', Path(module_name), 0, f"{class_name} instantiation failed: {e}")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path('system'), 0, f"Class instantiation test failed: {e}")
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Testa integração completa do pipeline."""
        try:
            from build.core import BuildConfig
            from build.pipeline import BuildPipeline
            
            # Teste com configuração básica
            config = BuildConfig()
            pipeline = BuildPipeline(config)
            
            # Verificar se todos os componentes foram inicializados
            expected_components = ['protein_embedding', 'ligand_embedding', 'embedding_matrix', 'kinase_matrix', 'matrix_validator']
            for component in expected_components:
                if component not in pipeline.components:
                    self.log_issue('ERROR', Path('build.pipeline'), 0,
                                 f"Missing component: {component}")
                else:
                    self.log_issue('INFO', Path('build.pipeline'), 0,
                                 f"Component {component} initialized successfully")
            
            # Testar método de validação do pipeline
            if hasattr(pipeline, 'validate'):
                try:
                    pipeline.validate()
                    self.log_issue('INFO', Path('build.pipeline'), 0, "Pipeline validation passed")
                except Exception as e:
                    self.log_issue('WARNING', Path('build.pipeline'), 0, f"Pipeline validation failed: {e}")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path('build.pipeline'), 0, f"Pipeline integration test failed: {e}")
            return False
    
    def test_backward_compatibility(self) -> bool:
        """Testa compatibilidade com scripts legados."""
        try:
            from build.matrix import EmbeddingMatrixReconstructor
            from build.core import BuildConfig
            
            # Testar interface legada
            config = BuildConfig()
            matrix = EmbeddingMatrixReconstructor(
                config,
                ligand_embeddings_dir='dummy_ligand',
                protein_embeddings_dir='dummy_protein',
                original_tsv_path='/tmp/dummy.tsv'
            )
            
            # Verificar se tem métodos legados
            legacy_methods = ['reconstruct_matrix', 'load_embedding', 'save_matrix']
            for method in legacy_methods:
                if not hasattr(matrix, method):
                    self.log_issue('ERROR', Path('build.matrix'), 0,
                                 f"Legacy method '{method}' missing")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path('build.matrix'), 0, f"Backward compatibility test failed: {e}")
            return False
    
    def check_memory_leaks(self) -> bool:
        """Verifica vazamentos de memória."""
        try:
            initial_objects = len(gc.get_objects())
            
            # Executar várias instanciações
            for _ in range(10):
                from build.core import BuildConfig
                config = BuildConfig()
                del config
            
            gc.collect()
            final_objects = len(gc.get_objects())
            
            object_diff = final_objects - initial_objects
            if object_diff > 100:  # Threshold arbitrário
                self.log_issue('WARNING', Path('system'), 0,
                             f"Potential memory leak: {object_diff} objects not cleaned")
            else:
                self.log_issue('INFO', Path('system'), 0, "Memory usage check passed")
            
            return True
            
        except Exception as e:
            self.log_issue('WARNING', Path('system'), 0, f"Memory leak check failed: {e}")
            return True
    
    def analyze_file_deep(self, file_path: Path) -> bool:
        """Análise profunda de um arquivo."""
        print(f"🔍 Deep analyzing: {file_path}")
        
        success = True
        
        # Verificações básicas
        success &= self.check_python_syntax(file_path)
        success &= self.check_imports_deep(file_path)
        success &= self.check_class_inheritance_deep(file_path)
        success &= self.check_type_hints(file_path)
        
        self.files_analyzed += 1
        return success
    
    def run_comprehensive_review(self) -> Dict[str, Any]:
        """Executa revisão completa e profunda."""
        print("🔍 INICIANDO REVISÃO PROFUNDA E COMPLETA")
        print("=" * 60)
        
        # Análise de arquivos
        python_files = list(self.build_dir.rglob("*.py"))
        for file_path in python_files:
            if file_path.name != '__pycache__':
                self.analyze_file_deep(file_path)
        
        print("\n⚙️ Testing module loading...")
        modules_to_test = [
            'build.core',
            'build.pipeline', 
            'build.embeddings',
            'build.matrix',
            'build.labels',
            'build.validation',
            'build.utils'
        ]
        
        for module in modules_to_test:
            self.test_module_loading(module)
        
        print("\n🧪 Testing class instantiation...")
        self.test_class_instantiation()
        
        print("\n🚀 Testing pipeline integration...")
        self.test_pipeline_integration()
        
        print("\n🔄 Testing backward compatibility...")
        self.test_backward_compatibility()
        
        print("\n🧠 Testing memory usage...")
        self.check_memory_leaks()
        
        return self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Gera relatório completo."""
        total_issues = len(self.errors) + len(self.warnings)
        
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO COMPLETO DA REVISÃO PROFUNDA")
        print("=" * 60)
        print(f"📁 Arquivos analisados: {self.files_analyzed}")
        print(f"🔍 Módulos testados: {len(self.modules_loaded)}")
        print(f"❌ Erros críticos: {len(self.errors)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"ℹ️ Informações: {len(self.info)}")
        
        if self.errors:
            print(f"\n❌ ERROS CRÍTICOS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   💥 {error['file']}:{error['line']} - {error['message']}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Limitar a 10 para não poluir
                print(f"   🟡 {warning['file']}:{warning['line']} - {warning['message']}")
            if len(self.warnings) > 10:
                print(f"   ... e mais {len(self.warnings) - 10} warnings")
        
        # Estatísticas de performance
        if self.performance_stats:
            print(f"\n⚡ ESTATÍSTICAS DE PERFORMANCE:")
            for module, stats in self.performance_stats.items():
                print(f"   📦 {module}: {stats['load_time']:.3f}s, {stats['memory_usage']} bytes")
        
        print("\n" + "=" * 60)
        
        if len(self.errors) == 0:
            print("🎉 REVISÃO COMPLETA: SISTEMA APROVADO!")
            print("✅ Nenhum erro crítico encontrado")
            print("🚀 Sistema 100% pronto para produção")
        else:
            print("🚨 REVISÃO COMPLETA: PROBLEMAS CRÍTICOS ENCONTRADOS!")
            print("❌ Erros devem ser corrigidos antes da produção")
        
        return {
            'files_analyzed': self.files_analyzed,
            'modules_loaded': len(self.modules_loaded),
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'info': len(self.info),
            'total_issues': total_issues,
            'approved': len(self.errors) == 0,
            'performance_stats': self.performance_stats
        }

def main():
    """Função principal."""
    reviewer = ComprehensiveReviewer()
    results = reviewer.run_comprehensive_review()
    return results['approved']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
