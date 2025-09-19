#!/usr/bin/env python3
"""
Script de Revisão Completa - Análise de Inconsistências e Erros

Este script realiza uma análise sistemática de todos os módulos do sistema build,
procurando por erros de sintaxe, inconsistências de imports, problemas de configuração
e outras questões que podem afetar o funcionamento.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback

# Adicionar src ao path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

class CodeReviewer:
    """Analisador de código para detectar problemas e inconsistências."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.files_analyzed = 0
        self.current_dir = current_dir  # Adicionar referência ao diretório atual
        self.build_dir = src_dir / "build"
    
    def log_issue(self, level: str, file_path: Path, line_number: int, message: str) -> None:
        """Log um problema encontrado."""
        try:
            relative_path = str(file_path.relative_to(self.current_dir))
        except ValueError:
            # Se não conseguir resolver o path relativo, usar o path absoluto
            relative_path = str(file_path)
        
        issue = {
            'level': level,
            'file': relative_path,
            'line': line_number,
            'message': message
        }
        
        if level == 'ERROR':
            self.errors.append(issue)
        else:
            self.warnings.append(issue)
    
    def log_warning(self, file_path: str, message: str):
        """Log de warnings."""
        self.warnings.append({
            'file': str(file_path.relative_to(current_dir)),
            'message': message
        })
    
    def check_syntax(self, file_path: Path) -> bool:
        """Verifica sintaxe Python."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True
            
        except SyntaxError as e:
            self.log_issue('ERROR', file_path, e.lineno or 0, f"Syntax error: {e.msg}")
            return False
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"File read error: {e}")
            return False
    
    def check_imports(self, file_path: Path) -> bool:
        """Verifica imports e dependências."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Verificar imports circulares
            if any('build' in imp for imp in imports):
                relative_path = file_path.relative_to(self.build_dir)
                current_module = str(relative_path).replace('/', '.').replace('.py', '')
                
                for imp in imports:
                    if 'build' in imp and current_module in imp:
                        self.log_issue('WARNING', file_path, 0, f"Possible circular import: {imp}")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"Import analysis failed: {e}")
            return False
    
    def check_module_loading(self, file_path: Path) -> bool:
        """Tenta carregar o módulo para verificar problemas de runtime."""
        try:
            # Calcular nome do módulo
            relative_path = file_path.relative_to(src_dir)
            module_name = str(relative_path).replace('/', '.').replace('.py', '')
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                self.log_issue('ERROR', file_path, 0, "Could not create module spec")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return True
            
        except ImportError as e:
            self.log_issue('WARNING', file_path, 0, f"Import error: {e}")
            return False
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"Module loading failed: {e}")
            return False
    
    def check_class_consistency(self, file_path: Path) -> bool:
        """Verifica consistência de classes."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Ignorar classes de exceção
                    if 'exceptions.py' in str(file_path):
                        continue
                        
                    # Verificar se herda de BaseBuilder
                    inherits_base_builder = any(
                        base.id == 'BaseBuilder' for base in node.bases 
                        if isinstance(base, ast.Name)
                    )
                    
                    if inherits_base_builder:
                        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        required_methods = ['build', '_validate_config']
                        missing = [m for m in required_methods if m not in methods]
                        
                        if missing:
                            self.log_issue('WARNING', file_path, node.lineno, 
                                         f"Class {node.name} missing methods: {missing}")
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', file_path, 0, f"Class analysis failed: {e}")
            return False
    
    def check_configuration_consistency(self) -> bool:
        """Verifica consistência de configurações."""
        try:
            from build.core.constants import DEFAULT_LIGAND_DIM, DEFAULT_PROTEIN_DIM, ESM_MODELS, FM4M_MODELS
            from build.core import BuildConfig
            
            # Verificar valores padrão
            config = BuildConfig()
            
            issues_found = []
            
            if config.ligand_dim != DEFAULT_LIGAND_DIM:
                issues_found.append(f"Config ligand_dim ({config.ligand_dim}) != DEFAULT_LIGAND_DIM ({DEFAULT_LIGAND_DIM})")
            
            if config.protein_dim != DEFAULT_PROTEIN_DIM:
                issues_found.append(f"Config protein_dim ({config.protein_dim}) != DEFAULT_PROTEIN_DIM ({DEFAULT_PROTEIN_DIM})")
            
            # Verificar modelos
            if len(ESM_MODELS) == 0:
                issues_found.append("No ESM models defined")
            
            if len(FM4M_MODELS) == 0:
                issues_found.append("No FM4M models defined")
            
            # Verificar se modelos padrão existem
            default_esm = config.get('esm_model')
            if default_esm and default_esm not in ESM_MODELS:
                issues_found.append(f"Default ESM model '{default_esm}' not in ESM_MODELS")
            
            default_fm4m = config.get('fm4m_model')
            if default_fm4m and default_fm4m not in FM4M_MODELS:
                issues_found.append(f"Default FM4M model '{default_fm4m}' not in FM4M_MODELS")
            
            for issue in issues_found:
                self.log_issue('ERROR', Path('src/build/core/config.py'), 0, issue)
            
            return len(issues_found) == 0
            
        except Exception as e:
            self.log_issue('ERROR', Path('src/build/core/config.py'), 0, f"Configuration check failed: {e}")
            return False
    
    def check_backward_compatibility(self) -> bool:
        """Verifica compatibilidade com scripts legados."""
        try:
            from build.matrix import EmbeddingMatrixReconstructor
            
            # Testar inicialização
            matrix = EmbeddingMatrixReconstructor('/dev/null')
            
            # Verificar atributos essenciais
            required_attrs = ['ligand_dir', 'protein_dir', 'ligand_dim', 'protein_dim', 'embedding_type']
            missing_attrs = [attr for attr in required_attrs if not hasattr(matrix, attr)]
            
            if missing_attrs:
                self.log_issue('ERROR', Path('src/build/matrix/embedding_matrix.py'), 0, 
                              f"EmbeddingMatrixReconstructor missing attributes: {missing_attrs}")
                return False
            
            # Verificar método reconstruct_matrix
            if not hasattr(matrix, 'reconstruct_matrix'):
                self.log_issue('ERROR', Path('src/build/matrix/embedding_matrix.py'), 0, 
                              "EmbeddingMatrixReconstructor missing reconstruct_matrix method")
                return False
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path('src/build/matrix/'), 0, f"Backward compatibility check failed: {e}")
            return False
    
    def check_pipeline_integration(self) -> bool:
        """Verifica integração do pipeline."""
        try:
            from build.pipeline import BuildPipeline
            from build.core import BuildConfig
            
            config = BuildConfig()
            pipeline = BuildPipeline(config)
            
            # Verificar componentes esperados
            expected_components = [
                'protein_embedding',
                'ligand_embedding',
                'embedding_matrix',
                'kinase_matrix',
                'matrix_validator'
            ]
            
            missing_components = [comp for comp in expected_components if comp not in pipeline.components]
            
            if missing_components:
                self.log_issue('ERROR', Path('src/build/pipeline/build_pipeline.py'), 0,
                              f"Pipeline missing components: {missing_components}")
                return False
            
            # Verificar se componentes têm método build
            for comp_name, component in pipeline.components.items():
                if not hasattr(component, 'build'):
                    self.log_issue('ERROR', Path('src/build/pipeline/build_pipeline.py'), 0,
                                  f"Component {comp_name} missing build method")
                    return False
            
            return True
            
        except Exception as e:
            self.log_issue('ERROR', Path('src/build/pipeline/'), 0, f"Pipeline integration check failed: {e}")
            return False
    
    def analyze_file(self, file_path: Path) -> bool:
        """Analisa um arquivo específico."""
        print(f"📁 Analyzing: {file_path.relative_to(current_dir)}")
        
        self.files_analyzed += 1
        success = True
        
        # 1. Verificar sintaxe
        if not self.check_syntax(file_path):
            success = False
        
        # 2. Verificar imports
        if not self.check_imports(file_path):
            success = False
        
        # 3. Verificar estrutura de classes
        if not self.check_class_consistency(file_path):
            success = False
        
        # 4. Tentar carregar módulo (apenas se sintaxe OK)
        if success and not file_path.name.startswith('__'):
            if not self.check_module_loading(file_path):
                success = False
        
        return success
    
    def run_review(self) -> Dict[str, Any]:
        """Executa revisão completa."""
        print("🔍 INICIANDO REVISÃO COMPLETA DO CÓDIGO")
        print("=" * 50)
        
        # 1. Analisar todos os arquivos Python
        python_files = list(self.build_dir.rglob("*.py"))
        
        for file_path in python_files:
            self.analyze_file(file_path)
        
        # 2. Verificar consistência de configuração
        print("\n⚙️ Checking configuration consistency...")
        config_ok = self.check_configuration_consistency()
        
        # 3. Verificar compatibilidade
        print("🔄 Checking backward compatibility...")
        compat_ok = self.check_backward_compatibility()
        
        # 4. Verificar integração do pipeline
        print("🚀 Checking pipeline integration...")
        pipeline_ok = self.check_pipeline_integration()
        
        # Compilar resultados
        results = {
            'files_analyzed': self.files_analyzed,
            'total_issues': len(self.issues),
            'total_warnings': len(self.warnings),
            'issues': self.issues,
            'warnings': self.warnings,
            'config_ok': config_ok,
            'compatibility_ok': compat_ok,
            'pipeline_ok': pipeline_ok
        }
        
        return results

def main():
    """Executa revisão completa."""
    reviewer = CodeReviewer()
    results = reviewer.run_review()
    
    # Relatório de resultados
    print("\n" + "=" * 50)
    print("📊 RESULTADO DA REVISÃO")
    print("=" * 50)
    
    print(f"📁 Arquivos analisados: {results['files_analyzed']}")
    print(f"🔍 Issues encontrados: {results['total_issues']}")
    print(f"⚠️ Warnings: {results['total_warnings']}")
    
    # Issues críticos
    if results['issues']:
        print(f"\n❌ ISSUES ENCONTRADOS:")
        for issue in results['issues']:
            severity_icon = "🔴" if issue['severity'] == 'ERROR' else "🟡"
            print(f"   {severity_icon} {issue['file']}:{issue['line']} - {issue['message']}")
    
    # Warnings
    if results['warnings']:
        print(f"\n⚠️ WARNINGS:")
        for warning in results['warnings']:
            print(f"   🟡 {warning['file']} - {warning['message']}")
    
    # Verificações específicas
    print(f"\n🔧 VERIFICAÇÕES ESPECÍFICAS:")
    config_status = "✅" if results['config_ok'] else "❌"
    compat_status = "✅" if results['compatibility_ok'] else "❌"
    pipeline_status = "✅" if results['pipeline_ok'] else "❌"
    
    print(f"   {config_status} Configuração consistente")
    print(f"   {compat_status} Compatibilidade backward")
    print(f"   {pipeline_status} Integração do pipeline")
    
    # Resumo final
    total_problems = results['total_issues'] + results['total_warnings']
    all_checks_ok = results['config_ok'] and results['compatibility_ok'] and results['pipeline_ok']
    
    print(f"\n" + "=" * 50)
    if total_problems == 0 and all_checks_ok:
        print("🎉 REVISÃO COMPLETA: CÓDIGO EXCELENTE!")
        print("✅ Nenhum problema crítico encontrado")
        print("✅ Todas as verificações passaram")
        print("🚀 Sistema pronto para produção")
    else:
        print(f"⚠️ REVISÃO COMPLETA: {total_problems} problemas encontrados")
        if not all_checks_ok:
            print("❌ Algumas verificações falharam")
        print("🔧 Revisar problemas antes do uso em produção")
    
    return total_problems == 0 and all_checks_ok

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Erro durante revisão: {e}")
        traceback.print_exc()
        sys.exit(1)
