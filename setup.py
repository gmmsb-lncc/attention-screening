#!/usr/bin/env python3
"""
Script de setup automatizado completo para DockTKinase.
Configura ambiente, verifica dependências e valida instalação.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Any

def print_header(title: str) -> None:
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def print_section(title: str) -> None:
    """Imprime seção formatada."""
    print(f"\n📋 {title}")
    print("-" * 40)

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Executa comando e retorna resultado."""
    try:
        print(f"⚙️  {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description}: Sucesso")
            return True, result.stdout
        else:
            print(f"❌ {description}: Falhou")
            print(f"   Erro: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: Timeout (5min)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description}: Erro - {e}")
        return False, str(e)

def check_system_requirements() -> bool:
    """Verifica requisitos do sistema."""
    print_section("VERIFICAÇÃO DO SISTEMA")
    
    # Sistema operacional
    system = platform.system()
    print(f"🖥️  Sistema: {system} {platform.release()}")
    
    # Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ é necessário")
        return False
    
    print("✅ Versão do Python adequada")
    
    # Verificar pip
    try:
        import pip
        print(f"📦 pip: Disponível")
    except ImportError:
        print("❌ pip não encontrado")
        return False
    
    return True

def setup_virtual_environment() -> bool:
    """Configura ambiente virtual."""
    print_section("CONFIGURAÇÃO DO AMBIENTE VIRTUAL")
    
    env_path = Path("env")
    
    # Verificar se já existe
    if env_path.exists():
        print("📁 Ambiente virtual já existe")
        
        # Verificar se está funcional
        python_exe = env_path / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
        if python_exe.exists():
            print("✅ Ambiente virtual funcional")
            return True
        else:
            print("⚠️  Ambiente corrompido, recriando...")
            import shutil
            shutil.rmtree(env_path)
    
    # Criar novo ambiente com Python 3.12
    python_cmd = "python3.12"
    
    # Verificar se Python 3.12 está disponível
    try:
        subprocess.run([python_cmd, "--version"], capture_output=True, check=True)
        print(f"🐍 Usando {python_cmd}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback para python3 ou sys.executable
        python_cmd = "python3" if subprocess.run(["python3", "--version"], capture_output=True).returncode == 0 else sys.executable
        print(f"🐍 Usando {python_cmd}")
    
    success, _ = run_command([python_cmd, "-m", "venv", "env"], "Criando ambiente virtual")
    
    if success:
        print("✅ Ambiente virtual criado")
        return True
    else:
        # Tentar com virtualenv
        success, _ = run_command(["virtualenv", "env"], "Criando com virtualenv")
        return success

def check_package_installed(python_exe: str, package_name: str) -> bool:
    """Verifica se um pacote está instalado."""
    # Mapear nomes de pacotes para módulos de import
    import_map = {
        "scikit-learn": "sklearn",
        "torch-geometric": "torch_geometric",
        "torch-scatter": "torch_scatter",
        "torch-sparse": "torch_sparse",
        "torch-cluster": "torch_cluster",
        "torch-optimizer": "torch_optimizer",
        "umap-learn": "umap",
        "rdkit": "rdkit",
    }
    
    # Extrair nome base do pacote (remover versões)
    base_name = package_name.split(">=")[0].split("==")[0].split("<")[0]
    
    # Obter nome do módulo de import
    import_name = import_map.get(base_name, base_name.replace("-", "_"))
    
    try:
        result = subprocess.run(
            [python_exe, "-c", f"import {import_name}"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def install_dependencies() -> bool:
    """Instala dependências do projeto (apenas as que não estão instaladas)."""
    print_section("INSTALAÇÃO DE DEPENDÊNCIAS")
    
    # Determinar executável Python do ambiente virtual
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
        pip_exe = "env/Scripts/pip.exe"
    else:
        python_exe = "env/bin/python"
        pip_exe = "env/bin/pip"
    
    # Verificar se executáveis existem
    if not Path(python_exe).exists():
        print(f"❌ Python não encontrado em {python_exe}")
        return False
    
    # Atualizar pip
    print("🔄 Verificando pip...")
    success, _ = run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                           "Atualizando pip")
    if not success:
        print("⚠️  Falha ao atualizar pip, continuando...")
    
    # Instalar dependências básicas
    basic_deps = [
        "torch",
        "torchvision", 
        "numpy>=1.26.1",
        "pandas>=1.5.3",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.2",
        "seaborn",
        "jupyter",
        "pytest",
        # Dependências necessárias para src/build
        "tqdm>=4.66.4",           # Barras de progresso (todos os scripts)
        "psutil",                 # System utilities (embeddingIBM.py, embeddingBuild.py)  
        "pyspark>=3.0.0",        # Apache Spark (buildInteractionLabels.py, embeddingBuild.py)
        "optuna",                 # Hyperparameter optimization
        "scipy>=1.12.0",         # Scientific computing (também usado em compare_classifiers.py)
        "pyarrow>=14.0.1",       # Apache Arrow (para Spark e pandas)
        # Dependências para compare_classifiers.py e run_complete_pipeline.py
        "threadpoolctl>=3.1.0",  # Thread pool control (para evitar bugs do KNN)
    ]
    
    # Dependências opcionais (instaladas com tratamento de erro)
    optional_deps = [
        # Embeddings de Proteínas e Ligantes
        # ESM incluído localmente em ESM/ (não precisa instalar fair-esm)
        "transformers>=4.38",     # HuggingFace models (necessário para ESM)
        "sentencepiece",          # Tokenizer (usado por alguns modelos)
        
        # Química e Moléculas
        "rdkit",                  # Chemistry toolkit (instalar via pip no env)
        "umap-learn",             # Dimensionality reduction (FM4M)
        "selfies>=2.1.0",         # SELFIES molecular representation
        "mordred",                # Molecular descriptors
        "numba",                  # JIT compiler (necessário para umap)
        
        # Graph Neural Networks
        "torch-geometric>=2.3.1", # Graph neural networks
        "torch-scatter",          # Scatter operations for PyG
        "torch-sparse",           # Sparse operations for PyG
        "torch-cluster",          # Clustering for PyG
        "networkx>=2.8",          # Graph processing
        
        # Machine Learning e Otimização  
        "xgboost",                # Gradient boosting (necessário para FM4M)
        "torch-optimizer",        # Additional optimizers
        "datasets>=2.13.1",       # HuggingFace datasets
        "evaluate>=0.4.0",        # Model evaluation metrics
        
        # Utilitários
        "requests>=2.32.2",       # HTTP requests
        "urllib3>=2.2.2",         # HTTP client
        "aiohttp>=3.10.2",        # Async HTTP
        "zipp>=3.19.1",           # ZIP utilities
        "torchinfo>=1.8.0",       # Model summary
        "ase",                    # Atomic simulation environment
    ]
    
    print("📦 Verificando e instalando dependências básicas...")
    to_install_basic = []
    for dep in basic_deps:
        if check_package_installed(python_exe, dep):
            print(f"✅ {dep}: Já instalado")
        else:
            print(f"📥 {dep}: Será instalado")
            to_install_basic.append(dep)
    
    if to_install_basic:
        print(f"\n🔨 Instalando {len(to_install_basic)} dependências básicas...")
        for dep in to_install_basic:
            success, _ = run_command([pip_exe, "install", dep], f"Instalando {dep}")
            if not success:
                print(f"❌ Falha ao instalar {dep} (CRÍTICO)")
                return False  # Dependências básicas são obrigatórias
    else:
        print("✅ Todas as dependências básicas já estão instaladas!")
    
    print("\n🔧 Verificando e instalando dependências opcionais...")
    to_install_optional = []
    already_installed_optional = []
    
    for dep in optional_deps:
        if check_package_installed(python_exe, dep):
            already_installed_optional.append(dep)
            print(f"✅ {dep}: Já instalado")
        else:
            print(f"📥 {dep}: Será instalado")
            to_install_optional.append(dep)
    
    failed_optional = []
    if to_install_optional:
        print(f"\n🔨 Instalando {len(to_install_optional)} dependências opcionais...")
        for dep in to_install_optional:
            success, _ = run_command([pip_exe, "install", dep], f"Instalando {dep}")
            if not success:
                failed_optional.append(dep)
                print(f"⚠️  Falha ao instalar {dep} (opcional)")
    else:
        print("✅ Todas as dependências opcionais já estão instaladas!")
    
    # Resumo da instalação
    print("\n" + "=" * 60)
    print("📊 RESUMO DA INSTALAÇÃO")
    print("=" * 60)
    print(f"✅ Básicas instaladas: {len(to_install_basic)}/{len(basic_deps)}")
    print(f"🔧 Opcionais instaladas: {len(to_install_optional) - len(failed_optional)}/{len(optional_deps)}")
    print(f"♻️  Já presentes: {len(basic_deps) - len(to_install_basic) + len(already_installed_optional)}")
    
    if failed_optional:
        print(f"\n⚠️  Dependências opcionais não instaladas ({len(failed_optional)}):")
        for dep in failed_optional:
            print(f"   - {dep}")
        print("\n� Para instalar manualmente:")
        print("   source env/bin/activate")
        for dep in failed_optional:
            print(f"   pip install {dep}")
        print("\n📝 NOTA sobre rdkit:")
        print("   Se rdkit falhar via pip, você pode tentar:")
        print("   pip install rdkit-pypi")
    else:
        print("\n🎉 Todas as dependências foram instaladas com sucesso!")
    
    return True

def verify_installation() -> bool:
    """Verifica se a instalação está funcionando."""
    print_section("VERIFICAÇÃO DA INSTALAÇÃO")
    
    # Executável Python do ambiente virtual
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
    else:
        python_exe = "env/bin/python"
    
    # Testar imports principais
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import pandas; print(f'Pandas: {pandas.__version__}')",
        "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')",
        "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')",
        # Dependências específicas dos scripts de build
        "import tqdm; print(f'tqdm: {tqdm.__version__}')",
        "import psutil; print(f'psutil: {psutil.__version__}')",
        "import pyspark; print(f'PySpark: {pyspark.__version__}')",
        "import optuna; print(f'Optuna: {optuna.__version__}')",
        "import scipy; print(f'SciPy: {scipy.__version__}')",
    ]
    
    # Testar imports opcionais
    optional_imports = [
        "import esm; print(f'ESM: {esm.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
        "import umap; print(f'UMAP: disponível')",
        "import rdkit; print(f'RDKit: {rdkit.__version__}')",
        "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')",
    ]
    
    for test_import in test_imports:
        success, output = run_command([python_exe, "-c", test_import], 
                                    f"Testando: {test_import.split(';')[0]}")
        if success:
            print(f"   ✅ {output.strip()}")
        else:
            print(f"   ❌ Falhou: {test_import.split(';')[0]}")
    
    print("\n🔍 Testando imports opcionais...")
    for test_import in optional_imports:
        success, output = run_command([python_exe, "-c", test_import], 
                                    f"Testando: {test_import.split(';')[0]}")
        if success:
            print(f"   ✅ {output.strip()}")
        else:
            print(f"   ⚠️  Não disponível: {test_import.split(';')[0]}")
    
    # Testar scripts específicos dos build
    print("\n🔧 Testando scripts de build...")
    build_tests = [
        "import sys; sys.path.insert(0, 'src/build'); from buildbinaryLabels import BinaryLabelGenerator; print('✅ buildbinaryLabels: OK')",
        "import sys; sys.path.insert(0, 'src/build'); from checkEmbedding import EmbeddingCheck; print('✅ checkEmbedding: OK')",
        "import sys; sys.path.insert(0, 'src/build'); from buildEmbeddingMatrix import EmbeddingMatrixReconstructor; print('✅ buildEmbeddingMatrix: OK')",
    ]
    
    for test_script in build_tests:
        success, output = run_command([python_exe, "-c", test_script], 
                                    "Testando script de build")
        if success:
            print(f"   {output.strip()}")
        else:
            print("   ❌ Erro ao testar script de build")
    
    # Testar sistema DockTKinase
    print("\n🧪 Testando sistema DockTKinase...")
    
    test_script = '''
import sys
from pathlib import Path

# Adicionar src ao path
src_path = Path.cwd() / "src"
sys.path.insert(0, str(src_path))

try:
    # Testar imports do sistema modularizado
    from classifier.modular_classifier import main as classifier_main
    from classifier.modular_pipeline import MLPEmbeddingPipeline
    from classifier.models.mlp_classifier import MLPEmbeddingClassifier
    from classifier.core.data_loader import DataManager
    from classifier.core.evaluator import ModelEvaluator
    from classifier.utils.import_utils import safe_import_optional
    
    print("✅ Imports do sistema modularizado: OK")
    
    # Testar imports opcionais (Optuna e PySpark)
    optuna_module = safe_import_optional("optuna", "otimização")
    pyspark_module = safe_import_optional("pyspark", "processamento distribuído")
    
    if optuna_module:
        print("✅ Optuna disponível: OK")
    else:
        print("⚠️  Optuna não disponível")
    
    if pyspark_module:
        print("✅ PySpark disponível: OK")  
    else:
        print("⚠️  PySpark não disponível")
    
    # Testar instanciação do pipeline (apenas classe, sem argumentos)
    print("✅ Pipeline modular: Classe disponível")
    
    # Testar instanciação do modelo  
    model = MLPEmbeddingClassifier(input_dim=100, hidden_dim=64, dropout=0.3)
    print("✅ Modelo MLP: OK")
    
    # Testar componentes core (apenas classes, sem argumentos)
    print("✅ Componentes core: Classes disponíveis")
    
    print("🎉 SISTEMA DOCKTKINASE MODULARIZADO FUNCIONAL!")
    
except Exception as e:
    print(f"❌ Erro no teste: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    success, output = run_command([python_exe, "-c", test_script], 
                                "Testando DockTKinase")
    
    if success:
        print(output)
        return True
    else:
        print("❌ Sistema DockTKinase não está funcionando")
        return False

def create_startup_scripts() -> None:
    """Cria scripts de inicialização."""
    print_section("CRIANDO SCRIPTS DE INICIALIZAÇÃO")
    
    # Script para ativar ambiente (Unix)
    if platform.system() != "Windows":
        activate_script = '''#!/bin/bash
# Script para ativar ambiente DockTKinase

echo "🚀 Ativando ambiente DockTKinase..."

# Ativar ambiente virtual
source env/bin/activate

# Adicionar src ao PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "✅ Ambiente ativado!"
echo "📁 PYTHONPATH: $PYTHONPATH"
echo ""
echo "Para usar DockTKinase:"
echo "  python src/classifier/modular_classifier.py --help  # CLI interface"
echo "  python -c 'from classifier.modular_pipeline import ModularMLPPipeline; print(\"Sistema pronto!\")'"
echo "  jupyter lab  # Para notebooks"
echo ""
'''
        
        with open("activate_env.sh", "w") as f:
            f.write(activate_script)
        
        os.chmod("activate_env.sh", 0o755)
        print("✅ activate_env.sh criado")
    
    # Script Python universal
    launcher_script = '''#!/usr/bin/env python3
"""
Launcher para DockTKinase - Configura ambiente e inicia sistema.
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Configura ambiente Python."""
    # Adicionar src ao path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🚀 DockTKinase Launcher")
    print("=" * 30)
    print(f"📁 Projeto: {Path.cwd()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 Src path: {src_path}")

def test_system():
    """Testa se o sistema está funcionando."""
    try:
        from classifier.modular_classifier import main as classifier_main
        from classifier.modular_pipeline import ModularMLPPipeline
        from classifier.models.mlp_classifier import MLPEmbeddingClassifier
        from classifier.utils.import_utils import safe_import_optional
        
        print("✅ Sistema modularizado carregado com sucesso!")
        
        # Testar pipeline
        pipeline = ModularMLPPipeline()
        print("✅ Pipeline: OK")
        
        # Testar modelo
        model = MLPEmbeddingClassifier(input_dim=100, hidden_dims=[64, 32])
        print("✅ Modelo MLP: OK")
        
        # Verificar dependências opcionais
        optuna_available = safe_import_optional("optuna", "otimização")
        pyspark_available = safe_import_optional("pyspark", "processamento distribuído")
        
        print(f"� Optuna: {'✅ Disponível' if optuna_available else '⚠️  Não disponível'}")
        print(f"🔧 PySpark: {'✅ Disponível' if pyspark_available else '⚠️  Não disponível'}")
        
        print("")
        print("Sistema pronto para uso!")
        print("Para começar:")
        print("  from classifier.modular_pipeline import ModularMLPPipeline")
        print("  pipeline = ModularMLPPipeline()")
        print("  # ou usar CLI:")
        print("  python src/classifier/modular_classifier.py --help")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_environment()
    if test_system():
        print("\\n🎉 DockTKinase está pronto!")
    else:
        print("\\n⚠️  Verifique a instalação")
        sys.exit(1)
'''
    
    with open("launch_docktkinase.py", "w") as f:
        f.write(launcher_script)
    
    print("✅ launch_docktkinase.py criado")

def validate_build_dependencies() -> bool:
    """Valida dependências específicas dos scripts de build."""
    print_section("VALIDAÇÃO DE DEPENDÊNCIAS DOS SCRIPTS DE BUILD")
    
    # Executável Python do ambiente virtual
    if platform.system() == "Windows":
        python_exe = "env/Scripts/python.exe"
    else:
        python_exe = "env/bin/python"
    
    if not Path(python_exe).exists():
        print(f"❌ Ambiente virtual não encontrado em {python_exe}")
        return False
    
    # Dependências essenciais para os scripts de build
    essential_build_deps = [
        ("numpy", "import numpy"),
        ("pandas", "import pandas"), 
        ("tqdm", "import tqdm"),
        ("psutil", "import psutil"),
        ("pyspark", "import pyspark"),
    ]
    
    # Dependências opcionais para funcionalidade completa
    optional_build_deps = [
        ("esm", "import esm"),
        ("umap", "import umap"), 
        ("rdkit", "import rdkit"),
        ("transformers", "import transformers"),
        ("torch_geometric", "import torch_geometric"),
    ]
    
    print("🔍 Verificando dependências essenciais...")
    essential_failed = []
    for dep_name, import_cmd in essential_build_deps:
        success, _ = run_command([python_exe, "-c", import_cmd], 
                               f"Testando {dep_name}")
        if success:
            print(f"   ✅ {dep_name}: OK")
        else:
            print(f"   ❌ {dep_name}: FALHOU")
            essential_failed.append(dep_name)
    
    print("\n🔧 Verificando dependências opcionais...")
    optional_failed = []
    for dep_name, import_cmd in optional_build_deps:
        success, _ = run_command([python_exe, "-c", import_cmd], 
                               f"Testando {dep_name}")
        if success:
            print(f"   ✅ {dep_name}: OK")
        else:
            print(f"   ⚠️  {dep_name}: Não disponível")
            optional_failed.append(dep_name)
    
    # Resumo da validação
    print("\n📋 RESUMO DA VALIDAÇÃO:")
    print(f"   ✅ Essenciais: {len(essential_build_deps) - len(essential_failed)}/{len(essential_build_deps)}")
    print(f"   🔧 Opcionais: {len(optional_build_deps) - len(optional_failed)}/{len(optional_build_deps)}")
    
    if essential_failed:
        print(f"\n❌ DEPENDÊNCIAS CRÍTICAS AUSENTES: {', '.join(essential_failed)}")
        print("   Execute novamente o setup ou instale manualmente:")
        for dep in essential_failed:
            print(f"     pip install {dep}")
        return False
    
    if optional_failed:
        print(f"\n⚠️  DEPENDÊNCIAS OPCIONAIS AUSENTES: {', '.join(optional_failed)}")
        print("   Funcionalidade limitada para:")
        if 'esm' in optional_failed:
            print("     - Embeddings de proteínas (embeddingMeta.py)")
        if 'umap' in optional_failed or 'rdkit' in optional_failed:
            print("     - Embeddings de ligantes (embeddingIBM.py via FM4M)")
        if 'transformers' in optional_failed:
            print("     - Modelos de linguagem avançados")
    
    return True

def main() -> int:
    """Função principal do setup."""
    print_header("SETUP AUTOMATIZADO DOCKTKINASE")
    
    # Verificar diretório correto
    if not Path("docktkinase.py").exists():
        print("❌ Execute este script no diretório raiz do DockTKinase")
        return 1
    
    print("📁 Diretório do projeto detectado")
    
    # Lista de passos
    steps = [
        ("Verificar requisitos do sistema", check_system_requirements),
        ("Configurar ambiente virtual", setup_virtual_environment),
        ("Instalar dependências", install_dependencies),
        ("Validar dependências de build", validate_build_dependencies),
        ("Verificar instalação", verify_installation),
        ("Criar scripts de inicialização", lambda: (create_startup_scripts(), True)[1])
    ]
    
    # Executar passos
    for step_name, step_func in steps:
        print_section(step_name.upper())
        
        try:
            success = step_func()
            if success:
                print(f"✅ {step_name}: Concluído")
            else:
                print(f"❌ {step_name}: Falhou")
                print(f"\n⚠️  Setup interrompido em: {step_name}")
                return 1
        except Exception as e:
            print(f"❌ {step_name}: Erro - {e}")
            return 1
    
    # Sucesso
    print_header("SETUP CONCLUÍDO COM SUCESSO!")
    
    print("🎉 DockTKinase foi configurado com sucesso!")
    print("\n📋 Próximos passos:")
    
    if platform.system() != "Windows":
        print("   1. source activate_env.sh")
    print("   2. python launch_docktkinase.py")
    print("   3. jupyter lab (para notebooks)")
    
    print("\n📖 Para usar o sistema:")
    print("   from classifier.modular_pipeline import ModularMLPPipeline")
    print("   pipeline = ModularMLPPipeline()")
    print("   # ou usar interface CLI:")
    print("   python src/classifier/modular_classifier.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
