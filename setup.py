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
    
    # Criar novo ambiente
    success, _ = run_command([sys.executable, "-m", "venv", "env"], "Criando ambiente virtual")
    
    if success:
        print("✅ Ambiente virtual criado")
        return True
    else:
        # Tentar com virtualenv
        success, _ = run_command(["virtualenv", "env"], "Criando com virtualenv")
        return success

def install_dependencies() -> bool:
    """Instala dependências do projeto."""
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
    success, _ = run_command([python_exe, "-m", "pip", "install", "--upgrade", "pip"], 
                           "Atualizando pip")
    if not success:
        print("⚠️  Falha ao atualizar pip, continuando...")
    
    # Instalar dependências básicas
    basic_deps = [
        "torch",
        "torchvision", 
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "pytest",
        # Dependências necessárias para src/build
        "tqdm>=4.66.4",           # Barras de progresso (todos os scripts)
        "psutil",                 # System utilities (embeddingIBM.py, embeddingBuild.py)  
        "pyspark>=3.0.0",        # Apache Spark (buildInteractionLabels.py, embeddingBuild.py)
        "optuna",                 # Hyperparameter optimization (já adicionado anteriormente)
    ]
    
    # Dependências opcionais (instaladas com tratamento de erro)
    optional_deps = [
        "fair-esm",               # ESM models para embeddings de proteínas
        "umap-learn",             # Required by FM4M
        "rdkit",                  # Chemistry toolkit  
        "transformers>=4.38",     # HuggingFace models
        "torch-geometric>=2.3.1", # Graph neural networks
        "datasets>=2.13.1",       # HuggingFace datasets
        "requests>=2.32.2",       # HTTP requests
        "networkx>=2.8",          # Graph processing
    ]
    
    print("📦 Instalando dependências básicas...")
    for dep in basic_deps:
        success, _ = run_command([pip_exe, "install", dep], f"Instalando {dep}")
        if not success:
            print(f"❌ Falha ao instalar {dep} (CRÍTICO)")
            return False  # Dependências básicas são obrigatórias
    
    print("🔧 Instalando dependências opcionais...")
    failed_optional = []
    for dep in optional_deps:
        success, _ = run_command([pip_exe, "install", dep], f"Instalando {dep}")
        if not success:
            failed_optional.append(dep)
            print(f"⚠️  Falha ao instalar {dep} (opcional)")
    
    if failed_optional:
        print(f"\n⚠️  Dependências opcionais não instaladas: {', '.join(failed_optional)}")
        print("   - Funcionalidade pode estar limitada (ex: sem ESM para proteínas)")
        print("   - Para instalar manualmente: pip install <nome_da_dependencia>")
    else:
        print("✅ Todas as dependências opcionais instaladas com sucesso!")
    
    # Verificar se requirements.txt existe
    req_file = Path("requirements.txt")
    if req_file.exists():
        print("📋 Instalando dependências do requirements.txt...")
        success, _ = run_command([pip_exe, "install", "-r", "requirements.txt"], 
                               "Instalando requirements.txt")
        if not success:
            print("⚠️  Algumas dependências podem ter falhado")
    
    # Verificar se environment.yml existe (conda)
    conda_file = Path("environment.yml")
    if conda_file.exists():
        print("🐍 Arquivo conda detectado")
        # Tentar instalar com conda se disponível
        try:
            success, _ = run_command(["conda", "env", "update", "-f", "environment.yml"], 
                                   "Atualizando ambiente conda")
        except:
            print("ℹ️  Conda não disponível, usando pip")
    
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
        # Dependências específicas dos scripts de build
        "import tqdm; print(f'tqdm: {tqdm.__version__}')",
        "import psutil; print(f'psutil: {psutil.__version__}')",
        "import pyspark; print(f'PySpark: {pyspark.__version__}')",
    ]
    
    # Testar imports opcionais
    optional_imports = [
        "import esm; print(f'ESM: disponível')",
        "import umap; print(f'UMAP: disponível')",
        "import rdkit; print(f'RDKit: {rdkit.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
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
    from classifier.config.mlp_config import MLPConfig
    from classifier.utils.config_manager import ConfigManager
    from classifier.utils.device_manager import SmartDeviceManager
    
    print("✅ Imports principais: OK")
    
    # Testar instanciação
    config = MLPConfig()
    config_mgr = ConfigManager()
    device_mgr = SmartDeviceManager()
    
    print("✅ Instanciação: OK")
    
    # Testar funcionalidade básica
    device = device_mgr.get_device()
    template_config = config_mgr.create_config("development")
    
    print(f"✅ Device detectado: {device}")
    print(f"✅ Template criado: {template_config.model.hidden_layers}")
    print("🎉 SISTEMA DOCKTKINASE FUNCIONAL!")
    
except Exception as e:
    print(f"❌ Erro no teste: {e}")
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
echo "  python -c 'from classifier.main import MLPPipeline; print(\"Sistema pronto!\")'"
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
        from classifier.config.mlp_config import MLPConfig
        from classifier.utils.config_manager import ConfigManager
        from classifier.utils.device_manager import SmartDeviceManager
        
        print("✅ Sistema carregado com sucesso!")
        
        # Informações do sistema
        device_mgr = SmartDeviceManager()
        device = device_mgr.get_device()
        
        print(f"🖥️  Device: {device}")
        print("")
        print("Sistema pronto para uso!")
        print("Para começar:")
        print("  from classifier.main import MLPPipeline")
        print("  pipeline = MLPPipeline()")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
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
    print("   from classifier.main import MLPPipeline")
    print("   pipeline = MLPPipeline()")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
