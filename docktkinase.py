#!/usr/bin/env python3
"""
Arquivo principal de configuração e execução do pipeline DockTKinase.
Este é o único arquivo que os usuários devem modificar para configurar e executar o pipeline.
Para executar, use: python docktkinase.py
"""

import os
import sys

# Adiciona o diretório 'src' ao path para permitir a importação do 'interface'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from interface import main as run_pipeline
except ImportError:
    print("❌ Erro: Não foi possível encontrar o arquivo 'interface.py' no diretório 'src/'.")
    print("Certifique-se de que a estrutura de arquivos está correta.")
    sys.exit(1)

# === CONFIGURAÇÕES DO USUÁRIO ===

# Nome do arquivo de entrada TSV (deve estar em src/database/)
INPUT_TSV_FILENAME = "kinase_non_human_compounds.tsv"

# Nome da pasta de saída para todos os resultados do pipeline
OUTPUT_FOLDER_NAME = "non_human"

# === CONFIGURAÇÕES DO AMBIENTE (geralmente não precisam ser alteradas) ===

# Diretório base do projeto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Executável Python a ser usado nos subprocessos
# Detecta automaticamente o executável Python do ambiente atual
PYTHON_EXECUTABLE = sys.executable

# PYTHONPATH extra (se necessário)
PYTHONPATH_EXTRA = ""

def main():
    """Prepara a configuração e inicia o pipeline."""
    print("========================================")
    print("🚀 Iniciando Pipeline DockTKinase 🚀")
    print("========================================")

    config = {
        'INPUT_TSV_FILENAME': INPUT_TSV_FILENAME,
        'OUTPUT_FOLDER_NAME': OUTPUT_FOLDER_NAME,
        'PROJECT_ROOT': PROJECT_ROOT,
        'PYTHON_EXECUTABLE': PYTHON_EXECUTABLE,
        'PYTHONPATH_EXTRA': PYTHONPATH_EXTRA,
    }

    run_pipeline(config)


if __name__ == "__main__":
    main()
