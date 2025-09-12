#!/usr/bin/env python3
"""
Arquivo de configuração do pipeline DockTKinase.
Este é o único arquivo que os usuários devem modificar para configurar o pipeline.
"""

import os

# === CONFIGURAÇÕES DO USUÁRIO ===

# Nome do arquivo de entrada TSV (deve estar em src/database/)
INPUT_TSV_FILENAME = "kinase_non_human_compounds.tsv"

# Nome do arquivo de saída da matriz de embeddings
EMBEDDING_MATRIX_FILENAME = "nr_non_human.tsv"

# Nome da pasta de saída para todos os resultados do pipeline
OUTPUT_FOLDER_NAME = "concatenated_embeddings"

# === CONFIGURAÇÕES DO AMBIENTE ===

# Diretório base do projeto (normalmente detectado automaticamente)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diretório do ambiente Conda (detectado automaticamente)
CONDA_ENV_PATH = os.environ.get('CONDA_PREFIX', None)
if CONDA_ENV_PATH:
    PYTHON_EXECUTABLE = os.path.join(CONDA_ENV_PATH, 'bin', 'python')
else:
    # Fallback para o executável Python atual
    PYTHON_EXECUTABLE = os.path.join(os.sys.prefix, 'bin', 'python')

# PYTHONPATH (normalmente não precisa ser modificado)
PYTHONPATH_EXTRA = ""

# === FIM DAS CONFIGURAÇÕES ===