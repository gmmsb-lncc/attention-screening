#!/usr/bin/env python3
"""
Interface principal do pipeline DockTKinase.
Este arquivo é uma biblioteca chamada pelo executável principal.
"""

import subprocess
import sys
import os

def run_command(command, description, python_executable, pythonpath_extra=""):
    """Executa um comando e verifica se foi bem-sucedido."""
    print(f"\n🟡 Executando: {description}")
    
    env = os.environ.copy()
    if pythonpath_extra:
        env['PYTHONPATH'] = pythonpath_extra + ':' + env.get('PYTHONPATH', '')
    
    # Ensure we're using the correct Python executable
    if command.startswith('python '):
        command = command.replace('python ', python_executable + ' ', 1)
    
    # Also set the PATH to include the conda environment's bin directory
    conda_bin_dir = os.path.dirname(python_executable)
    env['PATH'] = conda_bin_dir + ':' + env.get('PATH', '')
    
    # Set CONDA_PREFIX to ensure the subprocess knows which conda environment is active
    env['CONDA_PREFIX'] = os.path.dirname(os.path.dirname(python_executable))
    
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {description}")
        sys.exit(1)
    print(f"✅ Finalizado: {description}")

def main(config):
    """
    Função principal que executa o pipeline completo.
    Recebe um dicionário de configuração.
    """
    # Extrai variáveis do dicionário de configuração
    output_folder_name = config['OUTPUT_FOLDER_NAME']
    project_root = config['PROJECT_ROOT']
    input_tsv_filename = config['INPUT_TSV_FILENAME']
    python_executable = config['PYTHON_EXECUTABLE']
    pythonpath_extra = config.get('PYTHONPATH_EXTRA', "") # Optional

    # Obter diretórios
    current_dir = os.getcwd()
    
    # Criar diretório de saída se não existir
    output_dir = os.path.join(current_dir, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Caminhos dos arquivos ---
    input_tsv = os.path.join(project_root, "src", "database", input_tsv_filename)
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_tsv):
        print(f"❌ Arquivo de entrada não encontrado: {input_tsv}")
        print("Certifique-se de que o arquivo existe no diretório src/database/")
        sys.exit(1)

    print("Iniciando o pipeline completo de embeddings...")

    # --- Executa o pipeline principal ---
    # Ensure the src directory is in the Python path
    src_path = os.path.join(project_root, 'src')
    if pythonpath_extra:
        full_pythonpath = f"{pythonpath_extra}:{src_path}"
    else:
        full_pythonpath = src_path
        
    run_command(
        f"python {os.path.join(project_root, 'src', 'build', 'buildEmbeddingMain.py')}" + 
        f" {input_tsv} --output_dir {output_dir}",
        "Gerar Embeddings, Matrizes e Labels",
        python_executable=python_executable,
        pythonpath_extra=full_pythonpath
    )

    print("\n🎉 Processo completo concluído com sucesso!")
