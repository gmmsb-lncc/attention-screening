#!/usr/bin/env python3
"""
Interface principal do pipeline DockTKinase.
Este arquivo NÃO DEVE SER MODIFICADO pelos usuários.
Todas as configurações devem ser feitas no arquivo docktkinase.py.
"""

import subprocess
import sys
import os

# Importar configurações do usuário
try:
    from docktkinase import (
        INPUT_TSV_FILENAME,
        EMBEDDING_MATRIX_FILENAME,
        OUTPUT_FOLDER_NAME,
        PROJECT_ROOT,
        PYTHON_EXECUTABLE,
        PYTHONPATH_EXTRA
    )
except ImportError as e:
    print(f"❌ Erro ao importar configurações: {e}")
    print("Certifique-se de que o arquivo 'docktkinase.py' existe no diretório.")
    sys.exit(1)


def run_command(command, description):
    """Executa um comando e verifica se foi bem-sucedido."""
    print(f"\n🟡 Executando: {description}")
    
    # Configurar o ambiente
    env = os.environ.copy()
    if PYTHONPATH_EXTRA:
        env['PYTHONPATH'] = PYTHONPATH_EXTRA + ':' + env.get('PYTHONPATH', '')
    
    # Usar o Python do ambiente configurado
    if command.startswith('python '):
        command = command.replace('python ', PYTHON_EXECUTABLE + ' ', 1)
    
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {description}")
        sys.exit(1)
    print(f"✅ Finalizado: {description}")


def main():
    """Função principal que executa o pipeline completo."""
    # Obter diretórios
    current_dir = os.getcwd()
    
    # Criar diretório de saída se não existir
    output_dir = os.path.join(current_dir, OUTPUT_FOLDER_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Caminhos dos arquivos ---
    input_tsv = os.path.join(PROJECT_ROOT, "src", "database", INPUT_TSV_FILENAME)
    embedding_matrix = os.path.join(current_dir, EMBEDDING_MATRIX_FILENAME)
    interaction_labels = os.path.join(output_dir, "interaction_labels.npy")

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_tsv):
        print(f"❌ Arquivo de entrada não encontrado: {input_tsv}")
        print("Certifique-se de que o arquivo existe no diretório src/database/")
        sys.exit(1)

    print("Sessão Spark configurada com sucesso.")
    print("Iniciando o pipeline completo...")

    # --- Etapa 1: Embeddings ---
    run_command(
        f"python {os.path.join(PROJECT_ROOT, 'src', 'build', 'buildEmbeddingMain.py')} {input_tsv}",
        "Gerar Embeddings"
    )

    # --- Etapa 2: Matriz kinase-composto ---
    run_command(
        f"python {os.path.join(PROJECT_ROOT, 'src', 'build', 'buildEmbeddingMatrix.py')} {input_tsv}",
        "Gerar Matriz de Embeddings"
    )

    # --- Etapa 3: Labels de interação ---
    run_command(
        f"python {os.path.join(PROJECT_ROOT, 'src', 'build', 'buildInteractionLabels.py')} {input_tsv}",
        "Gerar Labels de Interação"
    )

    # --- Etapa 4: Labels binárias ---
    run_command(
        f"python {os.path.join(PROJECT_ROOT, 'src', 'build', 'buildbinaryLabels.py')} {interaction_labels}",
        "Gerar Labels Binárias"
    )

    # --- Etapa 5: Checagem ---
    run_command(
        f"python {os.path.join(PROJECT_ROOT, 'src', 'build', 'checkConcatenate.py')} {input_tsv}",
        "Checar Embeddings Concatenados"
    )

    print("\n🎉 Processo completo concluído com sucesso!")


if __name__ == "__main__":
    main()