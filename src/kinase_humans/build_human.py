#!/usr/bin/env python3
import subprocess
import sys
import os


INPUT = "kinase_human_compounds.tsv"
EMBEDDING_MATRIX = "nr_human.tsv"


def run_command(command, description):
    print(f"\n🟡 Executando: {description}")
    # Usar o mesmo ambiente virtual para os subprocessos
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/leon/Desktop/latent_extractor/ibm/materials:' + env.get('PYTHONPATH', '')
    
    # Usar o Python do ambiente virtual
    python_executable = '/home/leon/docktkinase/env/bin/python'
    if command.startswith('python '):
        command = command.replace('python ', python_executable + ' ', 1)
    
    result = subprocess.run(command, shell=True, env=env)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {description}")
        sys.exit(1)
    print(f"✅ Finalizado: {description}")

def main():
    # Obter o diretório do projeto e o diretório de execução
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.getcwd()
    
    # --- Caminhos dos arquivos ---
    input_tsv = os.path.join(project_root, "src", "database", INPUT)  # entrada inicial
    embedding_matrix = os.path.join(current_dir, EMBEDDING_MATRIX)  # saída do primeiro passo
    interaction_labels = os.path.join(current_dir, "concatenated_embeddings", "interaction_labels.npy")  # corrigido o nome do arquivo

    # --- Etapa 1: Embeddings ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildEmbeddingMain.py')} {input_tsv}", "Gerar Embeddings")

    # --- Etapa 2: Matriz kinase-composto ---
    # Corrigido para passar o arquivo TSV original em vez de nr_non_human.tsv
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildEmbeddingMatrix.py')} {input_tsv}", "Gerar Matriz de Embeddings")

    # --- Etapa 3: Labels de interação ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildInteractionLabels.py')} {input_tsv}", "Gerar Labels de Interação")

    # --- Etapa 4: Labels binárias ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildbinaryLabels.py')} {interaction_labels}", "Gerar Labels Binárias")

    # --- Etapa 5: Checagem ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'checkConcatenate.py')} {input_tsv}", "Checar Embeddings Concatenados")

if __name__ == "__main__":
    main()
