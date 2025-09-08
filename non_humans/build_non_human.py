#!/usr/bin/env python3
import subprocess
import sys
import os

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
    input_tsv = os.path.join(project_root, "src", "database", "kinase_non_human_compounds.tsv")  # entrada inicial
    embedding_matrix = os.path.join(current_dir, "nr_non_human.tsv")  # saída do primeiro passo
    interaction_labels = os.path.join(current_dir, "concatenated_embeddings", "interaction_labels_non_human.npy")  # saída do terceiro passo

    # --- Etapa 1: Embeddings ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildEmbeddingMain.py')} {input_tsv}", "Gerar Embeddings")

    # --- Etapa 2: Matriz kinase-composto ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildEmbeddingMatrix.py')} {embedding_matrix}", "Gerar Matriz de Emnbeddings")

    # --- Etapa 3: Labels de interação ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildInteractionLabels.py')} {embedding_matrix}", "Gerar Labels de Interação")

    # --- Etapa 4: Labels binárias ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'buildbinaryLabels.py')} {interaction_labels}", "Gerar Labels Binárias")

    # --- Etapa 5: Checagem ---
    run_command(f"python {os.path.join(project_root, 'src', 'build', 'checkConcatenate.py')} {embedding_matrix}", "Checar Embeddings Concatenados")

if __name__ == "__main__":
    main()