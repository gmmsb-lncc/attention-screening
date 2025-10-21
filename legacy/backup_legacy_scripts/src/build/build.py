import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n🟡 Executando: {description}")
    # Usar o mesmo ambiente virtual para os subprocessos
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/leon/Desktop/latent_extractor/ibm/FM4M:' + env.get('PYTHONPATH', '')
    
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
    # Obter o diretório do script e o diretório de execução
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    current_dir = os.getcwd()
    
    # --- Caminhos dos arquivos ---
    input_tsv = os.path.join(project_root, "src", "database", "kinase_non_human_compounds.tsv")  # entrada inicial
    embedding_matrix = os.path.join(current_dir, "nr_kinase_all_compounds.tsv")  # saída do primeiro passo
    interaction_labels = os.path.join(current_dir, "concatenated_embeddings", "interaction_labels.npy")  # saída do terceiro passo

    # --- Etapa 1: Embeddings ---
    run_command(f"python {os.path.join(script_dir, 'buildEmbeddingMain.py')} {input_tsv}", "Gerar Embeddings")

    # --- Etapa 2: Matriz kinase-composto ---
    run_command(f"python {os.path.join(script_dir, 'buildEmbeddingMatrix.py')}", "Gerar Matriz de Embeddings")

    # --- Etapa 3: Labels de interação ---
    run_command(f"python {os.path.join(script_dir, 'buildInteractionLabels.py')}", "Gerar Labels de Interação")

    # --- Etapa 4: Labels binárias ---
    run_command(f"python {os.path.join(script_dir, 'buildbinaryLabels.py')}", "Gerar Labels Binárias")

    # --- Etapa 5: Checagem ---
    run_command(f"python {os.path.join(script_dir, 'checkConcatenate.py')}", "Checar Embeddings Concatenados")
    
    # --- Etapa 6: Checagem de Embeddings ---
    run_command(f"python {os.path.join(script_dir, 'checkEmbedding.py')}", "Checar Matrizes de Embeddings")

if __name__ == "__main__":
    main()
