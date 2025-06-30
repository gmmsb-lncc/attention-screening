import subprocess
import sys

def run_command(command, description):
    print(f"\n🟡 Executando: {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Erro ao executar: {description}")
        sys.exit(1)
    print(f"✅ Finalizado: {description}")

def main():
    # --- Caminhos dos arquivos ---
    input_tsv = "kinase_all_compounds.tsv"  # entrada inicial
    kinase_matrix = "nr_kinase_all_compounds.tsv"  # saída do primeiro passo
    interaction_labels = "concatenated_embeddings/interaction_labels.npy"  # saída do terceiro passo

    # --- Etapa 1: Embeddings ---
    run_command(f"python buildEmbeddingMain.py {input_tsv}", "Gerar Embeddings")

    # --- Etapa 2: Matriz kinase-composto ---
    run_command(f"python buildKinaseMatrix.py {kinase_matrix}", "Gerar Matriz Kinase")

    # --- Etapa 3: Labels de interação ---
    run_command(f"python buildInteractionLabels.py {kinase_matrix}", "Gerar Labels de Interação")

    # --- Etapa 4: Labels binárias ---
    run_command(f"python buildBinaryLabels.py {interaction_labels}", "Gerar Labels Binárias")

    # --- Etapa 5: Checagem ---
    run_command(f"python checkConcatenate.py {kinase_matrix}", "Checar Embeddings Concatenados")

if __name__ == "__main__":
    main()
