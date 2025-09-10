import sys
import os
import argparse

# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

from embeddingPreparation import EmbeddingPreparation
from embeddingBuild import EmbeddingBuild

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de construção de embeddings.")
    parser.add_argument("input_tsv_file", help="Caminho para o arquivo TSV de entrada.")
    parser.add_argument("--output_dir", default=".", help="Diretório para salvar todos os arquivos de saída.")
    parser.add_argument("--run_mode", choices=["ligand_embeddings", "protein_embeddings", "matrix_embeddings"], help="Executar apenas uma parte específica do pipeline.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.input_tsv_file):
        print(f"Erro: O arquivo '{args.input_tsv_file}' não foi encontrado.")
        sys.exit(1)

    # Passa o diretório de saída para as classes
    processor = EmbeddingPreparation(args.input_tsv_file, base_dir=args.output_dir)
    builder = EmbeddingBuild(base_dir=args.output_dir)

    if args.run_mode is None:
        print("Iniciando o pipeline completo...")
        checkpoint = builder._load_checkpoint()
        if checkpoint is None or checkpoint == "":
            processor.run()
            builder.run_all()
        elif checkpoint == "ligand_embeddings":
            print("Retomando a partir de embeddings de ligantes...")
            builder.run_protein_embeddings()
            builder.run_matrices()
        elif checkpoint == "protein_embeddings":
            print("Retomando a partir de embeddings de proteínas...")
            # Check if ligand embeddings exist before running matrices
            ligand_files = [f for f in os.listdir(builder.ligand_output) if f.endswith('.npy')]
            if len(ligand_files) == 0:
                print("Aviso: Nenhum embedding de ligante encontrado. Regenerando embeddings de ligantes.")
                builder.run_ligand_embeddings()
            builder.run_matrices()
        elif checkpoint == "embedding_matrix":
            print("Processamento já concluído anteriormente. Nenhuma ação necessária.")
        print("Processo completo concluído com sucesso.")
    else:
        argument = args.run_mode
        if argument == "ligand_embeddings":
            if builder._load_checkpoint() == "ligand_embeddings":
                print("Ligand embeddings já foram gerados. Pulando...")
            else:
                print("Executando apenas a geração de embeddings de ligantes.")
                builder.run_ligand_embeddings()
        elif argument == "--protein_embeddings":
            if builder._load_checkpoint() in ["ligand_embeddings", "protein_embeddings"]:
                print("Protein embeddings já foram gerados. Pulando...")
            else:
                print("Executando apenas a geração de embeddings de proteínas.")
                builder.run_protein_embeddings()
        elif argument == "--matrix_embeddings":
            if builder._load_checkpoint() == "embedding_matrix":
                print("As matrizes de embeddings já foram geradas. Pulando...")
            else:
                print("Executando apenas a geração das matrizes de embeddings.")
                builder.run_matrices()