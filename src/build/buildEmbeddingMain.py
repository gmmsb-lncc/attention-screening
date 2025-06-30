import sys
import os

from embeddingPreparation import EmbeddingPreparation
from embeddingBuild import EmbeddingBuild

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python buildEmbeddingScratch.py <kinase_all_compounds.tsv> [--ligand_embeddings | --protein_embeddings | --matrix_embeddings]")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        sys.exit(1)

    processor = EmbeddingPreparation(input_file)
    builder = EmbeddingBuild()

    if len(sys.argv) == 2:
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
            builder.run_matrices()
        elif checkpoint == "embedding_matrix":
            print("Processamento já concluído anteriormente. Nenhuma ação necessária.")
        print("Processo completo concluído com sucesso.")
    else:
        argument = sys.argv[2]
        if argument == "--ligand_embeddings":
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
        else:
            print("Argumento inválido. Use --ligand_embeddings, --protein_embeddings ou --matrix_embeddings.")
            sys.exit(1)
