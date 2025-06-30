import os
import sys
import gc
import json
import shutil
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor  # Usando threads para I/O
from pyspark.sql import SparkSession  # Necessário para a configuração do Spark

# Funções de salvamento definidas fora da classe para evitar a serialização do objeto inteiro
def save_ligand_func(row, ligand_dir):
    ligand_file = os.path.join(ligand_dir, f"{row['molregno']}_ligand.smi")
    if not os.path.exists(ligand_file):
        with open(ligand_file, 'w') as f:
            f.write(row['canonical_smiles'] + '\n')

def save_protein_func(row, protein_dir):
    protein_file = os.path.join(protein_dir, f"{row['seq_id']}_protein.fasta")
    if not os.path.exists(protein_file):
        with open(protein_file, 'w') as f:
            f.write(f"> {row['target_kinase']}\n{row['seq']}\n")

class EmbeddingPreparation:
    def __init__(self, input_file):
        self.input_file = input_file
        # Em vez de carregar todo o CSV na memória, usaremos leitura incremental (chunks)
        self.ligand_dir = 'ligand'
        self.protein_dir = 'protein'
        os.makedirs(self.ligand_dir, exist_ok=True)
        os.makedirs(self.protein_dir, exist_ok=True)
        self.checkpoint_file = "preparation_checkpoint.txt"

    def checkpoint_exists(self, step):
        """Verifica se um checkpoint para a etapa especificada existe."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                completed_steps = f.read().splitlines()
                return step in completed_steps
        return False

    def save_checkpoint(self, step):
        """Salva um checkpoint indicando que a etapa foi concluída."""
        with open(self.checkpoint_file, 'a') as f:
            f.write(step + '\n')

    def generate_index_files(self):
        if self.checkpoint_exists("index_files"):
            print("Checkpoint encontrado: Indexação já realizada.")
            return

        chunk_size = 10000
        unique_ligands = {}
        unique_proteins = {}
        # Leitura do CSV em chunks para reduzir o uso de memória
        for chunk in pd.read_csv(self.input_file, sep='\t', chunksize=chunk_size):
            for _, row in chunk.iterrows():
                lig_key = row['canonical_smiles']
                if lig_key not in unique_ligands:
                    unique_ligands[lig_key] = {
                        'molregno': row['molregno'],
                        'canonical_smiles': row['canonical_smiles']
                    }
                prot_key = row['seq']
                if prot_key not in unique_proteins:
                    unique_proteins[prot_key] = {
                        'seq_id': row['seq_id'],
                        'seq': row['seq'],
                        'target_kinase': row['target_kinase']
                    }
            del chunk  # Libera a memória do chunk atual
            gc.collect()

        unique_ligands_df = pd.DataFrame(list(unique_ligands.values()))
        unique_proteins_df = pd.DataFrame(list(unique_proteins.values()))
        unique_ligands_df.to_csv('unique_ligands.csv', index=False)
        unique_proteins_df.to_csv('unique_proteins.csv', index=False)
        self.save_checkpoint("index_files")
        print("Arquivos de índices únicos gerados com sucesso.")

    def save_ligands_parallel(self):
        if self.checkpoint_exists("ligands_saved"):
            print("Checkpoint encontrado: Ligantes já salvos.")
            return
        # Usa o arquivo de índices para evitar recarregar o CSV completo
        unique_ligands = pd.read_csv('unique_ligands.csv')
        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(lambda row: save_ligand_func(row, self.ligand_dir),
                             unique_ligands.to_dict(orient='records')),
                total=len(unique_ligands),
                desc="Salvando ligantes"
            ))
        self.save_checkpoint("ligands_saved")
        print("Ligantes salvos com sucesso.")

    def save_proteins_parallel(self):
        if self.checkpoint_exists("proteins_saved"):
            print("Checkpoint encontrado: Proteínas já salvas.")
            return
        # Usa o arquivo de índices para evitar recarregar o CSV completo
        unique_proteins = pd.read_csv('unique_proteins.csv')
        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(lambda row: save_protein_func(row, self.protein_dir),
                             unique_proteins.to_dict(orient='records')),
                total=len(unique_proteins),
                desc="Salvando proteínas"
            ))
        self.save_checkpoint("proteins_saved")
        print("Proteínas salvas com sucesso.")

    def run(self):
        self.generate_index_files()
        self.save_proteins_parallel()
        self.save_ligands_parallel()


class EmbeddingBuild:
    def __init__(self):
        self.ligand_dir = 'ligand'
        self.protein_dir = 'protein'
        self.ligand_output = 'ligand_embeddings'
        self.protein_output = 'protein_embeddings'
        self.matrix_output = 'matrix_embedding'
        self.checkpoint_file = 'embedding_checkpoint.txt'

        os.makedirs(self.ligand_output, exist_ok=True)
        os.makedirs(self.protein_output, exist_ok=True)
        os.makedirs(self.matrix_output, exist_ok=True)
        
        # Configura a sessão Spark dinamicamente
        self._configure_spark_session()

    def _configure_spark_session(self):
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)  # em GB
        self.spark = SparkSession.builder \
            .appName("ChemBERTa Fine-Tuning with Spark") \
            .master(f"local[{num_cores}]") \
            .config("spark.driver.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.memory", f"{int(total_memory * 0.8)}g") \
            .config("spark.executor.instances", f"{max(1, num_cores // 4)}") \
            .config("spark.executor.cores", f"{max(1, num_cores // 4)}") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.executor.memoryOverhead", f"{int(total_memory * 0.1)}g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", f"{int(total_memory * 0.2)}g") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.sql.debug.maxToStringFields", "200") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .config("spark.driver.maxResultSize", f"{int(total_memory * 0.1)}g") \
            .config("spark.sql.shuffle.partitions", f"{num_cores * 4}") \
            .config("spark.default.parallelism", f"{num_cores * 2}") \
            .getOrCreate()
        print("Sessão Spark configurada com sucesso.")

    def _save_checkpoint(self, step):
        """Salva o checkpoint indicando a última etapa concluída."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(step)

    def _load_checkpoint(self):
        """Carrega o último checkpoint salvo."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return f.read().strip()
        return None

    def generate_ligand_embeddings(self):
        checkpoint = self._load_checkpoint()
        if checkpoint == "ligand_embeddings":
            print("Checkpoint encontrado: Ligand embeddings já processados. Pulando...")
            return

        from embeddingIBM import EmbeddingIBM
        embedding_ibm = EmbeddingIBM()
        embedding_ibm.process_smiles_folder(self.ligand_dir, self.ligand_output)
        print("Embeddings de ligantes salvos com sucesso.")
        self._save_checkpoint("ligand_embeddings")

    def generate_protein_embeddings(self):
        checkpoint = self._load_checkpoint()
        if checkpoint == "protein_embeddings":
            print("Checkpoint encontrado: Protein embeddings já processados. Pulando...")
            return

        from embeddingMeta import EmbeddingMeta
        extractor = EmbeddingMeta(seq_input_dir=self.protein_dir, output_dir=self.protein_output)
        extractor.run()
        print("Embeddings de proteínas salvos com sucesso.")
        self._save_checkpoint("protein_embeddings")

    def generate_embedding_matrix(self, batch_size=256, use_spark=True):
        checkpoint = self._load_checkpoint()
        if checkpoint == "embedding_matrix":
            print("Checkpoint encontrado: Matrizes de embeddings já processadas. Pulando...")
            return

        def load_embeddings(directory):
            npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
            if use_spark:
                # Cria um RDD com a lista de arquivos e aplica np.load em paralelo
                rdd = self.spark.sparkContext.parallelize(npy_files)
                embeddings = rdd.map(lambda f: np.load(os.path.join(directory, f), allow_pickle=True)).collect()
            else:
                embeddings = [np.load(os.path.join(directory, f), allow_pickle=True) for f in npy_files]
            return embeddings

        print("Processando embeddings de ligantes...")
        ligand_embeddings = load_embeddings(self.ligand_output)
        all_embeddings_cls = []
        all_embeddings_mean = []
        for i in range(0, len(ligand_embeddings), batch_size):
            batch = ligand_embeddings[i:i + batch_size]
            cls_embeddings = np.vstack([emb[0] for emb in batch])
            mean_embeddings = np.vstack([emb.mean(axis=0) for emb in batch])
            all_embeddings_cls.append(cls_embeddings)
            all_embeddings_mean.append(mean_embeddings)
        ligand_cls_embeddings = np.vstack(all_embeddings_cls)
        ligand_mean_embeddings = np.vstack(all_embeddings_mean)
        np.save(os.path.join(self.matrix_output, 'ligand_matrix_cls.npy'), ligand_cls_embeddings)
        np.save(os.path.join(self.matrix_output, 'ligand_matrix_mean.npy'), ligand_mean_embeddings)
        del ligand_embeddings, ligand_cls_embeddings, ligand_mean_embeddings
        gc.collect()

        print("Processando embeddings de proteínas...")
        protein_embeddings = load_embeddings(self.protein_output)
        all_embeddings_cls = []
        all_embeddings_mean = []
        for i in range(0, len(protein_embeddings), batch_size):
            batch = protein_embeddings[i:i + batch_size]
            cls_embeddings = np.vstack([emb[0] for emb in batch])
            mean_embeddings = np.vstack([emb.mean(axis=0) for emb in batch])
            all_embeddings_cls.append(cls_embeddings)
            all_embeddings_mean.append(mean_embeddings)
        protein_cls_embeddings = np.vstack(all_embeddings_cls)
        protein_mean_embeddings = np.vstack(all_embeddings_mean)
        np.save(os.path.join(self.matrix_output, 'protein_matrix_cls.npy'), protein_cls_embeddings)
        np.save(os.path.join(self.matrix_output, 'protein_matrix_mean.npy'), protein_mean_embeddings)
        del protein_embeddings, protein_cls_embeddings, protein_mean_embeddings
        gc.collect()

        print("Matrizes de embeddings de ligantes e proteínas salvas com sucesso.")
        self._save_checkpoint("embedding_matrix")

    def run_ligand_embeddings(self):
        self.generate_ligand_embeddings()

    def run_protein_embeddings(self):
        self.generate_protein_embeddings()

    def run_matrices(self):
        self.generate_embedding_matrix()

    def run_all(self):
        self.run_protein_embeddings()
        self.run_ligand_embeddings()
        self.run_matrices()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python script.py <kinase_all_compounds.tsv> [--ligand_embeddings | --protein_embeddings | --matrix_embeddings]")
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
