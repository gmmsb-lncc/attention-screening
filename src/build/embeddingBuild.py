import os
import gc
import psutil
import numpy as np
from pyspark.sql import SparkSession

class EmbeddingBuild:
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
        self.ligand_dir = os.path.join(self.base_dir, 'ligand')
        self.protein_dir = os.path.join(self.base_dir, 'protein')
        self.ligand_output = os.path.join(self.base_dir, 'ligand_embeddings')
        self.protein_output = os.path.join(self.base_dir, 'protein_embeddings')
        self.matrix_output = os.path.join(self.base_dir, 'matrix_embedding')
        self.checkpoint_file = os.path.join(self.base_dir, 'embedding_checkpoint.txt')

        os.makedirs(self.ligand_output, exist_ok=True)
        os.makedirs(self.protein_output, exist_ok=True)
        os.makedirs(self.matrix_output, exist_ok=True)

        self._configure_spark_session()

    def _configure_spark_session(self):
        num_cores = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total // (1024 ** 3)
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
        with open(self.checkpoint_file, 'w') as f:
            f.write(step)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return f.read().strip()
        return None

    def generate_ligand_embeddings(self):
        if self._load_checkpoint() == "ligand_embeddings":
            print("Checkpoint encontrado: Ligand embeddings já processados.")
            return
        from embeddingIBM import EmbeddingIBM
        embedding_ibm = EmbeddingIBM()
        embedding_ibm.process_smiles_folder(self.ligand_dir, self.ligand_output)
        print("Embeddings de ligantes salvos.")
        self._save_checkpoint("ligand_embeddings")

    def generate_protein_embeddings(self):
        if self._load_checkpoint() == "protein_embeddings":
            print("Checkpoint encontrado: Protein embeddings já processados.")
            return
        from embeddingMeta import EmbeddingMeta
        print(f"Criando EmbeddingMeta com seq_input_dir: {self.protein_dir} e output_dir: {self.protein_output}")
        extractor = EmbeddingMeta(seq_input_dir=self.protein_dir, output_dir=self.protein_output)
        print(f"EmbeddingMeta criado com seq_input_dir: {extractor.seq_input_dir}")
        extractor.run()
        print("Embeddings de proteínas salvos.")
        self._save_checkpoint("protein_embeddings")

    def generate_embedding_matrix(self, batch_size=256, use_spark=True):
        if self._load_checkpoint() == "embedding_matrix":
            print("Checkpoint encontrado: Matrizes de embeddings já processadas.")
            return

        def load_embeddings(directory):
            npy_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
            if use_spark:
                rdd = self.spark.sparkContext.parallelize(npy_files)
                return rdd.map(lambda f: np.load(os.path.join(directory, f), allow_pickle=True)).collect()
            else:
                return [np.load(os.path.join(directory, f), allow_pickle=True) for f in npy_files]

        # Check if ligand embeddings exist
        ligand_files = [f for f in os.listdir(self.ligand_output) if f.endswith('.npy')]
        if len(ligand_files) == 0:
            print("Aviso: Nenhum embedding de ligante encontrado. Regenerando embeddings de ligantes.")
            self.run_ligand_embeddings()
            ligand_files = [f for f in os.listdir(self.ligand_output) if f.endswith('.npy')]
            if len(ligand_files) == 0:
                raise ValueError("Falha ao gerar embeddings de ligantes.")

        ligand_embeddings = load_embeddings(self.ligand_output)
        ligand_cls = np.vstack([e[0] for e in ligand_embeddings])
        ligand_mean = np.vstack([e.mean(axis=0) for e in ligand_embeddings])
        np.save(os.path.join(self.matrix_output, 'ligand_matrix_cls.npy'), ligand_cls)
        np.save(os.path.join(self.matrix_output, 'ligand_matrix_mean.npy'), ligand_mean)
        del ligand_embeddings, ligand_cls, ligand_mean
        gc.collect()

        protein_embeddings = load_embeddings(self.protein_output)
        protein_cls = np.vstack([e[0] for e in protein_embeddings])
        protein_mean = np.vstack([e.mean(axis=0) for e in protein_embeddings])
        np.save(os.path.join(self.matrix_output, 'protein_matrix_cls.npy'), protein_cls)
        np.save(os.path.join(self.matrix_output, 'protein_matrix_mean.npy'), protein_mean)
        del protein_embeddings, protein_cls, protein_mean
        gc.collect()

        print("Matrizes salvas.")
        self._save_checkpoint("embedding_matrix")

    def run_ligand_embeddings(self): 
        print(f"Running ligand embeddings. Checking directories:")
        print(f"  Ligand dir: {self.ligand_dir}")
        print(f"  Ligand output dir: {self.ligand_output}")
        ligand_files = [f for f in os.listdir(self.ligand_dir) if f.endswith('.smi')]
        print(f"  Found {len(ligand_files)} .smi files in ligand directory")
        output_files = os.listdir(self.ligand_output)
        print(f"  Found {len(output_files)} files in ligand_embeddings directory")
        
        # Clear checkpoint if ligand embeddings directory is empty
        if len(output_files) == 0:
            checkpoint = self._load_checkpoint()
            if checkpoint == "ligand_embeddings":
                print("Ligand embeddings checkpoint found but no files. Clearing checkpoint.")
                if os.path.exists(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
        
        self.generate_ligand_embeddings()
        
    def run_protein_embeddings(self): self.generate_protein_embeddings()
    def run_matrices(self): self.generate_embedding_matrix()
    def run_all(self):
        self.run_protein_embeddings()
        self.run_ligand_embeddings()
        self.run_matrices()
