import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

class EmbeddingMatrixReconstructor:
    def __init__(self, original_tsv_path, ligand_embeddings_dir='ligand_embeddings', protein_embeddings_dir='protein_embeddings', output_dir='concatenated_embeddings', embedding_type='cls'):
        self.original_tsv_path = original_tsv_path
        self.ligand_embeddings_dir = ligand_embeddings_dir
        self.protein_embeddings_dir = protein_embeddings_dir
        self.output_dir = output_dir
        self.embedding_type = embedding_type  # 'cls' ou 'mean'
        self.ligand_dim = None  # Será determinado dinamicamente
        self.protein_dim = None  # Será determinado dinamicamente
        os.makedirs(self.output_dir, exist_ok=True)
        self.ligand_cache = {}
        self.protein_cache = {}
        self.missing_log_file = os.path.join(self.output_dir, "missing_embeddings.log")
    
    def _load_embedding(self, file_path, is_protein=False):
        """Carrega um embedding a partir de um arquivo .npy e aplica o tipo (CLS ou média)."""
        try:
            embedding = np.load(file_path, allow_pickle=True)
            if self.embedding_type == 'cls':
                return embedding[1, :] if is_protein else embedding[0, :]
            elif self.embedding_type == 'mean':
                return np.mean(embedding, axis=0)
            else:
                raise ValueError(f"Tipo de embedding inválido: {self.embedding_type}. Use 'cls' ou 'mean'.")
        except Exception as e:
            print(f"Erro ao carregar embedding: {file_path} -> {e}")
            return None
    
    def reconstruct_matrix(self):
        """Reconstrói a matriz de embeddings concatenados preservando todas as linhas do TSV original."""
        df = pd.read_csv(self.original_tsv_path, sep='\t', dtype={'molregno': str, 'seq_id': str})
        
        # Determinar dinamicamente as dimensões dos embeddings
        self._determine_embedding_dimensions()
        
        concatenated_embeddings = []
        missing_entries = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processando pares molregno+seq_id"):
            molregno = row['molregno']
            seq_id = row['seq_id']
            
            ligand_emb = self._load_embedding(os.path.join(self.ligand_embeddings_dir, f"{molregno}_ligand.npy"))
            protein_emb = self._load_embedding(os.path.join(self.protein_embeddings_dir, f"{seq_id}_protein_embedding.npy"), is_protein=True)
            
            if ligand_emb is None or protein_emb is None:
                missing_entries.append(f"molregno: {molregno}, seq_id: {seq_id}")
                final_embedding = np.zeros(self.ligand_dim + self.protein_dim)
            else:
                final_embedding = np.zeros(self.ligand_dim + self.protein_dim)
                final_embedding[:self.protein_dim] = protein_emb
                final_embedding[-self.ligand_dim:] = ligand_emb
            
            concatenated_embeddings.append(final_embedding)
        
        if missing_entries:
            with open(self.missing_log_file, 'w') as log_file:
                log_file.write("\n".join(missing_entries))
            print(f"Arquivo de log salvo em {self.missing_log_file}, contendo {len(missing_entries)} entradas ausentes.")
        
        matrix = np.vstack(concatenated_embeddings)
        return matrix
    
    def _determine_embedding_dimensions(self):
        """Determina dinamicamente as dimensões dos embeddings de ligantes e proteínas."""
        import glob
        
        # Determinar dimensão dos embeddings de ligantes
        ligand_files = glob.glob(os.path.join(self.ligand_embeddings_dir, "*_ligand.npy"))
        if ligand_files:
            sample_embedding = np.load(ligand_files[0], allow_pickle=True)
            if self.embedding_type == 'cls':
                self.ligand_dim = sample_embedding.shape[1]  # Segunda dimensão para CLS
            else:  # mean
                self.ligand_dim = sample_embedding.shape[1]  # Mesma lógica para mean
        else:
            raise ValueError(f"Nenhum arquivo de embedding de ligante encontrado em {self.ligand_embeddings_dir}")
        
        # Determinar dimensão dos embeddings de proteínas
        protein_files = glob.glob(os.path.join(self.protein_embeddings_dir, "*_protein_embedding.npy"))
        if protein_files:
            sample_embedding = np.load(protein_files[0], allow_pickle=True)
            if self.embedding_type == 'cls':
                self.protein_dim = sample_embedding.shape[1]  # Segunda dimensão para CLS
            else:  # mean
                self.protein_dim = sample_embedding.shape[1]  # Mesma lógica para mean
        else:
            raise ValueError(f"Nenhum arquivo de embedding de proteína encontrado em {self.protein_embeddings_dir}")
        
        print(f"Dimensões determinadas automaticamente: ligand_dim={self.ligand_dim}, protein_dim={self.protein_dim}")
    
    def normalize_matrix(self, matrix):
        """Normaliza a matriz entre 0 e 1."""
        min_val = np.min(matrix, axis=0, keepdims=True)
        max_val = np.max(matrix, axis=0, keepdims=True)
        normalized_matrix = (matrix - min_val) / (max_val - min_val + 1e-8)
        return normalized_matrix
    
    def save_matrix(self, matrix, file_name='concatenated_embeddings.npy'):
        """Salva a matriz de embeddings concatenados e a matriz normalizada."""
        output_path = os.path.join(self.output_dir, file_name)
        np.save(output_path, matrix)
        print(f"Matriz de embeddings concatenados salva em: {output_path}")
        
        normalized_matrix = self.normalize_matrix(matrix)
        normalized_output_path = os.path.join(self.output_dir, "concatenated_embeddings_normalized.npy")
        np.save(normalized_output_path, normalized_matrix)
        print(f"Matriz de embeddings normalizados salva em: {normalized_output_path}")
    
    def run(self):
        """Executa o pipeline completo de reconstrução."""
        matrix = self.reconstruct_matrix()
        self.save_matrix(matrix)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso correto: python buildEmbeddingMatrix.py <input_tsv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    ligand_embedding_dir = "ligand_embeddings"
    protein_embedding_dir = "protein_embeddings"
    output_dir = "concatenated_embeddings"
    
    reconstructor = EmbeddingMatrixReconstructor(input_file, ligand_embedding_dir, protein_embedding_dir, output_dir)
    reconstructor.run()