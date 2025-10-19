import os
import numpy as np
import pandas as pd

class EmbeddingCheck:
    def __init__(self, matrix_dir="concatenated_embeddings", original_tsv_path="nr_kinase_all_compounds.tsv"):
        self.matrix_dir = matrix_dir
        self.original_tsv_path = original_tsv_path
        self.concatenated_path = os.path.join(self.matrix_dir, "concatenated_embeddings_normalized.npy")
        self.labels_path = os.path.join(self.matrix_dir, "interaction_labels.npy")
    
    def load_matrix(self, path):
        """Carrega a matriz de embeddings e retorna seu conteúdo."""
        if not os.path.exists(path):
            print(f"\n❌ Erro: Matriz {path} não encontrada.")
            return None
        return np.load(path, allow_pickle=True)
    
    def verify_matrix_shape(self, matrix, expected_rows, expected_dim, name):
        """Verifica se a matriz tem a dimensão correta."""
        if matrix is None:
            return False

        if matrix.shape != (expected_rows, expected_dim):
            print(f"\n❌ Erro: Dimensão incorreta para {name}. Esperado ({expected_rows}, {expected_dim}), encontrado {matrix.shape}.")
            return False
        
        print(f"\n✅ Matriz {name} carregada com sucesso: {matrix.shape}")
        return True
    
    def check_for_empty_values(self, matrix, name):
        """Verifica se há valores NaN ou linhas completamente zeradas na matriz."""
        if matrix is None:
            return False

        if np.isnan(matrix).any():
            print(f"\n❌ Erro: A matriz {name} contém valores NaN!")
            return False

        if np.all(matrix == 0, axis=1).any():
            print(f"\n⚠️ Atenção: A matriz {name} contém linhas totalmente zeradas. Verifique os embeddings.")
            return False
        
        print(f"\n✅ A matriz {name} não contém valores vazios ou zerados.")
        return True
    
    def check_matrices_alignment(self, concatenated_matrix, labels_matrix):
        """Verifica se as matrizes de embeddings e labels possuem o mesmo número de linhas."""
        if concatenated_matrix.shape[0] != labels_matrix.shape[0]:
            print(f"\n❌ Erro: O número de linhas das matrizes não bate.\nEmbeddings: {concatenated_matrix.shape[0]}, Labels: {labels_matrix.shape[0]}")
            return False
        print("\n✅ As matrizes de embeddings e labels possuem o mesmo número de linhas.")
        return True
    
    def run_all_checks(self):
        """Executa todas as verificações para as matrizes de embeddings e labels."""
        print("\n🔍 Carregando matriz concatenada...")
        concatenated_matrix = self.load_matrix(self.concatenated_path)
        labels_matrix = self.load_matrix(self.labels_path)
        if concatenated_matrix is None or labels_matrix is None:
            return
        
        expected_rows = pd.read_csv(self.original_tsv_path, sep='\t').shape[0]
        expected_dim = concatenated_matrix.shape[1]
        
        print("\n🔍 Verificando dimensões da matriz de embeddings:")
        matrix_ok = self.verify_matrix_shape(concatenated_matrix, expected_rows, expected_dim, "concatenated_embeddings.npy")
        
        print("\n🔍 Verificando dimensões da matriz de labels:")
        labels_ok = self.verify_matrix_shape(labels_matrix, expected_rows, labels_matrix.shape[1], "interaction_labels.npy")
        
        print("\n🔍 Verificando se há valores vazios:")
        valid_values = self.check_for_empty_values(concatenated_matrix, "concatenated_embeddings.npy")
        
        print("\n🔍 Verificando alinhamento das matrizes:")
        alignment_ok = self.check_matrices_alignment(concatenated_matrix, labels_matrix)
        
        # Resultado final
        if all([matrix_ok, labels_ok, valid_values, alignment_ok]):
            print("\n✅ Todas as verificações foram concluídas com sucesso! As matrizes estão corretas e alinhadas com o dataset original.")
        else:
            print("\n❌ Foram encontradas inconsistências nas matrizes de embeddings e labels!")

if __name__ == "__main__":
    checker = EmbeddingCheck()
    checker.run_all_checks()
