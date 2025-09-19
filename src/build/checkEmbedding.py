import os
import numpy as np

class EmbeddingCheck:
    def __init__(self, matrix_dir="matrix_embedding", ligand_emb_dir="ligand_embeddings", protein_emb_dir="protein_embeddings"):
        self.matrix_dir = matrix_dir
        self.ligand_emb_dir = ligand_emb_dir
        self.protein_emb_dir = protein_emb_dir

        # Caminhos das matrizes
        self.ligand_path = os.path.join(self.matrix_dir, "ligand_matrix.npy")
        self.protein_cls_path = os.path.join(self.matrix_dir, "protein_matrix_cls.npy")
        self.protein_mean_path = os.path.join(self.matrix_dir, "protein_matrix_mean.npy")

        # Dimensões esperadas
        self.expected_ligand_dim = 768
        self.expected_protein_dim = 2560

        # Contagem esperada de arquivos
        self.num_ligands = len([f for f in os.listdir(self.ligand_emb_dir) if f.endswith(".npy")])
        self.num_proteins = len([f for f in os.listdir(self.protein_emb_dir) if f.endswith(".npy")])

    def load_matrix(self, path):
        """Carrega a matriz de embeddings e retorna seu conteúdo."""
        if not os.path.exists(path):
            print(f"❌ Erro: Matriz {path} não encontrada.")
            return None
        return np.load(path)

    def verify_matrix_shape(self, matrix, expected_rows, expected_dim, name):
        """Verifica se a matriz tem a dimensão correta."""
        if matrix is None:
            return False

        if matrix.shape != (expected_rows, expected_dim):
            print(f"❌ Erro: Dimensão incorreta para {name}. Esperado ({expected_rows}, {expected_dim}), encontrado {matrix.shape}.")
            return False

        print(f"✅ Matriz {name} está correta: {matrix.shape}.")
        return True

    def check_for_empty_values(self, matrix, name):
        """Verifica se há valores NaN ou linhas completamente zeradas na matriz."""
        if matrix is None:
            return False

        if np.isnan(matrix).any():
            print(f"❌ Erro: A matriz {name} contém valores NaN!")
            return False

        if np.all(matrix == 0, axis=1).any():
            print(f"⚠️ Atenção: A matriz {name} contém linhas totalmente zeradas. Verifique os embeddings.")
            return False

        print(f"✅ A matriz {name} não contém valores vazios ou zerados.")
        return True

    def run_all_checks(self):
        """Executa todas as verificações para as matrizes de embeddings."""
        print("\n🔍 Carregando matrizes...")
        ligand_matrix = self.load_matrix(self.ligand_path)
        protein_cls = self.load_matrix(self.protein_cls_path)
        protein_mean = self.load_matrix(self.protein_mean_path)

        print("\n🔍 Verificando dimensões das matrizes:")
        lig_ok = self.verify_matrix_shape(ligand_matrix, self.num_ligands, self.expected_ligand_dim, "ligand_matrix.npy")
        prot_cls_ok = self.verify_matrix_shape(protein_cls, self.num_proteins, self.expected_protein_dim, "protein_matrix_cls.npy")
        prot_mean_ok = self.verify_matrix_shape(protein_mean, self.num_proteins, self.expected_protein_dim, "protein_matrix_mean.npy")

        print("\n🔍 Verificando se há valores vazios:")
        lig_valid = self.check_for_empty_values(ligand_matrix, "ligand_matrix.npy")
        prot_cls_valid = self.check_for_empty_values(protein_cls, "protein_matrix_cls.npy")
        prot_mean_valid = self.check_for_empty_values(protein_mean, "protein_matrix_mean.npy")

        # Resultado final
        if all([lig_ok, prot_cls_ok, prot_mean_ok, lig_valid, prot_cls_valid, prot_mean_valid]):
            print("\n✅ Todas as verificações foram concluídas com sucesso! As matrizes estão corretas.")
        else:
            print("\n❌ Foram encontradas inconsistências nas matrizes de embeddings!")

if __name__ == "__main__":
    checker = EmbeddingCheck()
    checker.run_all_checks()
