import numpy as np
import os

# Verificar dimensão de um arquivo de embedding de proteína
protein_embedding_file = "/home/leon/docktkinase/non_humans/protein_embeddings/10030_protein_embedding.npy"
if os.path.exists(protein_embedding_file):
    embedding = np.load(protein_embedding_file, allow_pickle=True)
    print(f"Dimensão do embedding de proteína: {embedding.shape}")
    
    # Verificar dimensão de um arquivo de embedding de ligante
    ligand_embedding_file = "/home/leon/docktkinase/non_humans/ligand_embeddings/10030_ligand.npy"
    if os.path.exists(ligand_embedding_file):
        ligand_embedding = np.load(ligand_embedding_file, allow_pickle=True)
        print(f"Dimensão do embedding de ligante: {ligand_embedding.shape}")
else:
    print("Arquivo de embedding de proteína não encontrado")