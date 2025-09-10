import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, input_file, base_dir='.'):
        self.input_file = input_file
        self.base_dir = base_dir
        self.ligand_dir = os.path.join(self.base_dir, 'ligand')
        self.protein_dir = os.path.join(self.base_dir, 'protein')
        os.makedirs(self.ligand_dir, exist_ok=True)
        os.makedirs(self.protein_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.base_dir, "preparation_checkpoint.txt")

    def checkpoint_exists(self, step):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return step in f.read().splitlines()
        return False

    def save_checkpoint(self, step):
        with open(self.checkpoint_file, 'a') as f:
            f.write(step + '\n')

    def generate_index_files(self):
        if self.checkpoint_exists("index_files"):
            print("Checkpoint encontrado: Indexação já realizada.")
            return
        chunk_size = 10000
        unique_ligands, unique_proteins = {}, {}
        for chunk in pd.read_csv(self.input_file, sep='\t', chunksize=chunk_size):
            for _, row in chunk.iterrows():
                lig_key = row['canonical_smiles']
                if lig_key not in unique_ligands:
                    unique_ligands[lig_key] = {
                        'molregno': row['molregno'],
                        'canonical_smiles': row['canonical_smiles']
                    }
                prot_key = row['seq_id']  # Usar seq_id como chave em vez de seq
                if prot_key not in unique_proteins:
                    unique_proteins[prot_key] = {
                        'seq_id': row['seq_id'],
                        'seq': row['seq'],
                        'target_kinase': row['target_kinase']
                    }
            del chunk
            gc.collect()
        pd.DataFrame(unique_ligands.values()).to_csv(os.path.join(self.base_dir, 'unique_ligands.csv'), index=False)
        pd.DataFrame(unique_proteins.values()).to_csv(os.path.join(self.base_dir, 'unique_proteins.csv'), index=False)
        self.save_checkpoint("index_files")
        print("Arquivos de índices únicos gerados com sucesso.")

    def save_ligands_parallel(self):
        if self.checkpoint_exists("ligands_saved"):
            print("Checkpoint encontrado: Ligantes já salvos.")
            return
        unique_ligands = pd.read_csv(os.path.join(self.base_dir, 'unique_ligands.csv'))
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
        unique_proteins = pd.read_csv(os.path.join(self.base_dir, 'unique_proteins.csv'))
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

