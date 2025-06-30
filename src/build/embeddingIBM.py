import os
import sys
import psutil
import numpy as np
import pandas as pd
import models.fm4m as fm4m
from concurrent.futures import ThreadPoolExecutor

class EmbeddingIBM:
    def __init__(self, model_type="SMI-TED", batch_size=None, checkpoint_file="processed_files.log"):
        """
        Inicializa a classe EmbeddingIBM com o tipo de modelo especificado.
        
        :param model_type: Nome do modelo a ser usado (padrão: "SMI-TED").
        :param batch_size: Número de SMILES processados por vez.
        :param checkpoint_file: Nome do arquivo de log para checkpoint.
        """
        self.model_type = model_type
        self.batch_size = batch_size or self._calculate_optimal_batch_size()
        self.checkpoint_file = checkpoint_file
        self.processed_files = self._load_checkpoint()

    def _calculate_optimal_batch_size(self):
        """
        Calcula o tamanho ótimo do batch com base na memória disponível.
        
        :return: Tamanho do batch calculado dinamicamente.
        """
        total_memory = psutil.virtual_memory().total
        usable_memory = total_memory * 0.75
        memory_per_smiles = 1 * 1024 * 1024  # 1 MB por SMILES (ajuste experimental)
        batch_size = int(usable_memory / memory_per_smiles)
        return max(10, batch_size)

    def _load_checkpoint(self):
        """
        Carrega os arquivos processados do arquivo de checkpoint.
        
        :return: Conjunto de arquivos processados.
        """
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return set(line.strip() for line in f)
        return set()

    def _update_checkpoint(self, file_name):
        """
        Atualiza o arquivo de checkpoint com o nome do arquivo concluído.
        
        :param file_name: Nome do arquivo concluído.
        """
        with open(self.checkpoint_file, "a") as f:
            f.write(f"{file_name}\n")

    def get_latent_representation(self, smiles_list):
        """
        Gera a representação latente para uma lista de SMILES.
        
        :param smiles_list: Lista de strings SMILES.
        :return: DataFrame contendo as representações latentes.
        """
        representations, _ = fm4m.get_representation(
            train_data=smiles_list,
            test_data=smiles_list,  # Reutilizar train_data para evitar test_data vazio
            model_type=self.model_type,
            return_tensor=False
        )
        return representations

    def process_file(self, file_path, output_dir):
        """
        Processa um único arquivo de SMILES e salva as representações latentes no diretório de saída.
        
        :param file_path: Caminho do arquivo .smi.
        :param output_dir: Caminho para o diretório de saída.
        """
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.npy")

        if file_name in self.processed_files:
            print(f"{file_name} já processado. Pulando.")
            return

        with open(file_path, "r") as file:
            smiles_list = [line.strip() for line in file]

        if not smiles_list:
            print(f"Aviso: O arquivo '{file_name}' está vazio. Pulando.")
            return

        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            representations = self.get_latent_representation(batch)
            np.save(output_file, representations.values)

        self._update_checkpoint(file_name)

    def process_smiles_folder(self, input_dir, output_dir):
        """
        Processa todos os arquivos .smi no diretório de entrada em paralelo.
        
        :param input_dir: Caminho para o diretório contendo os arquivos .smi.
        :param output_dir: Caminho para o diretório onde as representações serão salvas.
        """
        os.makedirs(output_dir, exist_ok=True)
        smi_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".smi")]

        if not smi_files:
            print("Nenhum arquivo .smi encontrado no diretório de entrada.")
            return

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for file_path in smi_files:
                executor.submit(self.process_file, file_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py /pasta_input/")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = os.path.join(input_dir, "ligand_embeddings")

    if not os.path.exists(input_dir):
        print(f"Erro: O diretório de entrada '{input_dir}' não existe.")
        sys.exit(1)

    embedding_ibm = EmbeddingIBM()
    embedding_ibm.process_smiles_folder(input_dir, output_dir)
