import os
import torch
import esm
import numpy as np
from tqdm import tqdm

class EmbeddingMeta:
    def __init__(self, model_name="esm2_t33_650M_UR50D", seq_input_dir="./seq_inputs", output_dir="./output_esm"):
        self.model_name = model_name
        self.seq_input_dir = seq_input_dir
        self.output_dir = output_dir
        print(f"EmbeddingMeta inicializado com seq_input_dir: {self.seq_input_dir} e output_dir: {self.output_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        # Cria o diretório de saída, se necessário
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        """Carrega o modelo ESM pré-treinado com verificação de nome válido."""
        print(f"Carregando o modelo {self.model_name}...")
        try:
            # Tentar carregar o modelo com o nome especificado
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        except Exception as e:
            print(f"Erro ao carregar modelo '{self.model_name}': {e}")
            print("Tentando modelo alternativo...")
            # Tentar modelo alternativo
            try:
                alternative_model = "esm2_t33_650M_UR50D"
                print(f"Tentando carregar modelo alternativo: {alternative_model}")
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(alternative_model)
                self.model_name = alternative_model
            except Exception as e2:
                raise ValueError(f"Falha ao carregar modelo '{self.model_name}' e modelo alternativo. "
                                 f"Erro: {e2}")
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print("Modelo carregado com sucesso.")

    def read_sequences(self):
        """Lê as sequências da pasta seq_inputs."""
        print(f"Procurando sequências na pasta: {self.seq_input_dir}")
        if not os.path.exists(self.seq_input_dir):
            raise FileNotFoundError(f"A pasta '{self.seq_input_dir}' não foi encontrada.")

        sequences = []
        files = os.listdir(self.seq_input_dir)
        print(f"Arquivos encontrados na pasta: {len(files)}")
        for filename in files:
            if filename.endswith(".fasta") or filename.endswith(".txt"):
                filepath = os.path.join(self.seq_input_dir, filename)
                with open(filepath, "r") as file:
                    lines = file.readlines()
                    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
                    sequences.append((os.path.splitext(filename)[0], sequence))
        if not sequences:
            raise ValueError(f"Nenhuma sequência válida encontrada na pasta '{self.seq_input_dir}'.")
        print(f"{len(sequences)} sequência(s) carregada(s) para processamento.")
        return sequences

    def extract_latent_embedding(self, name, sequence):
        """Extrai e salva o embedding latente para uma única sequência."""
        try:
            output_path = os.path.join(self.output_dir, f"{name}_embedding.npy")
            if os.path.exists(output_path):
                print(f"Embedding já existe. Pulando: {output_path}")
                return

            data = [(name, sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers], return_contacts=False)
                embedding = results["representations"][self.model.num_layers][0]

            np.save(output_path, embedding.cpu().numpy())
            print(f"Embedding salvo com sucesso: {output_path}")
        except Exception as e:
            print(f"Erro ao processar sequência '{name}': {e}")

    def run(self):
        """Executa o pipeline completo."""
        self.load_model()
        sequences = self.read_sequences()
        for name, sequence in tqdm(sequences, desc="Processando sequências"):
            self.extract_latent_embedding(name, sequence)
        print("Processamento concluído. Todos os embeddings foram salvos.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python script.py <pasta_sequencias>")
        sys.exit(1)

    seq_input_dir = sys.argv[1]
    output_dir = f"{seq_input_dir.rstrip('/')}_embedding"

    extractor = EmbeddingMeta(seq_input_dir=seq_input_dir, output_dir=output_dir)
    extractor.run()
