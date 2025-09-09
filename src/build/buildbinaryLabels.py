import os
import numpy as np
import pandas as pd

class BinaryLabelGenerator:
    def __init__(self, labels_path, output_dir="concatenated_embeddings"):
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.binary_labels_path = os.path.join(self.output_dir, "binary_labels.npy")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_binary_labels(self):
        """Carrega interaction_labels.npy e gera rótulos binários baseados no threshold de 1000 nM (1 µM)."""
        labels = np.load(self.labels_path, allow_pickle=True)
        
        # Pega apenas a coluna do standard_value (índice 3) e converte para float
        standard_values = labels[:, 3].astype(float)
        
        # Aplica a regra de threshold para definir os rótulos binários
        binary_labels = np.where(standard_values <= 1000, 1, 0)
        
        # Salva os rótulos binários
        np.save(self.binary_labels_path, binary_labels)
        print(f"\n✅ Arquivo de rótulos binários salvo em: {self.binary_labels_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso correto: python buildbinaryLabels.py <interaction_labels_path>")
        sys.exit(1)
    
    labels_path = sys.argv[1]
    generator = BinaryLabelGenerator(labels_path)
    generator.generate_binary_labels()