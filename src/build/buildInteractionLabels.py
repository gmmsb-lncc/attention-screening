import os
import numpy as np
from pyspark.sql import SparkSession

class InteractionLabels:
    def __init__(self, original_tsv_path, output_dir="concatenated_embeddings"):
        self.original_tsv_path = original_tsv_path
        self.output_dir = output_dir
        self.labels_path = os.path.join(self.output_dir, "interaction_labels.npy")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.spark = SparkSession.builder \
            .appName("InteractionLabelsProcessing") \
            .master("local[*]") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
    
    def extract_labels(self):
        """Extrai informações do TSV e salva os rótulos em um arquivo .npy utilizando Spark."""
        df = self.spark.read.csv(self.original_tsv_path, sep='\t', header=True, inferSchema=True)
        df_selected = df.select("molregno", "target_kinase", "standard_type", "standard_value")
        
        labels = df_selected.collect()
        labels_np = np.array(labels)
        np.save(self.labels_path, labels_np)
        
        print(f"\n✅ Arquivo de rótulos salvo em: {self.labels_path}")
    
    def stop_spark(self):
        """Encerra a sessão Spark."""
        self.spark.stop()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso correto: python buildInteractionLabels.py <input_tsv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    label_extractor = InteractionLabels(input_file)
    label_extractor.extract_labels()
    label_extractor.stop_spark()