import pandas as pd
import os

# Ler o arquivo TSV
df = pd.read_csv('${PROJECT_ROOT}/src/database/kinase_non_human_compounds.tsv', sep='\t')

# Obter seq_ids únicos do TSV
tsv_seq_ids = set(df['seq_id'].astype(str))

# Obter seq_ids dos arquivos .fasta na pasta protein
protein_files = os.listdir('${PROJECT_ROOT}/non_humans/protein/')
fasta_seq_ids = set()
for file in protein_files:
    if file.endswith('.fasta'):
        seq_id = file.replace('_protein.fasta', '')
        fasta_seq_ids.add(seq_id)

# Encontrar seq_ids que estão no TSV mas não têm arquivos .fasta
missing_seq_ids = tsv_seq_ids - fasta_seq_ids

print(f"Seq_ids no TSV: {len(tsv_seq_ids)}")
print(f"Seq_ids com arquivos .fasta: {len(fasta_seq_ids)}")
print(f"Seq_ids faltando: {len(missing_seq_ids)}")
print("\nSeq_ids faltando:")
for seq_id in sorted(missing_seq_ids, key=int):
    print(seq_id)