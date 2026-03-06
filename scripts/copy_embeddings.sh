#!/bin/bash

# Script para copiar embeddings de proteínas (ESM-2 8M) e ligantes (MolFormer) 
# da máquina diamante-01 para a máquina diamante-03 via SSH/SCP

set -e  # Sair imediatamente se algum comando falhar

# Configurações de origem e destino
SOURCE_HOST="leon@diamante-01.cmc3.lncc.br"
SOURCE_DIR="/media/storage/leon/semantic-screening/results"

# Diretório de destino na máquina atual (diamante-03)
DEST_DIR="/storage/leon/semantic-screening/results"

echo "Iniciando cópia dos embeddings da máquina diamante-01 para diamante-03..."

# Verificar conexão com a máquina de origem
echo "Verificando conexão com $SOURCE_HOST..."
ssh -q $SOURCE_HOST exit
if [ $? -eq 0 ]; then
    echo "✓ Conexão com $SOURCE_HOST bem-sucedida"
else
    echo "✗ Falha na conexão com $SOURCE_HOST"
    exit 1
fi

# Criar diretórios de destino se não existirem
mkdir -p "$DEST_DIR"/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build
mkdir -p "$DEST_DIR"/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build

# Copiar pastas de embeddings de proteínas (human)
SOURCE_PATH_HUMAN_PROTEIN="$SOURCE_DIR/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build/protein_matrices"
if ssh $SOURCE_HOST "[ -d $SOURCE_PATH_HUMAN_PROTEIN ]"; then
    echo "Copiando embeddings de proteínas para human..."
    scp -r $SOURCE_HOST:$SOURCE_PATH_HUMAN_PROTEIN \
          "$DEST_DIR/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build/"
    echo "✓ Embeddings de proteínas (human) copiados com sucesso"
else
    echo "⚠ Aviso: Pasta de embeddings de proteínas para human não encontrada"
    echo "  Caminho esperado: $SOURCE_PATH_HUMAN_PROTEIN"
fi

# Copiar pastas de embeddings de proteínas (non_human)
SOURCE_PATH_NONHUMAN_PROTEIN="$SOURCE_DIR/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/protein_matrices"
if ssh $SOURCE_HOST "[ -d $SOURCE_PATH_NONHUMAN_PROTEIN ]"; then
    echo "Copiando embeddings de proteínas para non_human..."
    scp -r $SOURCE_HOST:$SOURCE_PATH_NONHUMAN_PROTEIN \
          "$DEST_DIR/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/"
    echo "✓ Embeddings de proteínas (non_human) copiados com sucesso"
else
    echo "⚠ Aviso: Pasta de embeddings de proteínas para non_human não encontrada"
    echo "  Caminho esperado: $SOURCE_PATH_NONHUMAN_PROTEIN"
fi

# Copiar pastas de embeddings de ligantes (MolFormer) - human
SOURCE_PATH_HUMAN_LIGAND="$SOURCE_DIR/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build/molformer_matrix"
if ssh $SOURCE_HOST "[ -d $SOURCE_PATH_HUMAN_LIGAND ]"; then
    echo "Copiando embeddings de ligantes (MolFormer) para human..."
    scp -r $SOURCE_HOST:$SOURCE_PATH_HUMAN_LIGAND \
          "$DEST_DIR/protein_model_benchmark_human_v2/esm2_t6_8M_UR50D/build/"
    echo "✓ Embeddings de ligantes (MolFormer, human) copiados com sucesso"
else
    echo "⚠ Aviso: Pasta de embeddings de ligantes (MolFormer) para human não encontrada"
    echo "  Caminho esperado: $SOURCE_PATH_HUMAN_LIGAND"
fi

# Copiar pastas de embeddings de ligantes (MolFormer) - non_human
SOURCE_PATH_NONHUMAN_LIGAND="$SOURCE_DIR/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/molformer_matrix"
if ssh $SOURCE_HOST "[ -d $SOURCE_PATH_NONHUMAN_LIGAND ]"; then
    echo "Copiando embeddings de ligantes (MolFormer) para non_human..."
    scp -r $SOURCE_HOST:$SOURCE_PATH_NONHUMAN_LIGAND \
          "$DEST_DIR/protein_model_benchmark_non_human_v2/esm2_t6_8M_UR50D/build/"
    echo "✓ Embeddings de ligantes (MolFormer, non_human) copiados com sucesso"
else
    echo "⚠ Aviso: Pasta de embeddings de ligantes (MolFormer) para non_human não encontrada"
    echo "  Caminho esperado: $SOURCE_PATH_NONHUMAN_LIGAND"
fi

echo ""
echo "Cópia concluída!"
echo ""
echo "Estrutura criada em: $DEST_DIR"
echo ""
echo "Conteúdo:"
tree -d "$DEST_DIR" 2>/dev/null || echo "(Instale 'tree' para visualizar a estrutura de diretórios)"