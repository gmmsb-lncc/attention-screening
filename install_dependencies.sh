#!/bin/bash
# Script para instalar todas as dependências necessárias para o pipeline build

echo "🚀 Instalando dependências para DockTKinase Build Pipeline"
echo "=========================================================="

# Ativar ambiente virtual
source env/bin/activate

echo ""
echo "📦 Instalando dependências core..."
pip install --upgrade pip

echo ""
echo "🧬 Instalando fair-esm (para embeddings de proteínas)..."
pip install fair-esm

echo ""
echo "🔬 Instalando transformers (para modelos ESM)..."
pip install transformers>=4.38

echo ""
echo "📊 Instalando dependências do FM4M..."
cd FM4M
pip install -r requirements.txt
cd ..

echo ""
echo "🎯 Instalando dependências adicionais para build..."
pip install scikit-learn>=1.5.0
pip install numpy>=1.26.1
pip install pandas>=1.5.3
pip install tqdm>=4.66.4

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "Para verificar, execute:"
echo "  python -c 'import fair_esm; import transformers; print(\"✅ Todas as bibliotecas instaladas!\")'"
