#!/bin/bash
# Script to install all required dependencies for the build pipeline

echo "🚀 Installing dependencies for DockTKinase Build Pipeline"
echo "=========================================================="

# Activate virtual environment
source env/bin/activate

echo ""
echo "📦 Installing core dependencies..."
pip install --upgrade pip

echo ""
echo "🧬 Installing fair-esm (for protein embeddings)..."
pip install fair-esm

echo ""
echo "🔬 Installing transformers (for ESM models)..."
pip install transformers>=4.38

echo ""
echo "📊 Installing FM4M dependencies..."
cd llm/FM4M
pip install -r requirements.txt
cd ../..

echo ""
echo "🎯 Installing additional build dependencies..."
pip install scikit-learn>=1.5.0
pip install numpy>=1.26.1
pip install pandas>=1.5.3
pip install tqdm>=4.66.4

echo ""
echo "✅ Installation completed!"
echo ""
echo "To verify, run:"
echo "  python -c 'import fair_esm; import transformers; print(\"✅ All libraries installed!\")'"
