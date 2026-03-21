#!/bin/bash
# Script para ativar ambiente DockTKinase

echo "🚀 Ativando ambiente DockTKinase..."

# Ativar ambiente virtual
source env/bin/activate

# Adicionar src ao PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "✅ Ambiente ativado!"
echo "📁 PYTHONPATH: $PYTHONPATH"
echo ""
echo "Para usar DockTKinase:"
echo "  python src/classifier/modular_classifier.py --help  # CLI interface"
echo "  python -c 'from classifier.modular_pipeline import ModularMLPPipeline; print("Sistema pronto!")'"
echo "  jupyter lab  # Para notebooks"
echo ""
