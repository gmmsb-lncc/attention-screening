#!/bin/bash
# Script to activate DockTKinase environment

echo "🚀 Activating DockTKinase environment..."

# Activate virtual environment
source env/bin/activate

# Add src to PYTHONPATH
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "✅ Environment activated!"
echo "📁 PYTHONPATH: $PYTHONPATH"
echo ""
echo "To use DockTKinase:"
echo "  python src/classifier/modular_classifier.py --help  # CLI interface"
echo "  python -c 'from classifier.modular_pipeline import ModularMLPPipeline; print("System ready!")'"
echo "  jupyter lab  # For notebooks"
echo ""
