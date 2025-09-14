#!/bin/bash
# Setup script for DockTKinase environment

echo "Setting up DockTKinase environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    exit 1
fi

# Create the conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment"
    exit 1
fi

echo "Conda environment created successfully!"

# Activate the environment
echo "Activating docktkinase environment..."
eval "$(conda shell.bash hook)"
conda activate docktkinase

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate docktkinase environment"
    exit 1
fi

echo "Environment activated successfully!"

# Run the post-install script
echo "Downloading required model files..."
cd "$(dirname "$0")/.."  # Go to project root
python scripts/post_install.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to download model files"
    exit 1
fi

echo "Model files downloaded successfully!"

echo ""
echo "🎉 DockTKinase setup completed successfully!"
echo "You can now run the pipeline with: python docktkinase.py"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate docktkinase"