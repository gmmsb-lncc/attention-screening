#!/bin/bash
# Post-install script for DockTKinase conda environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Running DockTKinase post-install setup..."

# Activate the environment
source $(dirname $(dirname $(which conda)))/bin/activate docktkinase

# Run the Python post-install script
cd "$PROJECT_DIR"
python scripts/post_install.py

# Check the exit code
if [ $? -eq 0 ]; then
    echo "Post-install setup completed successfully!"
else
    echo "Post-install setup failed!"
    exit 1
fi