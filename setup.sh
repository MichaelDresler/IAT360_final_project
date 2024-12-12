#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Activate Conda
echo "Activating Conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment.yaml exists
if [ ! -f "environment.yaml" ]; then
    echo "Error: environment.yaml file not found in the current directory."
    exit 1
fi

# Create the Conda environment
echo "Creating Conda environment from environment.yaml..."
conda env create -f environment.yaml

# Optional: Activate the new environment
ENV_NAME=$(head -n 1 environment.yaml | awk '{print $2}')
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "Conda environment setup is complete."
