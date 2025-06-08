#!/bin/bash
# run_in_env.sh
# This script ensures the correct Conda environment is activated before running any command.
# Usage: ./run_in_env.sh <command> [args...]

set -e  # Exit immediately if a command fails
set -o pipefail  # Ensure pipeline errors are also caught

# Name of the environment (should match create_conda.sh)
ENV_NAME="milvus_env"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Initialize conda for the shell if not already done
if ! grep -q "conda initialize" ~/.bashrc; then
    conda init bash
fi

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Error: Environment '$ENV_NAME' not found. Please run create_conda.sh first."
    exit 1
fi

# Check if a command was provided
if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo "Usage: ./run_in_env.sh <command> [args...]"
    exit 1
fi

# Run the provided command in the environment
echo "Running command in $ENV_NAME environment: $@"
exec "$@" 