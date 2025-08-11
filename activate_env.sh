#!/bin/bash
# Script to activate the distortion-framing conda environment
# 
# USAGE: source activate_env.sh
# (Don't use 'bash activate_env.sh' - that won't persist the environment)

echo "🔄 Activating distortion-framing environment with Python 3.10.11..."

# Initialize conda if not already done
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first."
    exit 1
fi

# Source conda and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate distortion-framing

echo "✅ Environment activated!"
echo "Python version: $(python --version)"
echo "📍 Current environment: $CONDA_DEFAULT_ENV"

# Test key imports
python -c "
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
print('✅ All key libraries imported successfully!')
print(f'📦 PyTorch version: {torch.__version__}')
print(f'📦 Transformers version: {transformers.__version__}')
"

echo ""
echo "🚀 Ready to run your data science project!"
echo "💡 Usage examples:"
echo "   python main.py --task all"
echo "   jupyter notebook"
echo "   python scripts/run_experiments.py"