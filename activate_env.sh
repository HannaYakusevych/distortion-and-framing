#!/bin/bash
# Script to activate the distortion-framing conda environment

echo "🔄 Activating distortion-framing environment with Python 3.10.11..."
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