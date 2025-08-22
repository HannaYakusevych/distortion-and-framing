#!/bin/bash

# Script to run wrong predictions analysis for all models
# This script analyzes wrong predictions from trained models and saves them to CSV files

echo "=== Wrong Predictions Analysis ==="
echo "This script will analyze wrong predictions from all trained models"
echo "Results will be saved to the 'wrong_predictions' directory"
echo ""

# Check if models exist
if [ ! -d "out" ]; then
    echo "Error: 'out' directory not found. Please run training first."
    exit 1
fi

# Check if test data exists
if [ ! -d "temp" ]; then
    echo "Error: 'temp' directory not found. Please generate datasets first."
    exit 1
fi

# Create results directory
mkdir -p wrong_predictions

echo "Starting analysis for all models..."
echo ""

# Run analysis for RoBERTa models
echo "=== Analyzing RoBERTa models ==="
python scripts/analyze_wrong_predictions.py --model all

echo ""
echo "=== Analyzing SciBERT models (if available) ==="
python scripts/analyze_wrong_predictions.py --model all --use-scibert

echo ""
echo "=== Analysis Complete ==="
echo "Results saved to: wrong_predictions/"
echo ""
echo "Generated files:"
ls -la wrong_predictions/

echo ""
echo "You can also run analysis for specific models:"
echo "  python scripts/analyze_wrong_predictions.py --model causality"
echo "  python scripts/analyze_wrong_predictions.py --model certainty"
echo "  python scripts/analyze_wrong_predictions.py --model sensationalism"
echo "  python scripts/analyze_wrong_predictions.py --model multitask-classification"
echo "  python scripts/analyze_wrong_predictions.py --model multitask-regression"
