# Distortion and Framing Analysis

A research project analyzing distortion and framing in scientific communication using transformer-based models for multiple classification and regression tasks.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or use the provided activation script
source activate_env.sh

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Run All Experiments
```bash
python run_all_experiments.py
```

#### Run Individual Components
```bash
# Single-task baseline models
python run_baselines.py --task all --model all

# Multi-task models
python run_multitask.py --task all --model all

# Quick test
python scripts/run_experiments.py quick
```

## Available Tasks

### Single-Task Models
- **Causality**: Classification (3-4 classes)
- **Certainty**: Classification (3-4 classes) 
- **Generalization**: Classification (2-3 classes)
- **Sensationalism**: Regression (continuous)

### Multi-Task Models
- **Multi-task Classification**: Causality + Certainty
- **Multi-task Regression**: All tasks combined

## Model Options

- **Base Models**: RoBERTa (default), SciBERT
- **Data**: Full dataset or compressed (fewer classes)
- **Loss Balancing**: Fixed, adaptive, gradnorm (multi-task only)

## Usage Examples

```bash
# Train specific model
python run_baselines.py --task train --model causality

# Use SciBERT with compressed data
python run_baselines.py --use-scibert true --use-compressed-data true

# Multi-task with adaptive loss balancing
python run_multitask.py --loss-balancing adaptive

# Hyperparameter optimization
python run_multitask.py --optimize-hyperparams true --max-combinations 30
```

## Configuration

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--task` | generate, train, all | all | What to run |
| `--model` | varies by script | all | Which model(s) to train |
| `--use-scibert` | true, false | false | Use SciBERT instead of RoBERTa |
| `--use-compressed-data` | true, false | false | Use reduced class dataset |
| `--loss-balancing` | fixed, adaptive, gradnorm | fixed | Loss balancing strategy |
| `--output-dir` | path | out | Output directory |

## Performance

- **Runtime**: 10-30 min (quick test) to 6-12 hours (full experiments)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: 4GB minimum, 8GB recommended

## Output

Results are saved to the output directory:
- Model checkpoints: `out/baseline_*_*/`
- Training logs: `out/logs/`
- Results summary printed to console

## Troubleshooting

- **Missing data**: Run `--task generate` first
- **Import errors**: Check `pip install -r requirements.txt`


