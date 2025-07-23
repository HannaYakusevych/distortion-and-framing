# Distortion and Framing Analysis

A research project analyzing distortion and framing in scientific communication using transformer-based models for multiple classification tasks including causality, certainty, generalization, and sensationalism detection.

## Project Structure

```
distortion-and-framing/
├── src/                          # Main source code
│   ├── data/                     # Data processing modules
│   │   ├── dataset_generator.py  # Dataset generation utilities
│   │   └── __init__.py
│   ├── models/                   # Model implementations
│   │   ├── base_trainer.py       # Base training class
│   │   ├── causality_model.py    # Causality classification
│   │   ├── certainty_model.py    # Certainty classification
│   │   └── __init__.py
│   ├── utils/                    # Utility functions
│   │   ├── evaluation.py         # Evaluation metrics and visualization
│   │   └── __init__.py
│   └── __init__.py
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb # Data exploration and visualization
│   ├── 02_baseline_models.ipynb  # Baseline model training
│   └── 03_multitask_experiments.ipynb # Multitask learning experiments
├── data/                         # Raw data files
├── temp/                         # Generated datasets
├── out/                          # Model outputs and checkpoints
├── main.py                       # Main execution script
├── config.py                     # Configuration settings
└── requirements.txt              # Python dependencies
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Generate datasets and train all models:
```bash
python main.py --task all
```

#### Generate datasets only:
```bash
python main.py --task generate
```

#### Train specific model:
```bash
python main.py --task train --model causality
python main.py --task train --model certainty
```

#### Using the modules programmatically:
```python
from src.data.dataset_generator import DatasetGenerator
from src.models.causality_model import CausalityTrainer

# Generate datasets
generator = DatasetGenerator()
generator.generate_all_datasets()

# Train causality model
trainer = CausalityTrainer()
results = trainer.run_training()
```

## Tasks and Models

### 1. Causality Classification
- **Classes**: No relation, Causation, Correlation, No mention
- **Model**: RoBERTa-base fine-tuned for sequence classification
- **Evaluation**: F1-score per class and macro F1

### 2. Certainty Classification  
- **Classes**: Certain, Somewhat certain, Somewhat uncertain, Uncertain
- **Model**: RoBERTa-base fine-tuned for sequence classification
- **Evaluation**: F1-score per class and macro F1

### 3. Generalization Classification
- **Classes**: Paper finding more general, Same level, Reported finding more general
- **Model**: RoBERTa-base with paired input processing

### 4. Sensationalism Detection
- **Task**: Regression/ranking task for sensationalism scoring
- **Model**: RoBERTa-base adapted for scoring

## Notebooks

### 01_data_exploration.ipynb
- Explore the SPICED dataset structure
- Analyze label distributions
- Visualize data characteristics

### 02_baseline_models.ipynb
- Train baseline RoBERTa models for each task
- Compare performance across tasks
- Generate evaluation reports and visualizations

### 03_multitask_experiments.ipynb
- Implement multitask learning approaches
- Experiment with different loss weighting strategies
- Compare single-task vs multitask performance

## Results

### Baseline Performance Comparison

#### RoBERTa vs SciBERT Performance Summary
| Model | Data | Causality mF1 | Certainty mF1 | Generalization mF1 | Sensationalism r |
|-------|------|---------------|---------------|-------------------|------------------|
| RoBERTa | Full (4-class) | 0.539 | 0.398 | 0.502 | 0.614 |
| RoBERTa | Compressed (3-class) | 0.568 | 0.554 | 0.438 | 0.619 |
| SciBERT | Full (4-class) | 0.538 | 0.437 | 0.489 | 0.610 |
| SciBERT | Compressed (3-class) | 0.563 | 0.542 | 0.489 | 0.615 |

### Detailed Results by Task

#### Causality Classification

**RoBERTa - Full Data (4 classes)**
| Class | F1-Score |
|-------|----------|
| Explicitly states: no relation | 0.455 | 
| Causation | 0.557 |
| Correlation | 0.612 |
| No mention of a relation | 0.534 |
| **Macro F1** | **0.539** |

**RoBERTa - Compressed Data (3 classes)**
| Class | F1-Score | F1-Score - from paper |
|-------|----------| --------------------- |
| Unclear | 0.563 | 0.59 |
| Causation | 0.559 | 0.56 |
| Correlation | 0.582 | 0.62 |
| **Macro F1** | **0.568** | **0.57+-0.01** |

**SciBERT - Full Data (4 classes)**
| Class | F1-Score |
|-------|----------|
| Explicitly states: no relation | 0.476 |
| Causation | 0.561 |
| Correlation | 0.608 |
| No mention of a relation | 0.506 |
| **Macro F1** | **0.538** |

**SciBERT - Compressed Data (3 classes)**
| Class | F1-Score | F1-Score - from paper |
|-------|----------| --------------------- |
| Unclear | 0.570 | 0.6 |
| Causation | 0.556 | 0.58 |
| Correlation | 0.563 | 0.57 |
| **Macro F1** | **0.563** | **0.54+-0.04** |

#### Certainty Classification

**RoBERTa - Full Data (4 classes)**
| Class | F1-Score |
|-------|----------|
| Certain | 0.692 |
| Somewhat certain | 0.506 |
| Somewhat uncertain | 0.393 |
| Uncertain | 0.000 |
| **Macro F1** | **0.398** |

**RoBERTa - Compressed Data (3 classes)**
| Class | F1-Score | F1-Score - from paper |
|-------|----------| --------------------- |
| Certain | 0.664 | 0.67 |
| Somewhat certain | 0.462 | 0.48 |
| Uncertain | 0.535 | 0.51 |
| **Macro F1** | **0.554** | **0.59+-0.04** |

**SciBERT - Full Data (4 classes)**
| Class | F1-Score |
|-------|----------|
| Certain | 0.667 |
| Somewhat certain | 0.453 |
| Somewhat uncertain | 0.343 |
| Uncertain | 0.286 |
| **Macro F1** | **0.437** |

**SciBERT - Compressed Data (3 classes)**
| Class | F1-Score | F1-Score - from paper |
|-------|----------| --------------------- |
| Certain | 0.664 | 0.7 |
| Somewhat certain | 0.472 | 0.5 |
| Uncertain | 0.490 | 0.5 |
| **Macro F1** | **0.542** | **0.53+-0.02** |

#### Generalization Classification

**RoBERTa**
| Class | F1-Score |  F1-Score - from paper |
|-------|----------| --------------------- |
| Paper Finding | 0.517 | 0.43 |
| Same level of generality | 0.250 | 0.32 |
| Reported Finding | 0.740 | 0.69 |
| **Macro F1** | **0.502** | **0.4+-0.06** |

**SciBERT**
| Class | F1-Score |  F1-Score - from paper |
|-------|----------| --------------------- |
| Paper Finding | 0.410 | 0.49 |
| Same level of generality | 0.366 | 0.32 |
| Reported Finding | 0.692 | 0.72 |
| **Macro F1** | **0.489** | **0.47+-0.04** |

#### Sensationalism Regression

| Model | Data | Pearson r | P-value | MSE |
|-------|------|-----------|---------|-----|
| RoBERTa | - | 0.614 | 6.39e-36 | 0.056 |
| RoBERTa | From paper | 0.61+-0.02 | - | - |
| SciBERT | - | 0.610 | 1.75e-35 | 0.056 |
| SciBERT | From paper | 0.57+-0.03 | - | - |

## Configuration

The `config.py` file contains all configuration settings:
- Model hyperparameters
- Task-specific settings
- File paths and directories
- Multitask learning configurations


