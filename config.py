"""Configuration settings for the project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "out"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configurations
MODEL_CONFIGS = {
    'roberta-base': {
        'max_length': 1536,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'weight_decay': 0.01,
        'warmup_steps': 200,
    },
    'scibert': {
        'model_name': 'allenai/scibert_scivocab_uncased',
        'max_length': 1536,
        'batch_size': 8,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'weight_decay': 0.01,
        'warmup_steps': 200,
    }
}

# Task configurations
TASK_CONFIGS = {
    'causality': {
        'labels': {
            0: "Explicitly states: no relation",
            1: "Causation", 
            2: "Correlation",
            3: "No mention of a relation"
        },
        'train_file': 'causality_train.csv',
        'test_file': 'causality_test.csv',
    },
    'certainty': {
        'labels': {
            0: "Certain",
            1: "Somewhat certain",
            2: "Somewhat uncertain", 
            3: "Uncertain"
        },
        'train_file': 'certainty_train.csv',
        'test_file': 'certainty_test.csv',
    },
    'generalization': {
        'labels': {
            0: "Paper Finding",
            1: "They are at the same level of generality",
            2: "Reported Finding"
        },
        'train_file': 'generalization_train.csv',
        'test_file': 'generalization_test.csv',
    },
    'sensationalism': {
        'train_file': 'sensationalism_train.csv',
        'test_file': 'sensationalism_test.csv',
    }
}

# Multitask configurations
MULTITASK_CONFIGS = {
    'loss_weights': {
        'causality': 0.7,
        'certainty': 0.3,
    },
    'shared_layers': True,
    'task_specific_heads': True,
}