#!/usr/bin/env python3
"""Script to run comprehensive experiments."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_generator import DatasetGenerator
from src.models.causality_base_model import CausalityBaselineTrainer
from src.models.certainty_base_model import CertaintyBaselineTrainer
from src.models.generalization_base_model import GeneralizationBaselineTrainer
from src.models.sensationalism_base_model import SensationalismBaselineTrainer
from src.utils.evaluation import EvaluationUtils
import json

from transformers import set_seed

def run_baseline_experiments(use_compressed_data: bool = False, use_scibert: bool = False):
    """Run all baseline experiments and save results."""
    print("Starting baseline experiments...")
    
    # Generate datasets
    generator = DatasetGenerator()
    generator.generate_all_datasets()
    
    # Train models
    results = {}
    
    # Causality
    model_type = "SciBERT" if use_scibert else "RoBERTa"
    print(f"Training causality model with {model_type}...")
    causality_trainer = CausalityBaselineTrainer(
        use_compressed_data=use_compressed_data,
        use_scibert=use_scibert
    )
    results['causality'] = causality_trainer.run_training()
    
    # Certainty  
    print(f"Training certainty model with {model_type}...")
    certainty_trainer = CertaintyBaselineTrainer(
        use_compressed_data=use_compressed_data,
        use_scibert=use_scibert
    )
    results['certainty'] = certainty_trainer.run_training()
    
    # Generalization
    print(f"Training generalization model with {model_type}...")
    generalization_trainer = GeneralizationBaselineTrainer(use_scibert=use_scibert)
    results['generalization'] = generalization_trainer.run_training()
    
    # Sensationalism
    print(f"Training sensationalism model with {model_type}...")
    sensationalism_trainer = SensationalismBaselineTrainer(use_scibert=use_scibert)
    results['sensationalism'] = sensationalism_trainer.run_training()
    
    # Save results
    model_suffix = "scibert" if use_scibert else "roberta"
    output_file = Path(f'out/baseline_results_{use_compressed_data}_{model_suffix}.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for task, task_results in results.items():
        if task == 'sensationalism':
            # Sensationalism is regression, different metrics
            json_results[task] = {
                'pearson_correlation': float(task_results['pearson_correlation']),
                'p_value': float(task_results['p_value']),
                'mse': float(task_results['mse'])
            }
        else:
            # Classification tasks
            json_results[task] = {
                'f1_per_class': task_results['f1_per_class'].tolist(),
                'macro_f1': float(task_results['macro_f1'])
            }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results


if __name__ == '__main__':
    set_seed(42)

    # Run experiments with RoBERTa
    print("="*60)
    print("RUNNING ROBERTA EXPERIMENTS")
    print("="*60)
    run_baseline_experiments(use_compressed_data=False, use_scibert=False)
    run_baseline_experiments(use_compressed_data=True, use_scibert=False)
    
    # Run experiments with SciBERT
    print("\n" + "="*60)
    print("RUNNING SCIBERT EXPERIMENTS")
    print("="*60)
    run_baseline_experiments(use_compressed_data=False, use_scibert=True)
    run_baseline_experiments(use_compressed_data=True, use_scibert=True)