#!/usr/bin/env python3
"""Script to run comprehensive experiments."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_generator import DatasetGenerator
from src.models.causality_base_model import CausalityBaselineTrainer
from src.models.certainty_base_model import CertaintyBaselineTrainer
from src.models.generalization_base_model import GeneralizationBaselineTrainer
from src.utils.evaluation import EvaluationUtils
import json

from transformers import set_seed

def run_baseline_experiments(use_compressed_data: bool = False):
    """Run all baseline experiments and save results."""
    print("Starting baseline experiments...")
    
    # Generate datasets
    generator = DatasetGenerator()
    generator.generate_all_datasets()
    
    # Train models
    results = {}
    
    # Causality
    causality_trainer = CausalityBaselineTrainer(use_compressed_data = use_compressed_data)
    results['causality'] = causality_trainer.run_training()
    
    # Certainty  
    certainty_trainer = CertaintyBaselineTrainer(use_compressed_data = use_compressed_data)
    results['certainty'] = certainty_trainer.run_training()
    
    # Generalization
    generalization_trainer = GeneralizationBaselineTrainer()
    results['generalization'] = generalization_trainer.run_training()
    
    # Save results
    output_file = Path(f'out/baseline_results_{use_compressed_data}.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for task, task_results in results.items():
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

    run_baseline_experiments(use_compressed_data=False)
    run_baseline_experiments(use_compressed_data=True)