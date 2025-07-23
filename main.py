#!/usr/bin/env python3
"""Main script for running distortion and framing analysis experiments."""

import argparse
from pathlib import Path

from src.data.dataset_generator import DatasetGenerator
from src.models.causality_base_model import CausalityBaselineTrainer
from src.models.certainty_base_model import CertaintyBaselineTrainer
from src.models.generalization_base_model import GeneralizationBaselineTrainer
from src.models.sensationalism_base_model import SensationalismBaselineTrainer
from src.utils.evaluation import EvaluationUtils

from transformers import set_seed

def main():
    parser = argparse.ArgumentParser(description='Distortion and Framing Analysis')
    parser.add_argument('--task', choices=['generate', 'train', 'all'], 
                       default='all', help='Task to run')
    parser.add_argument('--model', choices=['causality-base', 'certainty-base', 'generalization-base', 'sensationalism-base', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--use-compressed-data', choices=['true', 'false'], 
                       default='false', help='Output directory')
    parser.add_argument('--output-dir', default='out', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.task in ['generate', 'all']:
        print("Generating datasets...")
        generator = DatasetGenerator()
        generator.generate_all_datasets()
        print("✓ Datasets generated")
    
    set_seed(42)
    
    if args.task in ['train', 'all']:
        results = {}

        use_compressed_data = args.use_compressed_data == 'true'
        
        if args.model in ['causality-base', 'all']:
            print("Training causality model...")
            causality_trainer = CausalityBaselineTrainer( \
                use_compressed_data = use_compressed_data, \
                output_dir=args.output_dir)
            results['causality'] = causality_trainer.run_training()
            print("✓ Causality model trained")
        
        if args.model in ['certainty-base', 'all']:
            print("Training certainty model...")
            certainty_trainer = CertaintyBaselineTrainer( \
                use_compressed_data = use_compressed_data, \
                output_dir=args.output_dir)
            results['certainty'] = certainty_trainer.run_training()
            print("✓ Certainty model trained")
        
        if args.model in ['generalization-base', 'all']:
            print("Training generalization model...")
            generalization_trainer = GeneralizationBaselineTrainer(output_dir=args.output_dir)
            results['generalization'] = generalization_trainer.run_training()
            print("✓ Generalization model trained")
        
        if args.model in ['sensationalism-base', 'all']:
            print("Training sensationalism model...")
            sensationalism_trainer = SensationalismBaselineTrainer(output_dir=args.output_dir)
            results['sensationalism'] = sensationalism_trainer.run_training()
            print("✓ Sensationalism model trained")
        
        # Print summary
        if results:
            print("\n" + "="*50)
            print("RESULTS SUMMARY")
            print("="*50)
            
            eval_utils = EvaluationUtils()
            results_table = eval_utils.create_results_table(
                results, list(results.keys())
            )
            print(results_table.to_string(index=False))


if __name__ == '__main__':
    main()