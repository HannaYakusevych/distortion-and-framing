#!/usr/bin/env python3
"""Main script for running distortion and framing analysis experiments."""

import argparse
from pathlib import Path

from src.data.dataset_generator import DatasetGenerator
from src.data.multitask_dataset_generator import MultitaskDatasetGenerator
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
                       default='false', help='Use compressed data with fewer classes')
    parser.add_argument('--use-scibert', choices=['true', 'false'], 
                       default='false', help='Use SciBERT instead of RoBERTa for causality model')
    parser.add_argument('--output-dir', default='out', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.task in ['generate', 'all']:
        print("Generating datasets...")
        generator = DatasetGenerator()
        generator.generate_all_datasets()
        print("✓ Single-task datasets generated")
        
        # Generate multi-task datasets
        multitask_generator = MultitaskDatasetGenerator()
        multitask_generator.generate_all_multitask_datasets()
        print("✓ Multi-task datasets generated")
    
    set_seed(42)
    
    if args.task in ['train', 'all']:
        results = {}

        use_compressed_data = args.use_compressed_data == 'true'
        use_scibert = args.use_scibert == 'true'
        
        if args.model in ['causality-base', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training causality model with {model_type}...")
            causality_trainer = CausalityBaselineTrainer( \
                use_compressed_data = use_compressed_data, \
                use_scibert = use_scibert, \
                output_dir=args.output_dir)
            results['causality'] = causality_trainer.run_training()
            print("✓ Causality model trained")
        
        if args.model in ['certainty-base', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training certainty model with {model_type}...")
            certainty_trainer = CertaintyBaselineTrainer( \
                use_compressed_data = use_compressed_data, \
                use_scibert = use_scibert, \
                output_dir=args.output_dir)
            results['certainty'] = certainty_trainer.run_training()
            print("✓ Certainty model trained")
        
        if args.model in ['generalization-base', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training generalization model with {model_type}...")
            generalization_trainer = GeneralizationBaselineTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir)
            results['generalization'] = generalization_trainer.run_training()
            print("✓ Generalization model trained")
        
        if args.model in ['sensationalism-base', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training sensationalism model with {model_type}...")
            sensationalism_trainer = SensationalismBaselineTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir)
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