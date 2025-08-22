#!/usr/bin/env python3
"""Command-line script for running baseline model experiments."""

import argparse
from pathlib import Path

from src.data.dataset_generator import DatasetGenerator
from src.models.causality_base_model import CausalityBaselineTrainer
from src.models.certainty_base_model import CertaintyBaselineTrainer
from src.models.generalization_base_model import GeneralizationBaselineTrainer
from src.models.sensationalism_base_model import SensationalismBaselineTrainer
from src.utils.evaluation import EvaluationUtils

from transformers import set_seed


def run_baseline_experiments(args):
    """Run baseline model experiments based on provided arguments."""
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Generate datasets if requested
    if args.task in ['generate', 'all']:
        print("Generating single-task datasets...")
        generator = DatasetGenerator()
        generator.generate_all_datasets()
        print("✓ Single-task datasets generated")
    
    # Train models if requested
    if args.task in ['train', 'all']:
        results = {}
        
        use_compressed_data = args.use_compressed_data == 'true'
        use_scibert = args.use_scibert == 'true'
        
        # Train causality baseline
        if args.model in ['causality', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training causality baseline with {model_type}...")
            causality_trainer = CausalityBaselineTrainer(
                use_compressed_data=use_compressed_data,
                use_scibert=use_scibert,
                output_dir=args.output_dir
            )
            results['causality'] = causality_trainer.run_training()
            print("✓ Causality baseline trained")
        
        # Train certainty baseline
        if args.model in ['certainty', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training certainty baseline with {model_type}...")
            certainty_trainer = CertaintyBaselineTrainer(
                use_compressed_data=use_compressed_data,
                use_scibert=use_scibert,
                output_dir=args.output_dir
            )
            results['certainty'] = certainty_trainer.run_training()
            print("✓ Certainty baseline trained")
        
        # Train generalization baseline
        if args.model in ['generalization', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training generalization baseline with {model_type}...")
            generalization_trainer = GeneralizationBaselineTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir
            )
            results['generalization'] = generalization_trainer.run_training()
            print("✓ Generalization baseline trained")
        
        # Train sensationalism baseline
        if args.model in ['sensationalism', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training sensationalism baseline with {model_type}...")
            sensationalism_trainer = SensationalismBaselineTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir
            )
            results['sensationalism'] = sensationalism_trainer.run_training()
            print("✓ Sensationalism baseline trained")
        
        # Print results summary
        print_results_summary(results)
        
        return results


def print_results_summary(results):
    """Print a formatted summary of training results."""
    
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "="*50)
    print("BASELINE RESULTS SUMMARY")
    print("="*50)
    
    # Print results for each model
    for model_name, model_results in results.items():
        if model_results is None:
            print(f"{model_name}: FAILED")
            continue
            
        print(f"\n{model_name.upper()}:")
        print("-" * 30)
        
        if model_name == 'sensationalism':
            # Regression task
            print(f"  Pearson Correlation: {model_results['pearson_correlation']:.4f}")
            print(f"  P-value: {model_results['p_value']:.4f}")
            print(f"  MSE: {model_results['mse']:.4f}")
        else:
            # Classification tasks
            print(f"  Macro F1: {model_results['macro_f1']:.4f}")
            if 'f1_per_class' in model_results:
                print(f"  F1 per class: {[f'{f:.3f}' for f in model_results['f1_per_class']]}")
    
    # Create results table for classification tasks
    try:
        eval_utils = EvaluationUtils()
        classification_results = {
            k: v for k, v in results.items() 
            if k != 'sensationalism' and v is not None
        }
        if classification_results:
            print(f"\nCLASSIFICATION RESULTS TABLE:")
            print("-" * 40)
            results_table = eval_utils.create_results_table(
                classification_results, list(classification_results.keys())
            )
            print(results_table.to_string(index=False))
    except Exception as e:
        print(f"\nNote: Could not create results table: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run baseline model experiments')
    
    parser.add_argument('--task', choices=['generate', 'train', 'all'], 
                       default='all', help='Task to run (default: all)')
    
    parser.add_argument('--model', 
                       choices=['causality', 'certainty', 'generalization', 'sensationalism', 'all'],
                       default='all', help='Which baseline model to train (default: all)')
    
    parser.add_argument('--use-compressed-data', choices=['true', 'false'], 
                       default='false', help='Use compressed data with fewer classes (default: false)')
    
    parser.add_argument('--use-scibert', choices=['true', 'false'], 
                       default='false', help='Use SciBERT instead of RoBERTa (default: false)')
    
    parser.add_argument('--output-dir', default='out', 
                       help='Output directory for results (default: out)')
    
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Run the experiments
    results = run_baseline_experiments(args)
    
    print(f"\n✓ Baseline experiments completed. Results saved to '{args.output_dir}'")


if __name__ == '__main__':
    main()