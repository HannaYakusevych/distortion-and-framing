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
from src.models.multitask_classifier_model_sep import MultitaskBaselineTrainer
from src.models.multitask_regression_model import MultitaskRegressionTrainer
from src.data.prepare_multitask_regression_data import prepare_multitask_regression_data
from src.utils.evaluation import EvaluationUtils

from transformers import set_seed

def main():
    parser = argparse.ArgumentParser(description='Distortion and Framing Analysis')
    parser.add_argument('--task', choices=['generate', 'train', 'all'], 
                       default='all', help='Task to run')
    parser.add_argument('--model', choices=['causality-base', 'certainty-base', 'generalization-base', 'sensationalism-base', 'multitask-base', 'multitask-regression', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--use-compressed-data', choices=['true', 'false'], 
                       default='false', help='Use compressed data with fewer classes')
    parser.add_argument('--use-scibert', choices=['true', 'false'], 
                       default='false', help='Use SciBERT instead of RoBERTa for causality model')
    parser.add_argument('--optimize-hyperparams', choices=['true', 'false'], 
                       default='false', help='Run hyperparameter optimization for multi-task model')
    parser.add_argument('--max-combinations', type=int, default=20,
                       help='Maximum hyperparameter combinations to test')
    parser.add_argument('--loss-balancing', choices=['fixed', 'adaptive', 'gradnorm'], 
                       default='fixed', help='Loss balancing strategy for multitask regression')
    parser.add_argument('--include-regression', choices=['true', 'false'], 
                       default='true', help='Include regression tasks in multitask models')
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
        
        # Generate regression datasets if needed
        if args.include_regression == 'true' or args.model in ['multitask-regression', 'all']:
            print("Preparing multi-task regression data...")
            prepare_multitask_regression_data()
            print("✓ Multi-task regression datasets prepared")
    
    set_seed(42)
    
    if args.task in ['train', 'all']:
        results = {}

        use_compressed_data = args.use_compressed_data == 'true'
        use_scibert = args.use_scibert == 'true'
        optimize_hyperparams = args.optimize_hyperparams == 'true'
        include_regression = args.include_regression == 'true'
        loss_balancing = args.loss_balancing
        
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
        
        if args.model in ['multitask-base', 'all']:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training multi-task classification model with {model_type}...")
            multitask_trainer = MultitaskBaselineTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir)
            
            if optimize_hyperparams:
                print("Running hyperparameter optimization...")
                multitask_trainer.run_hyperparameter_optimization(
                    max_combinations=args.max_combinations
                )
                results['multitask_classification'] = multitask_trainer.run_training_with_best_hyperparams()
            else:
                results['multitask_classification'] = multitask_trainer.run_training()
            print("✓ Multi-task classification model trained")
        
        if args.model in ['multitask-regression', 'all'] and include_regression:
            model_type = "SciBERT" if use_scibert else "RoBERTa"
            print(f"Training multi-task regression model with {model_type} and {loss_balancing} loss balancing...")
            
            multitask_regression_trainer = MultitaskRegressionTrainer(
                use_scibert=use_scibert,
                output_dir=args.output_dir,
                loss_balancing=loss_balancing,
                num_train_epochs=5,
                per_device_train_batch_size=8,
                early_stopping_patience=3
            )
            
            results['multitask_regression'] = multitask_regression_trainer.run_training(
                use_validation_split=True,
                validation_size=0.2
            )
            print("✓ Multi-task regression model trained")
        
        # Print summary
        if results:
            print("\n" + "="*50)
            print("RESULTS SUMMARY")
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
                    
                elif model_name == 'multitask_regression':
                    # Multi-task with regression
                    print(f"  Causality F1: {model_results['causality']['macro_f1']:.4f}")
                    print(f"  Certainty F1: {model_results['certainty']['macro_f1']:.4f}")
                    print(f"  Sensationalism Correlation: {model_results['sensationalism']['pearson_correlation']:.4f}")
                    print(f"  Sensationalism MSE: {model_results['sensationalism']['mse']:.4f}")
                    
                elif model_name == 'multitask_classification':
                    # Multi-task classification only
                    print(f"  Causality F1: {model_results['causality']['macro_f1']:.4f}")
                    print(f"  Certainty F1: {model_results['certainty']['macro_f1']:.4f}")
                    
                else:
                    # Single classification tasks
                    print(f"  Macro F1: {model_results['macro_f1']:.4f}")
                    if 'f1_per_class' in model_results:
                        print(f"  F1 per class: {[f'{f:.3f}' for f in model_results['f1_per_class']]}")
            
            # Try to create results table if evaluation utils work
            try:
                eval_utils = EvaluationUtils()
                # Filter out regression results for the table (it expects classification format)
                classification_results = {
                    k: v for k, v in results.items() 
                    if k not in ['sensationalism', 'multitask_regression'] and v is not None
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


if __name__ == '__main__':
    main()