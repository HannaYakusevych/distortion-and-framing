#!/usr/bin/env python3
"""Script to run comprehensive experiments."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

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
import json
import time
import traceback

from transformers import set_seed

def run_hyperparameter_optimization(use_scibert: bool = False, max_combinations: int = 20, 
                                   model_type: str = "classification"):
    """Run hyperparameter optimization for multi-task models.
    
    Args:
        use_scibert: Whether to use SciBERT instead of RoBERTa
        max_combinations: Maximum number of hyperparameter combinations to test
        model_type: "classification" for causality+certainty, "regression" for all three tasks
    """
    print(f"Starting hyperparameter optimization for {model_type} model...")
    
    # Generate multi-task datasets
    multitask_generator = MultitaskDatasetGenerator()
    multitask_generator.generate_all_multitask_datasets()
    
    if model_type == "regression":
        # Also prepare regression data
        print("Preparing multi-task regression data...")
        prepare_multitask_regression_data()
    
    # Run hyperparameter optimization
    base_model = "SciBERT" if use_scibert else "RoBERTa"
    print(f"Optimizing hyperparameters for {model_type} multi-task model with {base_model}...")
    print(f"Testing up to {max_combinations} hyperparameter combinations...")
    
    try:
        if model_type == "regression":
            trainer = MultitaskRegressionTrainer(
                use_scibert=use_scibert,
                loss_balancing="fixed"  # Use fixed balancing for hyperparameter search
            )
            optimization_results = trainer.run_hyperparameter_optimization(max_combinations=max_combinations)
            optimization_results['model_type'] = 'regression'
        else:
            trainer = MultitaskBaselineTrainer(use_scibert=use_scibert)
            optimization_results = trainer.run_hyperparameter_optimization(max_combinations=max_combinations)
            optimization_results['model_type'] = 'classification'
        
        print(f"Hyperparameter optimization completed!")
        print(f"Best hyperparameters: {optimization_results['best_hyperparams']}")
        if 'best_score' in optimization_results:
            print(f"Best score: {optimization_results['best_score']:.4f}")
        
        return optimization_results
        
    except Exception as e:
        print(f"Error during hyperparameter optimization: {e}")
        traceback.print_exc()
        return None


def run_single_task_experiments(use_compressed_data: bool = False, use_scibert: bool = False):
    """Run individual task experiments."""
    print("Starting single-task experiments...")
    
    # Generate datasets
    generator = DatasetGenerator()
    generator.generate_all_datasets()
    
    results = {}
    model_type = "SciBERT" if use_scibert else "RoBERTa"
    
    tasks = [
        ('causality', CausalityBaselineTrainer),
        ('certainty', CertaintyBaselineTrainer),
        ('generalization', GeneralizationBaselineTrainer),
        ('sensationalism', SensationalismBaselineTrainer)
    ]
    
    for task_name, trainer_class in tasks:
        print(f"\nTraining {task_name} model with {model_type}...")
        start_time = time.time()
        
        try:
            if task_name in ['causality', 'certainty']:
                trainer = trainer_class(
                    use_compressed_data=use_compressed_data,
                    use_scibert=use_scibert
                )
            else:
                trainer = trainer_class(use_scibert=use_scibert)
            
            results[task_name] = trainer.run_training()
            
            elapsed = time.time() - start_time
            print(f"{task_name} training completed in {elapsed:.1f}s")
            
            # Print results summary
            if task_name == 'sensationalism':
                print(f"  Pearson correlation: {results[task_name]['pearson_correlation']:.4f}")
                print(f"  MSE: {results[task_name]['mse']:.4f}")
            else:
                print(f"  Macro F1: {results[task_name]['macro_f1']:.4f}")
                
        except Exception as e:
            print(f"Error training {task_name} model: {e}")
            traceback.print_exc()
            results[task_name] = None
    
    return results


def run_multitask_experiments(use_scibert: bool = False, include_regression: bool = True):
    """Run multi-task experiments."""
    print("Starting multi-task experiments...")
    
    # Generate multi-task datasets
    multitask_generator = MultitaskDatasetGenerator()
    multitask_generator.generate_all_multitask_datasets()
    
    results = {}
    model_type = "SciBERT" if use_scibert else "RoBERTa"
    
    # Multi-task classification (causality + certainty)
    print(f"\nTraining multi-task classification model with {model_type}...")
    start_time = time.time()
    
    try:
        multitask_trainer = MultitaskBaselineTrainer(use_scibert=use_scibert)
        results['multitask_classification'] = multitask_trainer.run_training()
        
        elapsed = time.time() - start_time
        print(f"Multi-task classification training completed in {elapsed:.1f}s")
        print(f"  Causality F1: {results['multitask_classification']['causality']['macro_f1']:.4f}")
        print(f"  Certainty F1: {results['multitask_classification']['certainty']['macro_f1']:.4f}")
        
    except Exception as e:
        print(f"Error training multi-task classification model: {e}")
        traceback.print_exc()
        results['multitask_classification'] = None
    
    # Multi-task regression (causality + certainty + sensationalism)
    if include_regression:
        print(f"\nPreparing multi-task regression data...")
        prepare_multitask_regression_data()
        
        print(f"Training multi-task regression model with {model_type}...")
        start_time = time.time()
        
        try:
            # Test both loss balancing strategies
            for strategy in ['fixed', 'adaptive']:
                print(f"  Testing {strategy} loss balancing...")
                
                regression_trainer = MultitaskRegressionTrainer(
                    use_scibert=use_scibert,
                    loss_balancing=strategy,
                    num_train_epochs=5,
                    per_device_train_batch_size=8,
                    early_stopping_patience=3
                )
                
                strategy_results = regression_trainer.run_training(use_validation_split=True)
                results[f'multitask_regression_{strategy}'] = strategy_results
                
                print(f"    Causality F1: {strategy_results['causality']['macro_f1']:.4f}")
                print(f"    Certainty F1: {strategy_results['certainty']['macro_f1']:.4f}")
                print(f"    Sensationalism Correlation: {strategy_results['sensationalism']['pearson_correlation']:.4f}")
                print(f"    Sensationalism MSE: {strategy_results['sensationalism']['mse']:.4f}")
            
            elapsed = time.time() - start_time
            print(f"Multi-task regression training completed in {elapsed:.1f}s")
            
        except Exception as e:
            print(f"Error training multi-task regression model: {e}")
            traceback.print_exc()
            results['multitask_regression_fixed'] = None
            results['multitask_regression_adaptive'] = None
    
    return results


def run_baseline_experiments(use_compressed_data: bool = False, use_scibert: bool = False, 
                           include_regression: bool = True):
    """Run all baseline experiments and save results."""
    print("="*60)
    print(f"RUNNING BASELINE EXPERIMENTS")
    print(f"Model: {'SciBERT' if use_scibert else 'RoBERTa'}")
    print(f"Compressed data: {use_compressed_data}")
    print(f"Include regression: {include_regression}")
    print("="*60)
    
    all_results = {}
    
    # Run single-task experiments
    single_task_results = run_single_task_experiments(use_compressed_data, use_scibert)
    all_results.update(single_task_results)
    
    # Run multi-task experiments
    multitask_results = run_multitask_experiments(use_scibert, include_regression)
    all_results.update(multitask_results)
    
    # Save results
    model_suffix = "scibert" if use_scibert else "roberta"
    compressed_suffix = "compressed" if use_compressed_data else "full"
    regression_suffix = "with_regression" if include_regression else "no_regression"
    
    output_file = Path(f'out/baseline_results_{compressed_suffix}_{model_suffix}_{regression_suffix}.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert results to JSON-serializable format
    json_results = {}
    for task, task_results in all_results.items():
        if task_results is None:
            json_results[task] = None
            continue
            
        if task == 'sensationalism':
            # Sensationalism is regression
            json_results[task] = {
                'pearson_correlation': float(task_results['pearson_correlation']),
                'p_value': float(task_results['p_value']),
                'mse': float(task_results['mse'])
            }
        elif 'multitask_classification' in task:
            # Multi-task classification has nested results
            json_results[task] = {
                'causality': {
                    'f1_per_class': task_results['causality']['f1_per_class'].tolist(),
                    'macro_f1': float(task_results['causality']['macro_f1'])
                },
                'certainty': {
                    'f1_per_class': task_results['certainty']['f1_per_class'].tolist(),
                    'macro_f1': float(task_results['certainty']['macro_f1'])
                }
            }
        elif 'multitask_regression' in task:
            # Multi-task regression has all three tasks
            json_results[task] = {
                'causality': {
                    'f1_per_class': task_results['causality']['f1_per_class'],
                    'macro_f1': float(task_results['causality']['macro_f1'])
                },
                'certainty': {
                    'f1_per_class': task_results['certainty']['f1_per_class'],
                    'macro_f1': float(task_results['certainty']['macro_f1'])
                },
                'sensationalism': {
                    'mse': float(task_results['sensationalism']['mse']),
                    'mae': float(task_results['sensationalism']['mae']),
                    'rmse': float(task_results['sensationalism']['rmse']),
                    'pearson_correlation': float(task_results['sensationalism']['pearson_correlation']),
                    'p_value': float(task_results['sensationalism']['p_value'])
                }
            }
        else:
            # Single classification tasks
            json_results[task] = {
                'f1_per_class': task_results['f1_per_class'].tolist(),
                'macro_f1': float(task_results['macro_f1'])
            }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\nEXPERIMENT SUMMARY:")
    print("-" * 40)
    for task, results in json_results.items():
        if results is None:
            print(f"{task}: FAILED")
        elif task == 'sensationalism':
            print(f"{task}: Correlation={results['pearson_correlation']:.4f}, MSE={results['mse']:.4f}")
        elif 'multitask' in task:
            if 'sensationalism' in results:
                print(f"{task}: Caus={results['causality']['macro_f1']:.4f}, "
                      f"Cert={results['certainty']['macro_f1']:.4f}, "
                      f"Sens_Corr={results['sensationalism']['pearson_correlation']:.4f}, "
                      f"Sens_MSE={results['sensationalism']['mse']:.4f}")
            else:
                print(f"{task}: Caus={results['causality']['macro_f1']:.4f}, "
                      f"Cert={results['certainty']['macro_f1']:.4f}")
        else:
            print(f"{task}: F1={results['macro_f1']:.4f}")
    
    return all_results


def run_comprehensive_experiments():
    """Run comprehensive experiments with both models and configurations."""
    set_seed(42)
    
    print("STARTING COMPREHENSIVE EXPERIMENTS")
    print("="*80)
    
    # Configuration matrix
    configurations = [
        # (use_scibert, use_compressed_data, include_regression)
        (False, False, True),   # RoBERTa, full data, with regression
        (False, True, True),    # RoBERTa, compressed data, with regression
        (True, False, True),    # SciBERT, full data, with regression
        (True, True, False),    # SciBERT, compressed data, no regression (faster)
    ]
    
    all_experiment_results = {}
    
    for i, (use_scibert, use_compressed, include_regression) in enumerate(configurations, 1):
        model_name = "SciBERT" if use_scibert else "RoBERTa"
        data_type = "compressed" if use_compressed else "full"
        regression_type = "with_regression" if include_regression else "classification_only"
        
        config_name = f"{model_name}_{data_type}_{regression_type}"
        
        print(f"\n[{i}/{len(configurations)}] Running configuration: {config_name}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            
            # Run baseline experiments
            results = run_baseline_experiments(
                use_compressed_data=use_compressed,
                use_scibert=use_scibert,
                include_regression=include_regression
            )
            
            elapsed = time.time() - start_time
            print(f"Configuration {config_name} completed in {elapsed/60:.1f} minutes")
            
            all_experiment_results[config_name] = {
                'results': results,
                'config': {
                    'use_scibert': use_scibert,
                    'use_compressed_data': use_compressed,
                    'include_regression': include_regression
                },
                'runtime_minutes': elapsed / 60
            }
            
        except Exception as e:
            print(f"Error in configuration {config_name}: {e}")
            traceback.print_exc()
            all_experiment_results[config_name] = {
                'results': None,
                'config': {
                    'use_scibert': use_scibert,
                    'use_compressed_data': use_compressed,
                    'include_regression': include_regression
                },
                'error': str(e)
            }
    
    # Save comprehensive results
    output_file = Path('out/comprehensive_experiment_results.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for config_name, config_data in all_experiment_results.items():
        json_results[config_name] = {
            'config': config_data['config'],
            'runtime_minutes': config_data.get('runtime_minutes', 0),
            'error': config_data.get('error', None),
            'success': config_data['results'] is not None
        }
        
        if config_data['results'] is not None:
            # Add summary metrics
            results = config_data['results']
            summary = {}
            
            for task, task_results in results.items():
                if task_results is None:
                    summary[task] = None
                elif task == 'sensationalism':
                    summary[task] = {
                        'correlation': float(task_results['pearson_correlation']),
                        'mse': float(task_results['mse'])
                    }
                elif 'multitask' in task and 'sensationalism' in task_results:
                    summary[task] = {
                        'causality_f1': float(task_results['causality']['macro_f1']),
                        'certainty_f1': float(task_results['certainty']['macro_f1']),
                        'sensationalism_correlation': float(task_results['sensationalism']['pearson_correlation']),
                        'sensationalism_mse': float(task_results['sensationalism']['mse'])
                    }
                elif 'multitask' in task:
                    summary[task] = {
                        'causality_f1': float(task_results['causality']['macro_f1']),
                        'certainty_f1': float(task_results['certainty']['macro_f1'])
                    }
                else:
                    summary[task] = {
                        'macro_f1': float(task_results['macro_f1'])
                    }
            
            json_results[config_name]['summary'] = summary
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENTS COMPLETED")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print("-" * 50)
    for config_name, config_data in json_results.items():
        if config_data['success']:
            runtime = config_data['runtime_minutes']
            print(f"{config_name}: SUCCESS ({runtime:.1f} min)")
        else:
            print(f"{config_name}: FAILED - {config_data.get('error', 'Unknown error')}")
    
    return all_experiment_results


def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("RUNNING QUICK TEST")
    print("="*40)
    
    set_seed(42)
    
    # Quick test with RoBERTa, no compression, classification only
    results = run_baseline_experiments(
        use_compressed_data=False,
        use_scibert=False,
        include_regression=False  # Skip regression for speed
    )
    
    print("\nQUICK TEST COMPLETED")
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test mode
        run_quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'hyperopt':
        # Hyperparameter optimization only
        set_seed(42)
        print("Running hyperparameter optimization...")
        
        # Run for both model types
        for use_scibert in [False, True]:
            model_name = "SciBERT" if use_scibert else "RoBERTa"
            print(f"\nOptimizing {model_name} classification model...")
            run_hyperparameter_optimization(
                use_scibert=use_scibert, 
                max_combinations=20, 
                model_type="classification"
            )
    else:
        # Full comprehensive experiments
        run_comprehensive_experiments()