#!/usr/bin/env python3
"""Command-line script for running frozen multi-task sensationalism experiments."""

import argparse
from pathlib import Path

from src.data.dataset_generator import DatasetGenerator
from src.models.frozen_multitask_sensationalism_model import FrozenMultitaskSensationalismTrainer

from transformers import set_seed


def run_frozen_multitask_sensationalism_experiments(args):
    """Run frozen multi-task sensationalism experiments based on provided arguments."""
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Generate datasets if requested
    if args.task in ['generate', 'all']:
        print("Generating sensationalism datasets...")
        generator = DatasetGenerator()
        generator.generate_sensationalism_datasets()
        print("✓ Sensationalism datasets generated")
    
    # Train models if requested
    if args.task in ['train', 'all']:
        results = {}
        
        # Define the pre-trained multi-task models to use
        pretrained_models = [
            "out/baseline_roberta_multitask",
            "out/baseline_scibert_multitask"
        ]
        
        # Train frozen models for each pre-trained multi-task model
        for pretrained_model_path in pretrained_models:
            if not Path(pretrained_model_path).exists():
                print(f"Warning: Pre-trained model not found at {pretrained_model_path}, skipping...")
                continue
            
            model_name = Path(pretrained_model_path).name
            print(f"Training frozen sensationalism model with {model_name}...")
            
            # Create trainer for this pre-trained model
            trainer = FrozenMultitaskSensationalismTrainer(
                pretrained_model_path=pretrained_model_path,
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                max_length=args.max_length,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_steps=args.warmup_steps
            )
            
            # Run training
            result = trainer.run_training(
                use_validation_split=args.use_validation_split,
                validation_size=args.validation_size
            )
            
            results[model_name] = result
            print(f"✓ Frozen sensationalism model trained with {model_name}")
        
        # Print results summary
        print_results_summary(results)
        
        return results


def print_results_summary(results):
    """Print a formatted summary of training results."""
    
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "="*60)
    print("FROZEN MULTI-TASK SENSATIONALISM RESULTS SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Pearson Correlation: {result['sensationalism']['pearson_correlation']:.4f}")
        print(f"  P-value: {result['sensationalism']['p_value']:.4f}")
        print(f"  MSE: {result['sensationalism']['mse']:.4f}")
        print(f"  MAE: {result['sensationalism']['mae']:.4f}")
        print(f"  RMSE: {result['sensationalism']['rmse']:.4f}")
    
    print("\n" + "="*60)


def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="Run frozen multi-task sensationalism experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["generate", "train", "all"],
        help="Task to perform: generate datasets, train models, or both"
    )
    
    # Model arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Output directory for models and results"
    )
    
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="temp",
        help="Directory containing temporary datasets"
    )
    
    # Training arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per device"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps"
    )
    
    # Validation arguments
    parser.add_argument(
        "--use_validation_split",
        action="store_true",
        default=False,
        help="Whether to use validation split for early stopping"
    )
    
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation"
    )
    
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=4,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0001,
        help="Early stopping threshold"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Run experiments
    run_frozen_multitask_sensationalism_experiments(args)


if __name__ == "__main__":
    main()
