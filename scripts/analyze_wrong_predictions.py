#!/usr/bin/env python3
"""Script to analyze wrong predictions from trained models.

This script loads trained models from the 'out' folder and runs them on test sets,
saving all wrong predictions (along with inputs and true labels) to CSV files.
"""

import argparse
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import classification_report

# Import custom models
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.multitask_classifier_model_sep import MultiTaskClassifierModel, MultitaskBaselineTrainer
from models.multitask_regression_model import MultiTaskRegressionModel, MultitaskRegressionTrainer
from models.causality_base_model import CausalityBaselineTrainer
from models.certainty_base_model import CertaintyBaselineTrainer
from models.sensationalism_base_model import SensationalismBaselineTrainer


class WrongPredictionsAnalyzer:
    """Analyzer for wrong predictions from trained models."""
    
    def __init__(self, output_dir: str = "out", temp_dir: str = "temp", results_dir: str = "wrong_predictions"):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_baseline_model(self, model_type: str, use_scibert: bool = False) -> Tuple[Any, Any, Dict]:
        """Load a trained baseline model."""
        model_name = 'baseline_scibert_' + model_type if use_scibert else 'baseline_roberta_' + model_type
        model_path = self.output_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load tokenizer from original model name, not from saved directory
        original_model_name = 'allenai/scibert_scivocab_uncased' if use_scibert else 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        
        # Load config and model from saved directory
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, config
    
    def load_multitask_classification_model(self, use_scibert: bool = False) -> Tuple[Any, Any, Dict]:
        """Load a trained multitask classification model."""
        # Look for the model in different possible locations
        model_type = "multitask"
        model_name = 'baseline_scibert_' + model_type if use_scibert else 'baseline_roberta_' + model_type
        model_path = self.output_dir / model_name

        if model_path is None:
            raise FileNotFoundError(f"Multitask classification model not found in {self.output_dir}")
        
        # Load tokenizer from original model name
        model_name = 'allenai/scibert_scivocab_uncased' if use_scibert else 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = MultiTaskClassifierModel(
            model_name=model_name,
            causality_num_labels=3,
            certainty_num_labels=3
        )
        
        # Load state dict - try different formats
        model_file = None
        if (model_path / 'pytorch_model.bin').exists():
            model_file = model_path / 'pytorch_model.bin'
            state_dict = torch.load(model_file, map_location=self.device)
            model.load_state_dict(state_dict)
        elif (model_path / 'model.safetensors').exists():
            # For safetensors format, load the model directly
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, {'model_type': 'multitask_classification'}
    
    def load_multitask_regression_model(self, use_scibert: bool = False) -> Tuple[Any, Any, Dict]:
        """Load a trained multitask regression model."""
        # Look for the model in different possible locations
        possible_paths = [
            self.output_dir / 'multitask_regression',
            self.output_dir / 'baseline_roberta_multitask_regression'
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Multitask regression model not found in {self.output_dir}")
        
        # Load tokenizer from original model name
        model_name = 'allenai/scibert_scivocab_uncased' if use_scibert else 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        model = MultiTaskRegressionModel(
            model_name=model_name,
            causality_num_labels=3,
            certainty_num_labels=3,
            loss_balancing="fixed"
        )
        
        # Load state dict - try different formats
        model_file = None
        if (model_path / 'pytorch_model.bin').exists():
            model_file = model_path / 'pytorch_model.bin'
            state_dict = torch.load(model_file, map_location=self.device)
            model.load_state_dict(state_dict)
        elif (model_path / 'model.safetensors').exists():
            # For safetensors format, load the model directly
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, {'model_type': 'multitask_regression'}
    
    def load_test_data(self, task: str) -> pd.DataFrame:
        """Load test data for a specific task."""
        test_file = f"{task}_test.csv"
        test_path = self.temp_dir / test_file
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        data = pd.read_csv(test_path, index_col=0)
        
        # Apply label mapping for compressed models
        if task == 'causality':
            # Map original labels to compressed labels (3 classes)
            label_mapping = {
                "Explicitly states: no relation": 0,
                "Causation": 1, 
                "Correlation": 2,
                "No mention of a relation": 0  # Map to "Unclear"
            }
            data['label'] = data['label'].map(label_mapping)
        elif task == 'certainty':
            # Map original labels to compressed labels (3 classes)
            label_mapping = {
                "Certain": 0,
                "Somewhat certain": 1,
                "Uncertain": 2
            }
            data['label'] = data['label'].map(label_mapping)
        elif task == 'causality_certainty':
            # Combined multitask classification test set
            causality_mapping = {
                "Unclear": 0,
                "Causation": 1,
                "Correlation": 2,
                # Also handle full labels if present
                "Explicitly states: no relation": 0,
                "No mention of a relation": 0
            }
            certainty_mapping = {
                "Certain": 0,
                "Somewhat certain": 1,
                "Uncertain": 2
            }
            if 'causality' in data.columns:
                data['causality_labels'] = data['causality'].map(causality_mapping)
            if 'certainty' in data.columns:
                data['certainty_labels'] = data['certainty'].map(certainty_mapping)
        elif task == 'causality_certainty_sensationalism':
            # Combined multitask regression test set
            causality_mapping = {
                "Unclear": 0,
                "Causation": 1,
                "Correlation": 2,
                "Explicitly states: no relation": 0,
                "No mention of a relation": 0
            }
            certainty_mapping = {
                "Certain": 0,
                "Somewhat certain": 1,
                "Uncertain": 2
            }
            if 'causality' in data.columns:
                data['causality_labels'] = data['causality'].map(causality_mapping)
            if 'certainty' in data.columns:
                data['certainty_labels'] = data['certainty'].map(certainty_mapping)
            # Sensationalism target as float
            if 'sensationalism' in data.columns:
                data['sensationalism_labels'] = data['sensationalism'].astype(float)
        elif task == 'sensationalism':
            # Sensationalism uses 'score' column instead of 'label'
            data['label'] = data['score']
            data = data.drop('score', axis=1)
        
        return data
    
    def tokenize_data(self, data: pd.DataFrame, tokenizer: Any, max_length: int = None) -> Dataset:
        """Tokenize the data."""
        def tokenize_function(examples):
            return tokenizer(
                examples["finding"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None
            )
        
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        
        return dataset
    
    def predict_baseline(self, model: Any, dataset: Dataset, id2label: Dict[int, str]) -> Tuple[List[int], List[float]]:
        """Get predictions from a baseline model."""
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get the i-th example
                example = dataset[i]
                
                # Prepare inputs correctly
                input_ids = example['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = example['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.item())
                probabilities.append(probs.max().item())
        
        return predictions, probabilities
    
    def predict_multitask_classification(self, model: Any, dataset: Dataset) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Get predictions from a multitask classification model."""
        causality_predictions = []
        certainty_predictions = []
        causality_probs = []
        certainty_probs = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get the i-th example
                example = dataset[i]
                
                # Prepare inputs correctly
                input_ids = example['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = example['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Split combined logits: first 3 for causality, next 3 for certainty
                logits = outputs.logits
                causality_logits = logits[:, :3]
                certainty_logits = logits[:, 3:6]
                
                causality_probs_batch = torch.softmax(causality_logits, dim=-1)
                certainty_probs_batch = torch.softmax(certainty_logits, dim=-1)
                
                causality_pred = torch.argmax(causality_logits, dim=-1)
                certainty_pred = torch.argmax(certainty_logits, dim=-1)
                
                causality_predictions.append(causality_pred.item())
                certainty_predictions.append(certainty_pred.item())
                causality_probs.append(causality_probs_batch.max().item())
                certainty_probs.append(certainty_probs_batch.max().item())
        
        return causality_predictions, certainty_predictions, causality_probs, certainty_probs
    
    def predict_multitask_regression(self, model: Any, dataset: Dataset) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Get predictions from a multitask regression model."""
        causality_predictions = []
        certainty_predictions = []
        sensationalism_predictions = []
        causality_probs = []
        certainty_probs = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                # Get the i-th example
                example = dataset[i]
                
                # Prepare inputs correctly
                input_ids = example['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = example['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Split combined logits: first 3 for causality, next 3 for certainty, last value for regression
                logits = outputs.logits
                causality_logits = logits[:, :3]
                certainty_logits = logits[:, 3:6]
                sensationalism_pred = logits[:, -1]
                
                causality_probs_batch = torch.softmax(causality_logits, dim=-1)
                certainty_probs_batch = torch.softmax(certainty_logits, dim=-1)
                
                causality_pred = torch.argmax(causality_logits, dim=-1)
                certainty_pred = torch.argmax(certainty_logits, dim=-1)
                
                causality_predictions.append(causality_pred.item())
                certainty_predictions.append(certainty_pred.item())
                sensationalism_predictions.append(sensationalism_pred.item())
                causality_probs.append(causality_probs_batch.max().item())
                certainty_probs.append(certainty_probs_batch.max().item())
        
        return causality_predictions, certainty_predictions, sensationalism_predictions, causality_probs, certainty_probs
    
    def analyze_baseline_wrong_predictions(self, model_type: str, use_scibert: bool = False):
        """Analyze wrong predictions for a baseline model."""
        print(f"\nAnalyzing {model_type} baseline model...")
        
        # Load model
        model, tokenizer, config = self.load_baseline_model(model_type, use_scibert)
        
        # Load test data
        test_data = self.load_test_data(model_type)
        
        # Tokenize data using the model's max position embeddings
        max_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 512
        dataset = self.tokenize_data(test_data, tokenizer, max_length=max_length)
        
        # Get predictions
        predictions, probabilities = self.predict_baseline(model, dataset, config.id2label)
        
        # Create results dataframe
        results_df = test_data.copy()
        results_df['predicted_label'] = predictions
        results_df['predicted_probability'] = probabilities
        results_df['is_correct'] = results_df['label'] == results_df['predicted_label']
        
        # Get wrong predictions
        wrong_predictions = results_df[~results_df['is_correct']].copy()
        
        # Add label names
        id2label = config.id2label
        wrong_predictions['true_label_name'] = wrong_predictions['label'].map(id2label)
        wrong_predictions['predicted_label_name'] = wrong_predictions['predicted_label'].map(id2label)
        
        # Save results
        model_suffix = 'scibert' if use_scibert else 'roberta'
        output_file = self.results_dir / f"wrong_predictions_{model_type}_{model_suffix}.csv"
        wrong_predictions.to_csv(output_file, index=False)
        
        # Print summary
        total_samples = len(results_df)
        wrong_samples = len(wrong_predictions)
        accuracy = (total_samples - wrong_samples) / total_samples
        
        print(f"Total samples: {total_samples}")
        print(f"Wrong predictions: {wrong_samples}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Wrong predictions saved to: {output_file}")
        
        return wrong_predictions
    
    def analyze_multitask_classification_wrong_predictions(self, use_scibert: bool = False):
        """Analyze wrong predictions for multitask classification model."""
        print(f"\nAnalyzing multitask classification model...")
        
        # Load model
        model, tokenizer, config = self.load_multitask_classification_model(use_scibert)
        
        # Load test data
        test_data = self.load_test_data('causality_certainty')
        
        # Tokenize data using the model's max position embeddings
        max_length = config.get('max_position_embeddings', 512) if isinstance(config, dict) else 512
        dataset = self.tokenize_data(test_data, tokenizer, max_length=max_length)
        
        # Get predictions
        causality_preds, certainty_preds, causality_probs, certainty_probs = self.predict_multitask_classification(model, dataset)
        
        # Create results dataframe
        results_df = test_data.copy()
        results_df['predicted_causality'] = causality_preds
        results_df['predicted_certainty'] = certainty_preds
        results_df['causality_probability'] = causality_probs
        results_df['certainty_probability'] = certainty_probs
        
        # Check correctness
        results_df['causality_correct'] = results_df['causality_labels'] == results_df['predicted_causality']
        results_df['certainty_correct'] = results_df['certainty_labels'] == results_df['predicted_certainty']
        results_df['both_correct'] = results_df['causality_correct'] & results_df['certainty_correct']
        
        # Get wrong predictions (either task wrong)
        wrong_predictions = results_df[~results_df['both_correct']].copy()
        
        # Add label names
        causality_id2label = {0: "Unclear", 1: "Causation", 2: "Correlation"}
        certainty_id2label = {0: "Certain", 1: "Somewhat certain", 2: "Uncertain"}
        
        wrong_predictions['true_causality_name'] = wrong_predictions['causality_labels'].map(causality_id2label)
        wrong_predictions['predicted_causality_name'] = wrong_predictions['predicted_causality'].map(causality_id2label)
        wrong_predictions['true_certainty_name'] = wrong_predictions['certainty_labels'].map(certainty_id2label)
        wrong_predictions['predicted_certainty_name'] = wrong_predictions['predicted_certainty'].map(certainty_id2label)
        
        # Save results
        model_suffix = 'scibert' if use_scibert else 'roberta'
        output_file = self.results_dir / f"wrong_predictions_multitask_classification_{model_suffix}.csv"
        wrong_predictions.to_csv(output_file, index=False)
        
        # Print summary
        total_samples = len(results_df)
        wrong_samples = len(wrong_predictions)
        accuracy = (total_samples - wrong_samples) / total_samples
        
        causality_accuracy = results_df['causality_correct'].mean()
        certainty_accuracy = results_df['certainty_correct'].mean()
        
        print(f"Total samples: {total_samples}")
        print(f"Wrong predictions (either task): {wrong_samples}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Causality accuracy: {causality_accuracy:.4f}")
        print(f"Certainty accuracy: {certainty_accuracy:.4f}")
        print(f"Wrong predictions saved to: {output_file}")
        
        return wrong_predictions
    
    def analyze_multitask_regression_wrong_predictions(self, use_scibert: bool = False):
        """Analyze wrong predictions for multitask regression model."""
        print(f"\nAnalyzing multitask regression model...")
        
        # Load model
        model, tokenizer, config = self.load_multitask_regression_model(use_scibert)
        
        # Load test data
        test_data = self.load_test_data('causality_certainty_sensationalism')
        
        # Tokenize data using the model's max position embeddings
        max_length = config.get('max_position_embeddings', 512) if isinstance(config, dict) else 512
        dataset = self.tokenize_data(test_data, tokenizer, max_length=max_length)
        
        # Get predictions
        causality_preds, certainty_preds, sensationalism_preds, causality_probs, certainty_probs = self.predict_multitask_regression(model, dataset)
        
        # Create results dataframe
        results_df = test_data.copy()
        results_df['predicted_causality'] = causality_preds
        results_df['predicted_certainty'] = certainty_preds
        results_df['predicted_sensationalism'] = sensationalism_preds
        results_df['causality_probability'] = causality_probs
        results_df['certainty_probability'] = certainty_probs
        
        # Check correctness for classification tasks
        results_df['causality_correct'] = results_df['causality_labels'] == results_df['predicted_causality']
        results_df['certainty_correct'] = results_df['certainty_labels'] == results_df['predicted_certainty']
        
        # Calculate regression error for sensationalism
        results_df['sensationalism_error'] = abs(results_df['sensationalism'] - results_df['predicted_sensationalism'])
        results_df['sensationalism_correct'] = results_df['sensationalism_error'] < 0.5  # Threshold for "correct"
        
        # Overall correctness (all tasks correct)
        results_df['all_correct'] = (results_df['causality_correct'] & 
                                   results_df['certainty_correct'] & 
                                   results_df['sensationalism_correct'])
        
        # Get wrong predictions (any task wrong)
        wrong_predictions = results_df[~results_df['all_correct']].copy()
        
        # Add label names
        causality_id2label = {0: "Unclear", 1: "Causation", 2: "Correlation"}
        certainty_id2label = {0: "Certain", 1: "Somewhat certain", 2: "Uncertain"}
        
        wrong_predictions['true_causality_name'] = wrong_predictions['causality_labels'].map(causality_id2label)
        wrong_predictions['predicted_causality_name'] = wrong_predictions['predicted_causality'].map(causality_id2label)
        wrong_predictions['true_certainty_name'] = wrong_predictions['certainty_labels'].map(certainty_id2label)
        wrong_predictions['predicted_certainty_name'] = wrong_predictions['predicted_certainty'].map(certainty_id2label)
        
        # Save results
        model_suffix = 'scibert' if use_scibert else 'roberta'
        output_file = self.results_dir / f"wrong_predictions_multitask_regression_{model_suffix}.csv"
        wrong_predictions.to_csv(output_file, index=False)
        
        # Print summary
        total_samples = len(results_df)
        wrong_samples = len(wrong_predictions)
        accuracy = (total_samples - wrong_samples) / total_samples
        
        causality_accuracy = results_df['causality_correct'].mean()
        certainty_accuracy = results_df['certainty_correct'].mean()
        sensationalism_accuracy = results_df['sensationalism_correct'].mean()
        sensationalism_mae = results_df['sensationalism_error'].mean()
        
        print(f"Total samples: {total_samples}")
        print(f"Wrong predictions (any task): {wrong_samples}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Causality accuracy: {causality_accuracy:.4f}")
        print(f"Certainty accuracy: {certainty_accuracy:.4f}")
        print(f"Sensationalism accuracy: {sensationalism_accuracy:.4f}")
        print(f"Sensationalism MAE: {sensationalism_mae:.4f}")
        print(f"Wrong predictions saved to: {output_file}")
        
        return wrong_predictions
    
    def run_all_analyses(self, use_scibert: bool = False):
        """Run analysis for all available models."""
        print("Starting wrong predictions analysis...")
        print(f"Results will be saved to: {self.results_dir}")
        
        results = {}
        
        # Analyze baseline models
        baseline_tasks = ['causality', 'certainty', 'sensationalism']
        for task in baseline_tasks:
            try:
                results[f'baseline_{task}'] = self.analyze_baseline_wrong_predictions(task, use_scibert)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        # Analyze multitask classification model
        try:
            results['multitask_classification'] = self.analyze_multitask_classification_wrong_predictions(use_scibert)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        # Analyze multitask regression model
        try:
            results['multitask_regression'] = self.analyze_multitask_regression_wrong_predictions(use_scibert)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
        
        print(f"\nAnalysis complete! Results saved to: {self.results_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze wrong predictions from trained models')
    parser.add_argument('--model', choices=['causality', 'certainty', 'sensationalism', 'multitask-classification', 'multitask-regression', 'all'],
                       default='all', help='Which model(s) to analyze')
    parser.add_argument('--use-scibert', action='store_true', help='Use SciBERT models instead of RoBERTa')
    parser.add_argument('--output-dir', default='out', help='Directory containing trained models')
    parser.add_argument('--temp-dir', default='temp', help='Directory containing test data')
    parser.add_argument('--results-dir', default='wrong_predictions', help='Directory to save wrong predictions')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = WrongPredictionsAnalyzer(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        results_dir=args.results_dir
    )
    
    if args.model == 'all':
        analyzer.run_all_analyses(args.use_scibert)
    else:
        if args.model in ['causality', 'certainty', 'sensationalism']:
            analyzer.analyze_baseline_wrong_predictions(args.model, args.use_scibert)
        elif args.model == 'multitask-classification':
            analyzer.analyze_multitask_classification_wrong_predictions(args.use_scibert)
        elif args.model == 'multitask-regression':
            analyzer.analyze_multitask_regression_wrong_predictions(args.use_scibert)


if __name__ == "__main__":
    main()
