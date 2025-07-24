"""Multi-task model for causality and certainty classification."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import random
import itertools

import pandas as pd
import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score


class MultiTaskClassifierModel(nn.Module):
    """Multi-task model for causality and certainty classification."""
    
    def __init__(self, model_name: str, causality_num_labels: int = 3, certainty_num_labels: int = 3):
        super().__init__()
        
        # Load base model configuration and model
        self.config = AutoConfig.from_pretrained(model_name)
        # Set num_labels in config instead of passing as argument
        self.config.num_labels = 2  # Temporary, we'll replace the classifier
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=self.config
        )
        
        # Add task-specific heads
        hidden_size = self.config.hidden_size
        self.causality_classifier = nn.Linear(hidden_size, causality_num_labels)
        self.certainty_classifier = nn.Linear(hidden_size, certainty_num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # Store number of labels for each task
        self.causality_num_labels = causality_num_labels
        self.certainty_num_labels = certainty_num_labels
        self.num_labels = causality_num_labels + certainty_num_labels
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing if supported by base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            try:
                self.base_model.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                # Fallback for models that don't accept kwargs
                self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing if supported by base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
    def get_input_embeddings(self):
        """Get input embeddings from base model."""
        if hasattr(self.base_model, 'get_input_embeddings'):
            return self.base_model.get_input_embeddings()
        return None
    
    def set_input_embeddings(self, value):
        """Set input embeddings in base model."""
        if hasattr(self.base_model, 'set_input_embeddings'):
            self.base_model.set_input_embeddings(value)
        
    def forward(self, input_ids=None, attention_mask=None, 
                causality_labels=None, certainty_labels=None, **kwargs):
        """Forward pass through the multi-task model."""
        
        # Get base model outputs - we need to access the underlying encoder
        # For RoBERTa, we need to call the roberta encoder directly
        if hasattr(self.base_model, 'roberta'):
            encoder_outputs = self.base_model.roberta(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        elif hasattr(self.base_model, 'bert'):
            encoder_outputs = self.base_model.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        else:
            # Fallback for other model types
            encoder_outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
        
        # Get pooled representation
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled_output = encoder_outputs.pooler_output
        else:
            # Use CLS token representation from last hidden state
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific predictions
        causality_logits = self.causality_classifier(pooled_output)
        certainty_logits = self.certainty_classifier(pooled_output)
        
        # Calculate losses if labels are provided
        total_loss = None
        loss_fct = nn.CrossEntropyLoss()
        
        losses = []
        if causality_labels is not None:
            causality_loss = loss_fct(
                causality_logits.view(-1, self.causality_num_labels), 
                causality_labels.view(-1)
            )
            losses.append(causality_loss)
        
        if certainty_labels is not None:
            certainty_loss = loss_fct(
                certainty_logits.view(-1, self.certainty_num_labels), 
                certainty_labels.view(-1)
            )
            losses.append(certainty_loss)
        
        # Combine losses with equal weighting
        if losses:
            total_loss = sum(losses)
        
        # Combine logits for compatibility
        combined_logits = torch.cat([causality_logits, certainty_logits], dim=-1)
        
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=combined_logits,
            hidden_states=None,
            attentions=None,
        )


class MultitaskBaselineTrainer:
    """Trainer for multi-task causality and certainty classification."""
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 output_dir: str = "out", 
                 temp_dir: str = "temp",
                 max_length: int = 1536,
                 use_scibert: bool = False,
                 **training_kwargs):
        
        # Set model name based on scibert flag
        if use_scibert:
            model_name = 'allenai/scibert_scivocab_uncased'
        
        self.model_name = model_name
        self.use_scibert = use_scibert
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.training_kwargs = training_kwargs
        
        # Initialize tokenizer and get model config to check max position embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # Ensure max_length doesn't exceed model's maximum position embeddings
        model_max_length = getattr(config, 'max_position_embeddings', 512)
        self.max_length = min(max_length, model_max_length)
        
        if max_length > model_max_length:
            print(f"Warning: Requested max_length ({max_length}) exceeds model's max position embeddings ({model_max_length}). Using {self.max_length} instead.")
        
        # Label mappings (compressed to 3 classes each)
        self.causality_id2label = {0: "Unclear", 1: "Causation", 2: "Correlation"}
        self.causality_label2id = {v: k for k, v in self.causality_id2label.items()}
        
        self.certainty_id2label = {0: "Certain", 1: "Somewhat certain", 2: "Uncertain"}
        self.certainty_label2id = {v: k for k, v in self.certainty_id2label.items()}
    
    def load_data(self, train_file: str, test_file: str):
        """Load and prepare multi-task data."""
        train_dataset = pd.read_csv(self.temp_dir / train_file, index_col=0)
        test_dataset = pd.read_csv(self.temp_dir / test_file, index_col=0)
        
        # Map labels to IDs
        train_dataset["causality_labels"] = train_dataset["causality"].map(self.causality_label2id)
        train_dataset["certainty_labels"] = train_dataset["certainty"].map(self.certainty_label2id)
        
        test_dataset["causality_labels"] = test_dataset["causality"].map(self.causality_label2id)
        test_dataset["certainty_labels"] = test_dataset["certainty"].map(self.certainty_label2id)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_dataset)
        test_dataset = Dataset.from_pandas(test_dataset)
        
        return train_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """Tokenize function for text input."""
        return self.tokenizer(
            examples["finding"],
            padding="max_length",  # Pad to max_length for consistent tensor sizes
            truncation=True, 
            max_length=self.max_length,
            return_tensors=None  # Let datasets handle tensor conversion
        )
    
    def prepare_datasets(self, train_dataset: Dataset, test_dataset: Dataset):
        """Tokenize and format datasets for training."""
        # Tokenize datasets with reasonable batch size
        batch_size = min(1000, len(train_dataset))
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        batch_size = min(1000, len(test_dataset))
        test_dataset = test_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        # Set format for PyTorch
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "causality_labels", "certainty_labels"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "causality_labels", "certainty_labels"])
        
        return train_dataset, test_dataset
    
    def create_model(self):
        """Create multi-task model."""
        model = MultiTaskClassifierModel(
            model_name=self.model_name,
            causality_num_labels=len(self.causality_id2label),
            certainty_num_labels=len(self.certainty_id2label)
        )
        return model
    
    def train_model(self, model, train_dataset: Dataset):
        """Train the multi-task model."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "temp_training"),
            num_train_epochs=self.training_kwargs.get('num_train_epochs', 5),
            per_device_train_batch_size=self.training_kwargs.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.training_kwargs.get('per_device_eval_batch_size', 8),
            logging_dir=str(self.output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=self.training_kwargs.get('learning_rate', 2e-5),
            weight_decay=self.training_kwargs.get('weight_decay', 0.01),
            warmup_steps=self.training_kwargs.get('warmup_steps', 200),
            save_strategy="no",
        )
        
        # Create data collator for consistent padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset):
        """Evaluate multi-task model."""
        predictions = trainer.predict(test_dataset)
        
        # Handle different prediction formats
        if hasattr(predictions, 'predictions'):
            combined_logits = predictions.predictions
        else:
            combined_logits = predictions
        
        # Split the combined logits back into task-specific logits
        causality_logits = combined_logits[:, :len(self.causality_id2label)]
        certainty_logits = combined_logits[:, len(self.causality_id2label):]
        
        # Get predictions
        causality_preds = causality_logits.argmax(axis=1)
        certainty_preds = certainty_logits.argmax(axis=1)
        
        # Get true labels
        causality_true = test_dataset['causality_labels'].numpy()
        certainty_true = test_dataset['certainty_labels'].numpy()
        
        # Calculate F1 scores
        causality_f1_per_class = f1_score(causality_true, causality_preds, average=None)
        causality_macro_f1 = f1_score(causality_true, causality_preds, average='macro')
        
        certainty_f1_per_class = f1_score(certainty_true, certainty_preds, average=None)
        certainty_macro_f1 = f1_score(certainty_true, certainty_preds, average='macro')
        
        return {
            'causality': {
                'f1_per_class': causality_f1_per_class.tolist(),
                'macro_f1': causality_macro_f1
            },
            'certainty': {
                'f1_per_class': certainty_f1_per_class.tolist(),
                'macro_f1': certainty_macro_f1
            }
        }
    
    def run_training(self):
        """Run the complete training pipeline for multi-task classification."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'causality_certainty_train.csv',
            'causality_certainty_test.csv'
        )
        
        # Prepare datasets
        train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        
        # Create model
        model = self.create_model()
        
        # Train
        trainer = self.train_model(model, train_dataset)
        
        # Save model
        model_name = 'baseline_scibert_multitask' if self.use_scibert else 'baseline_roberta_multitask'
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path / 'pytorch_model.bin')
        
        # Evaluate
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Causality F1 scores per class: {results['causality']['f1_per_class']}")
        print(f"Causality Macro F1: {results['causality']['macro_f1']:.4f}")
        print(f"Certainty F1 scores per class: {results['certainty']['f1_per_class']}")
        print(f"Certainty Macro F1: {results['certainty']['macro_f1']:.4f}")
        
        return results
    
    def get_hyperparameter_space(self, memory_safe: bool = False) -> Dict[str, List[Any]]:
        """Define hyperparameter search space."""
        if memory_safe:
            # Conservative settings to avoid OOM
            return {
                'learning_rate': [1e-5, 2e-5, 3e-5],
                'per_device_train_batch_size': [4, 8],  # Smaller batch sizes
                'num_train_epochs': [3, 5],
                'weight_decay': [0.0, 0.01],
                'warmup_steps': [100, 200],
                'max_length': [256, 512, 1024]  # Shorter sequences
            }
        else:
            # Full search space
            return {
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'per_device_train_batch_size': [4, 8, 16],
                'num_train_epochs': [3, 5, 7],
                'weight_decay': [0.0, 0.01, 0.1],
                'warmup_steps': [100, 200, 500],
                'max_length': [256, 512, 1024]
            }
    
    def create_validation_split(self, train_dataset: Dataset, val_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Create validation split from training data."""
        train_test_split_dataset = train_dataset.train_test_split(
            test_size=val_size,
            seed=42
        )
        
        return train_test_split_dataset['train'], train_test_split_dataset['test']
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def is_memory_intensive_config(self, hyperparams: Dict[str, Any]) -> bool:
        """Check if hyperparameter configuration is likely to cause OOM."""
        batch_size = hyperparams.get('per_device_train_batch_size', 8)
        max_length = hyperparams.get('max_length', 1536)
        
        # Consider configurations with large batch size and long sequences as risky
        memory_score = batch_size * max_length
        return memory_score > 16 * 1024  # Threshold for risky configurations
    
    def evaluate_hyperparameters(self, hyperparams: Dict[str, Any], 
                                train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, float]:
        """Evaluate a specific hyperparameter configuration."""
        original_max_length = self.max_length
        
        try:
            # Clear GPU memory before starting
            self.clear_gpu_memory()
            
            # Check if this configuration is likely to cause OOM
            if self.is_memory_intensive_config(hyperparams):
                print("    Warning: Memory-intensive configuration detected")
                # Reduce batch size for memory-intensive configs
                if hyperparams.get('per_device_train_batch_size', 8) > 8:
                    hyperparams = hyperparams.copy()
                    hyperparams['per_device_train_batch_size'] = min(8, hyperparams['per_device_train_batch_size'])
                    print(f"    Reduced batch size to {hyperparams['per_device_train_batch_size']}")
            
            # Update training kwargs with hyperparameters
            temp_kwargs = self.training_kwargs.copy()
            temp_kwargs.update(hyperparams)
            
            # Update max_length if specified and re-tokenize datasets
            if 'max_length' in hyperparams:
                self.max_length = hyperparams['max_length']
                train_dataset, val_dataset = self._retokenize_datasets(train_dataset, val_dataset)
            
            # Create model
            model = self.create_model()
            
            # Create training arguments with hyperparameters
            training_args = self._create_training_args(temp_kwargs)
            
            # Create data collator for consistent padding
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Train model
            trainer.train()
            
            # Evaluate on validation set
            results = self.evaluate_model(trainer, val_dataset)
            
            # Calculate combined score (average of both tasks' macro F1)
            combined_score = (results['causality']['macro_f1'] + results['certainty']['macro_f1']) / 2
            
            # Clean up
            del model, trainer
            self.clear_gpu_memory()
            
            return {
                'combined_score': combined_score,
                'causality_f1': results['causality']['macro_f1'],
                'certainty_f1': results['certainty']['macro_f1'],
                'hyperparams': hyperparams
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"    CUDA OOM Error: {str(e)[:100]}...")
                self.clear_gpu_memory()
                
                return {
                    'combined_score': 0.0,
                    'causality_f1': 0.0,
                    'certainty_f1': 0.0,
                    'hyperparams': hyperparams,
                    'error': 'CUDA_OOM'
                }
            else:
                raise e
        finally:
            # Always restore original max_length
            self.max_length = original_max_length
    
    def _retokenize_datasets(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Re-tokenize datasets with current max_length."""
        batch_size = min(1000, len(train_dataset))
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        batch_size = min(1000, len(val_dataset))
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        # Set format for PyTorch
        columns = ["input_ids", "attention_mask", "causality_labels", "certainty_labels"]
        train_dataset.set_format("torch", columns=columns)
        val_dataset.set_format("torch", columns=columns)
        
        return train_dataset, val_dataset
    
    def _create_training_args(self, temp_kwargs: Dict[str, Any]) -> TrainingArguments:
        """Create training arguments with hyperparameters."""
        return TrainingArguments(
            output_dir=str(self.output_dir / "temp_hyperparam_training"),
            num_train_epochs=temp_kwargs.get('num_train_epochs', 5),
            per_device_train_batch_size=temp_kwargs.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=min(temp_kwargs.get('per_device_eval_batch_size', 8), 8),
            learning_rate=temp_kwargs.get('learning_rate', 2e-5),
            weight_decay=temp_kwargs.get('weight_decay', 0.01),
            warmup_steps=temp_kwargs.get('warmup_steps', 200),
            logging_dir=str(self.output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="no",
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )
    
    def grid_search_hyperparameters(self, max_combinations: int = 50, memory_safe: bool = False) -> Dict[str, Any]:
        """Perform grid search over hyperparameter space."""
        print("Starting hyperparameter grid search...")
        
        # Load and prepare data
        train_dataset, test_dataset = self.load_data(
            'causality_certainty_train.csv',
            'causality_certainty_test.csv'
        )
        
        # Create validation split
        train_split, val_split = self.create_validation_split(train_dataset)
        
        # Prepare datasets (tokenize)
        train_split, val_split = self.prepare_datasets(train_split, val_split)
        
        # Get hyperparameter space
        hyperparam_space = self.get_hyperparameter_space(memory_safe=memory_safe)
        
        if memory_safe:
            print("Using memory-safe hyperparameter space to avoid CUDA OOM errors")
        
        # Generate all combinations
        keys = list(hyperparam_space.keys())
        values = list(hyperparam_space.values())
        all_combinations = list(itertools.product(*values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            print(f"Limiting search to {max_combinations} random combinations out of {len(all_combinations)}")
            all_combinations = random.sample(all_combinations, max_combinations)
        
        best_score = -1
        best_hyperparams = None
        all_results = []
        
        print(f"Evaluating {len(all_combinations)} hyperparameter combinations...")
        
        successful_runs = 0
        oom_errors = 0
        other_errors = 0
        
        for i, combination in enumerate(all_combinations):
            hyperparams = dict(zip(keys, combination))
            
            print(f"[{i+1}/{len(all_combinations)}] Testing: {hyperparams}")
            
            try:
                result = self.evaluate_hyperparameters(hyperparams, train_split, val_split)
                all_results.append(result)
                
                if 'error' in result:
                    if result['error'] == 'CUDA_OOM':
                        oom_errors += 1
                        print(f"  CUDA OOM - Skipping this configuration")
                    continue
                
                successful_runs += 1
                print(f"  Combined F1: {result['combined_score']:.4f} "
                      f"(Causality: {result['causality_f1']:.4f}, "
                      f"Certainty: {result['certainty_f1']:.4f})")
                
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_hyperparams = hyperparams
                    print(f"  *** New best score: {best_score:.4f} ***")
                
            except Exception as e:
                other_errors += 1
                print(f"  Unexpected error: {str(e)[:100]}...")
                # Add failed result to track what didn't work
                all_results.append({
                    'combined_score': 0.0,
                    'causality_f1': 0.0,
                    'certainty_f1': 0.0,
                    'hyperparams': hyperparams,
                    'error': str(e)[:200]
                })
                continue
        
        # Save results
        results_file = self.output_dir / 'hyperparameter_search_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'best_hyperparams': best_hyperparams,
                'best_score': best_score,
                'all_results': all_results
            }, f, indent=2)
        
        print(f"\nHyperparameter search completed!")
        print(f"Successful runs: {successful_runs}/{len(all_combinations)}")
        print(f"CUDA OOM errors: {oom_errors}")
        print(f"Other errors: {other_errors}")
        print(f"Best hyperparameters: {best_hyperparams}")
        print(f"Best combined F1 score: {best_score:.4f}")
        print(f"Results saved to: {results_file}")
        
        # Print memory optimization suggestions
        if oom_errors > 0:
            print("\nMemory Optimization Suggestions:")
            print("- Consider reducing max batch size further (current max: 16)")
            print("- Consider reducing max sequence length (current max: 1536)")
            print("- Enable gradient accumulation to simulate larger batches")
            print("- Use mixed precision training (fp16) if available")
        
        return {
            'best_hyperparams': best_hyperparams,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def run_training_with_best_hyperparams(self) -> Dict[str, Any]:
        """Run training with the best hyperparameters found from search."""
        # Load hyperparameter search results
        results_file = self.output_dir / 'hyperparameter_search_results.json'
        
        if not results_file.exists():
            print("No hyperparameter search results found. Running grid search first...")
            search_results = self.grid_search_hyperparameters()
            best_hyperparams = search_results['best_hyperparams']
        else:
            with open(results_file, 'r') as f:
                search_results = json.load(f)
            best_hyperparams = search_results['best_hyperparams']
        
        print(f"Training with best hyperparameters: {best_hyperparams}")
        
        # Update training kwargs with best hyperparameters
        self.training_kwargs.update(best_hyperparams)
        
        # Update max_length if specified
        if 'max_length' in best_hyperparams:
            self.max_length = best_hyperparams['max_length']
        
        # Run normal training with optimized hyperparameters
        return self.run_training()
    
    def run_hyperparameter_optimization(self, strategy: str = "grid_search", max_combinations: int = 20) -> Dict[str, Any]:
        """Run hyperparameter optimization with specified strategy."""
        if strategy == "grid_search":
            return self.grid_search_hyperparameters(max_combinations=max_combinations)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")