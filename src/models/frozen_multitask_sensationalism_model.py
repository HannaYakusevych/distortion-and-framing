"""Frozen multi-task model with sensationalism head for transfer learning."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import warnings

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
    TrainerCallback,
    EarlyStoppingCallback,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

class FrozenMultitaskSensationalismModel(nn.Module):
    """Frozen multi-task model with sensationalism head for transfer learning."""
    
    def __init__(self, pretrained_model_path: str, hidden_size: int = 768):
        super().__init__()
        
        # Load the pre-trained multi-task model
        self.pretrained_model_path = pretrained_model_path
        
        # Load the base model from the pre-trained path
        self.base_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)
        
        # Freeze all parameters of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Enhanced sensationalism regression head with multiple layers
        self.sensationalism_head = nn.Sequential(
            nn.Dropout(0.2),  # Higher dropout for regularization
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize the sensationalism head with better initialization
        for layer in self.sensationalism_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Alternative: Use LayerNorm for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the frozen model with sensationalism head."""
        
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
            # Fallback for other model types - call the base model directly
            # but we need to handle the output correctly
            base_outputs = self.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            # For SequenceClassifierOutput, we need to access hidden_states
            if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states is not None:
                # Use the last hidden state
                pooled_output = base_outputs.hidden_states[-1][:, 0, :]
            else:
                # Fallback: try to get the pooled output
                pooled_output = base_outputs.pooler_output
            # Skip the rest of the pooling logic
            pooled_output = self.layer_norm(pooled_output)
            pooled_output = self.dropout(pooled_output)
            sensationalism_logits = self.sensationalism_head(pooled_output).squeeze(-1)
            
            # Calculate loss if labels are provided
            total_loss = None
            if labels is not None:
                # Try different loss functions for better performance
                mse_loss = nn.MSELoss()(sensationalism_logits.view(-1), labels.view(-1).float())
                mae_loss = nn.L1Loss()(sensationalism_logits.view(-1), labels.view(-1).float())
                
                # Combine MSE and MAE for better training
                total_loss = 0.7 * mse_loss + 0.3 * mae_loss
            
            return SequenceClassifierOutput(
                loss=total_loss,
                logits=sensationalism_logits.unsqueeze(-1),  # Add dimension back for compatibility
                hidden_states=None,
                attentions=None,
            )
        
        # Get pooled representation
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled_output = encoder_outputs.pooler_output
        else:
            # Use CLS token representation from last hidden state
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        
        # Apply layer normalization and dropout for better training stability
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        sensationalism_logits = self.sensationalism_head(pooled_output).squeeze(-1)
        
        # Calculate loss if labels are provided
        total_loss = None
        if labels is not None:
            # Try different loss functions for better performance
            mse_loss = nn.MSELoss()(sensationalism_logits.view(-1), labels.view(-1).float())
            mae_loss = nn.L1Loss()(sensationalism_logits.view(-1), labels.view(-1).float())
            
            # Combine MSE and MAE for better training
            total_loss = 0.7 * mse_loss + 0.3 * mae_loss
        
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=sensationalism_logits.unsqueeze(-1),  # Add dimension back for compatibility
            hidden_states=None,
            attentions=None,
        )


class FrozenMultitaskSensationalismTrainer:
    """Trainer for frozen multi-task model with sensationalism head."""
    
    def __init__(self, 
                 pretrained_model_path: str,
                 output_dir: str = "out", 
                 temp_dir: str = "temp",
                 max_length: int = 1536,
                 early_stopping_patience: int = 4,
                 early_stopping_threshold: float = 0.0001,
                 **training_kwargs):
        """
        Initialize frozen multi-task sensationalism trainer.
        
        Args:
            pretrained_model_path: Path to the pre-trained multi-task model
        """
        
        # Suppress common warnings
        warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")
        warnings.filterwarnings("ignore", message=".*max_length.*is ignored when.*padding.*True.*")
        
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.training_kwargs = training_kwargs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        
        # Load tokenizer from the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        
        # Get model config to check max position embeddings
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model_max_length = getattr(config, 'max_position_embeddings', 512)
        self.max_length = min(max_length, model_max_length)
        
        if max_length > model_max_length:
            print(f"Warning: Requested max_length ({max_length}) exceeds model's max position embeddings ({model_max_length}). Using {self.max_length} instead.")
    
    def load_data(self, train_file: str, test_file: str):
        """Load and prepare sensationalism regression data."""
        train_dataset = pd.read_csv(self.temp_dir / train_file, index_col=0)
        train_dataset["labels"] = train_dataset["score"].astype(float)
        
        test_dataset = pd.read_csv(self.temp_dir / test_file, index_col=0)
        test_dataset["labels"] = test_dataset["score"].astype(float)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_dataset)
        test_dataset = Dataset.from_pandas(test_dataset)
        
        return train_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """Tokenize function for text input."""
        return self.tokenizer(
            examples["finding"],
            padding="max_length",
            truncation=True, 
            max_length=self.max_length,
            return_tensors=None
        )
    
    def prepare_datasets(self, train_dataset: Dataset, test_dataset: Dataset):
        """Tokenize and format datasets for training."""
        # Tokenize datasets with reasonable batch size
        batch_size = min(1000, len(train_dataset))
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        batch_size = min(1000, len(test_dataset))
        test_dataset = test_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        # Set format for PyTorch
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return train_dataset, test_dataset
    
    def create_model(self):
        """Create frozen multi-task model with sensationalism head."""
        # Get hidden size from the pre-trained model config
        config = AutoConfig.from_pretrained(self.pretrained_model_path)
        hidden_size = config.hidden_size
        
        model = FrozenMultitaskSensationalismModel(
            pretrained_model_path=self.pretrained_model_path,
            hidden_size=hidden_size
        )
        
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation during training."""
        predictions, labels = eval_pred
        
        # Convert to numpy arrays if they aren't already
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        if hasattr(labels, 'numpy'):
            labels = labels.numpy()
        
        # Handle the predictions format
        if len(predictions.shape) == 2:
            predictions = predictions.squeeze()
        
        # Calculate MSE, MAE, and Pearson correlation
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        
        # Calculate Pearson correlation
        try:
            correlation, p_value = pearsonr(labels, predictions)
            # Handle NaN correlation (can happen with constant predictions)
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
        except Exception:
            correlation = 0.0
            p_value = 1.0
        
        return {
            'mse': mse,
            'mae': mae,
            'pearson_correlation': correlation,
            'p_value': p_value
        }
    
    def train_model(self, model, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Train the frozen model with sensationalism head."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "temp_training"),
            num_train_epochs=self.training_kwargs.get('num_train_epochs', 10),
            per_device_train_batch_size=self.training_kwargs.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.training_kwargs.get('per_device_eval_batch_size', 8),
            logging_dir=str(self.output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=self.training_kwargs.get('learning_rate', 2e-5),
            weight_decay=self.training_kwargs.get('weight_decay', 0.01),
            warmup_steps=self.training_kwargs.get('warmup_steps', 200),
            # Enable evaluation and saving for early stopping
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=50 if eval_dataset is not None else None,
            save_strategy="steps" if eval_dataset is not None else "no",
            save_steps=50 if eval_dataset is not None else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length
        )
        
        # Prepare callbacks
        callbacks = []
        if eval_dataset is not None:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold
            )
            callbacks.append(early_stopping_callback)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset is not None else None,
            callbacks=callbacks,
        )
        
        trainer.train()
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset):
        """Evaluate frozen model with sensationalism head."""
        predictions = trainer.predict(test_dataset)
        
        # Handle different prediction formats
        if hasattr(predictions, 'predictions'):
            sensationalism_preds = predictions.predictions
        else:
            sensationalism_preds = predictions
        
        # Handle the predictions format
        if len(sensationalism_preds.shape) == 2:
            sensationalism_preds = sensationalism_preds.squeeze()
        
        # Get true labels
        sensationalism_true = test_dataset['labels'].numpy()
        
        # Calculate regression metrics
        sensationalism_mse = mean_squared_error(sensationalism_true, sensationalism_preds)
        sensationalism_mae = mean_absolute_error(sensationalism_true, sensationalism_preds)
        sensationalism_rmse = np.sqrt(sensationalism_mse)
        
        # Calculate Pearson correlation
        try:
            sensationalism_corr, sensationalism_p_value = pearsonr(sensationalism_true, sensationalism_preds)
            # Handle NaN correlation (can happen with constant predictions)
            if np.isnan(sensationalism_corr):
                sensationalism_corr = 0.0
                sensationalism_p_value = 1.0
        except Exception:
            sensationalism_corr = 0.0
            sensationalism_p_value = 1.0
        
        return {
            'sensationalism': {
                'mse': sensationalism_mse,
                'mae': sensationalism_mae,
                'rmse': sensationalism_rmse,
                'pearson_correlation': sensationalism_corr,
                'p_value': sensationalism_p_value
            }
        }
    
    def run_training(self, use_validation_split: bool = True, validation_size: float = 0.2):
        """Run the complete training pipeline for frozen multi-task sensationalism."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'sensationalism_train.csv',
            'sensationalism_test.csv'
        )
        
        # Create validation split if requested
        eval_dataset = None
        if use_validation_split:
            train_dataset, eval_dataset = self.create_validation_split(train_dataset, validation_size)
            print(f"Created validation split: {len(train_dataset)} train, {len(eval_dataset)} validation")
        
        # Prepare datasets
        if eval_dataset is not None:
            train_dataset, eval_dataset = self.prepare_datasets(train_dataset, eval_dataset)
            train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        else:
            train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        
        # Create model
        model = self.create_model()
        
        # Print model summary
        print(f"Model created with frozen base model from: {self.pretrained_model_path}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train with early stopping if validation set is available
        trainer = self.train_model(model, train_dataset, eval_dataset)
        
        # Save the model
        model_name = f"frozen_{Path(self.pretrained_model_path).name}_sensationalism"
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save only the sensationalism head and model config
        torch.save({
            'sensationalism_head.state_dict': model.sensationalism_head.state_dict(),
            'dropout.state_dict': model.dropout.state_dict(),
            'layer_norm.state_dict': model.layer_norm.state_dict(),
            'model_config': {
                'pretrained_model_path': self.pretrained_model_path,
                'hidden_size': model.sensationalism_head[1].in_features  # First Linear layer (index 1 after Dropout)
            }
        }, model_path / 'sensationalism_head.pth')
        
        # Save tokenizer for later use
        self.tokenizer.save_pretrained(model_path)
        
        # Evaluate on test set
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Sensationalism Pearson Correlation: {results['sensationalism']['pearson_correlation']:.4f}")
        print(f"Sensationalism P-value: {results['sensationalism']['p_value']:.4f}")
        print(f"Sensationalism MSE: {results['sensationalism']['mse']:.4f}")
        print(f"Sensationalism MAE: {results['sensationalism']['mae']:.4f}")
        print(f"Sensationalism RMSE: {results['sensationalism']['rmse']:.4f}")
        
        return results
    
    def create_validation_split(self, train_dataset: Dataset, val_size: float = 0.2):
        """Create validation split from training data."""
        train_test_split_dataset = train_dataset.train_test_split(
            test_size=val_size,
            seed=42
        )
        
        return train_test_split_dataset['train'], train_test_split_dataset['test']
