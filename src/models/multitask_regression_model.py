"""Multi-task model for causality, certainty classification and sensationalism regression."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import random
import itertools

import pandas as pd
import numpy as np
import torch
import warnings
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
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


class MultiTaskRegressionModel(nn.Module):
    """Multi-task model for causality, certainty classification and sensationalism regression."""
    
    def __init__(self, model_name: str, causality_num_labels: int = 3, certainty_num_labels: int = 3, 
                 loss_balancing: str = "fixed"):
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
        self.sensationalism_regressor = nn.Linear(hidden_size, 1)  # Regression head for sensationalism
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # Store number of labels for each task
        self.causality_num_labels = causality_num_labels
        self.certainty_num_labels = certainty_num_labels
        self.num_labels = causality_num_labels + certainty_num_labels + 1  # +1 for regression
        
        # Loss balancing strategy
        self.loss_balancing = loss_balancing
        
        # For adaptive loss balancing, track running averages of loss magnitudes
        if loss_balancing == "adaptive":
            self.register_buffer('causality_loss_avg', torch.tensor(1.0))
            self.register_buffer('certainty_loss_avg', torch.tensor(1.0))
            self.register_buffer('sensationalism_loss_avg', torch.tensor(0.1))
            self.register_buffer('update_count', torch.tensor(0))
            self.momentum = 0.9  # For exponential moving average
        elif loss_balancing == "gradnorm":
            # GradNorm: learnable task weights (Chen et al., 2018)
            self.task_weights = nn.Parameter(torch.ones(3))  # 3 tasks
            self.register_buffer('initial_losses', torch.zeros(3))
            self.register_buffer('gradnorm_initialized', torch.tensor(0.0))  # Use 0.0 for False, 1.0 for True
            self.alpha = 1.5  # GradNorm hyperparameter
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing if supported by base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            try:
                # Always set use_reentrant=False to avoid gradient computation issues
                kwargs['use_reentrant'] = False
                self.base_model.gradient_checkpointing_enable(**kwargs)
            except TypeError:
                # Fallback for models that don't accept kwargs
                try:
                    self.base_model.gradient_checkpointing_enable()
                except:
                    pass  # Ignore if gradient checkpointing is not supported
    
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
                causality_labels=None, certainty_labels=None, sensationalism_labels=None, **kwargs):
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
        sensationalism_logits = self.sensationalism_regressor(pooled_output).squeeze(-1)  # Remove last dimension
        
        # Calculate losses if labels are provided
        total_loss = None
        classification_loss_fct = nn.CrossEntropyLoss()
        regression_loss_fct = nn.MSELoss()
        
        losses = []
        loss_weights = []
        
        if causality_labels is not None:
            causality_loss = classification_loss_fct(
                causality_logits.view(-1, self.causality_num_labels), 
                causality_labels.view(-1)
            )
            losses.append(causality_loss)
            loss_weights.append(1.0)  # Standard weight for classification
        
        if certainty_labels is not None:
            certainty_loss = classification_loss_fct(
                certainty_logits.view(-1, self.certainty_num_labels), 
                certainty_labels.view(-1)
            )
            losses.append(certainty_loss)
            loss_weights.append(1.0)  # Standard weight for classification
        
        if sensationalism_labels is not None:
            sensationalism_loss = regression_loss_fct(
                sensationalism_logits.view(-1), 
                sensationalism_labels.view(-1).float()
            )
            losses.append(sensationalism_loss)
            # Scale regression loss to be comparable to classification losses
            # Classification losses are typically in range [0, 2], MSE can be much smaller
            # We'll use a scaling factor to balance them
            loss_weights.append(10.0)  # Scale up regression loss
        
        # Combine losses with balanced weighting
        if losses:
            if self.loss_balancing == "adaptive" and self.training:
                # Simple adaptive loss balancing based on running averages (heuristic)
                total_loss = self._adaptive_loss_combination(losses, causality_labels, certainty_labels, sensationalism_labels)
            elif self.loss_balancing == "gradnorm" and self.training:
                # GradNorm: Gradient Normalization (Chen et al., 2018)
                total_loss = self._gradnorm_loss_combination(losses, causality_labels, certainty_labels, sensationalism_labels)
            else:
                # Fixed loss balancing with manual scaling
                weighted_losses = [loss * weight for loss, weight in zip(losses, loss_weights)]
                total_loss = sum(weighted_losses) / len(weighted_losses)  # Average to prevent scaling issues
        
        # Combine logits for compatibility - concatenate classification logits and regression output
        combined_logits = torch.cat([
            causality_logits, 
            certainty_logits, 
            sensationalism_logits.unsqueeze(-1)  # Add dimension back for concatenation
        ], dim=-1)
        
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=combined_logits,
            hidden_states=None,
            attentions=None,
        )
    
    def _adaptive_loss_combination(self, losses, causality_labels, certainty_labels, sensationalism_labels):
        """Adaptively combine losses based on their relative magnitudes."""
        causality_loss = losses[0] if causality_labels is not None else None
        certainty_loss = losses[1] if certainty_labels is not None else None
        sensationalism_loss = losses[2] if sensationalism_labels is not None else None
        
        # Update running averages
        self.update_count += 1
        
        if causality_loss is not None:
            self.causality_loss_avg = self.momentum * self.causality_loss_avg + (1 - self.momentum) * causality_loss.detach()
        if certainty_loss is not None:
            self.certainty_loss_avg = self.momentum * self.certainty_loss_avg + (1 - self.momentum) * certainty_loss.detach()
        if sensationalism_loss is not None:
            self.sensationalism_loss_avg = self.momentum * self.sensationalism_loss_avg + (1 - self.momentum) * sensationalism_loss.detach()
        
        # Calculate adaptive weights (inverse of running average to balance magnitudes)
        total_loss = 0
        num_tasks = 0
        
        if causality_loss is not None:
            weight = 1.0 / (self.causality_loss_avg + 1e-8)
            total_loss += causality_loss * weight
            num_tasks += 1
            
        if certainty_loss is not None:
            weight = 1.0 / (self.certainty_loss_avg + 1e-8)
            total_loss += certainty_loss * weight
            num_tasks += 1
            
        if sensationalism_loss is not None:
            weight = 1.0 / (self.sensationalism_loss_avg + 1e-8)
            total_loss += sensationalism_loss * weight
            num_tasks += 1
        
        return total_loss / num_tasks if num_tasks > 0 else None
    
    def _gradnorm_loss_combination(self, losses, causality_labels, certainty_labels, sensationalism_labels):
        """
        GradNorm loss balancing (Chen et al., 2018).
        "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
        """
        # Initialize on first call
        if self.gradnorm_initialized.item() == 0.0:  # Check if not initialized (0.0 = False)
            current_losses = torch.stack([loss.detach() for loss in losses])
            self.initial_losses.copy_(current_losses)
            self.gradnorm_initialized.fill_(1.0)  # Set to 1.0 (True)
        
        # Get current losses
        current_losses = torch.stack([loss.detach() for loss in losses])
        
        # Calculate relative loss ratios
        loss_ratios = current_losses / (self.initial_losses + 1e-8)
        
        # Target: average loss ratio
        target_ratio = loss_ratios.mean()
        
        # Calculate weighted loss
        weighted_losses = [loss * weight for loss, weight in zip(losses, self.task_weights)]
        total_loss = sum(weighted_losses)
        
        # Simplified GradNorm-inspired update
        # Full GradNorm requires custom optimizer integration, so we use a simplified approach
        if self.training and len(losses) > 1:
            try:
                # Simple adaptive weighting based on loss magnitudes
                # This avoids gradient computation issues during training
                with torch.no_grad():
                    # Get relative loss magnitudes
                    loss_values = torch.stack([loss.detach() for loss in losses])
                    loss_ratios = loss_values / (self.initial_losses + 1e-8)
                    
                    # Target: balanced loss ratios (all should be close to mean)
                    target_ratio = loss_ratios.mean()
                    
                    # Adjust weights to balance loss ratios
                    # If a task has higher loss ratio, reduce its weight
                    weight_adjustments = target_ratio / (loss_ratios + 1e-8)
                    
                    # Smooth update to avoid sudden changes
                    new_weights = 0.9 * self.task_weights.data + 0.1 * weight_adjustments
                    self.task_weights.data = torch.clamp(new_weights, min=0.1, max=10.0)
                            
            except Exception as e:
                # If update fails, fall back to equal weighting
                print(f"GradNorm-inspired update failed: {e}, using equal weights")
                with torch.no_grad():
                    self.task_weights.data.fill_(1.0)
        
        return total_loss


@dataclass
class MultiTaskRegressionDataCollator:
    """Data collator that properly handles multi-task labels including regression."""
    
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = "max_length"
    max_length: int = None
    pad_to_multiple_of: int = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract labels before padding
        labels = []
        if "labels" in features[0]:
            labels = [feature.pop("labels") for feature in features]
        
        # Use the specified padding strategy
        padding_strategy = self.padding
        
        # Use standard padding for input features
        batch = self.tokenizer.pad(
            features,
            padding=padding_strategy,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Add labels back as a proper tensor
        if labels:
            # Convert labels to consistent float32 format
            labels_array = np.array(labels, dtype=np.float32)
            batch["labels"] = torch.tensor(labels_array, dtype=torch.float32)
        
        return batch


class EarlyStoppingMultiTaskRegressionCallback(TrainerCallback):
    """Custom early stopping callback for multi-task learning with regression."""
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.best_model_checkpoint = None
        self.patience_counter = 0
        
    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Called after evaluation."""
        if logs is None:
            return
            
        # Get the combined metric (average of causality F1, certainty F1, and negative sensationalism MSE)
        current_metric = logs.get("eval_combined_score", None)
        if current_metric is None:
            return
            
        # Check if this is the best model so far
        if self.best_metric is None or current_metric > self.best_metric + self.early_stopping_threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
            # Save the best model checkpoint path
            self.best_model_checkpoint = f"{args.output_dir}/checkpoint-{state.global_step}"
            print(f"New best model found with combined score: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(f"No improvement for {self.patience_counter} evaluations")
            
        # Stop training if patience exceeded
        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            control.should_training_stop = True


class MultitaskRegressionTrainer:
    """Trainer for multi-task causality, certainty classification and sensationalism regression."""
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 output_dir: str = "out", 
                 temp_dir: str = "temp",
                 max_length: int = 1536,
                 use_scibert: bool = False,
                 early_stopping_patience: int = 3,
                 early_stopping_threshold: float = 0.001,
                 loss_balancing: str = "fixed",
                 **training_kwargs):
        """
        Initialize multi-task regression trainer.
        
        Args:
            loss_balancing: Loss balancing strategy:
                - "fixed": Manual weights (simple but effective)
                - "adaptive": Running average heuristic (custom implementation)
                - "gradnorm": GradNorm (Chen et al., 2018) - theoretically grounded
        """
        
        # Suppress common warnings
        warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")
        warnings.filterwarnings("ignore", message=".*max_length.*is ignored when.*padding.*True.*")
        
        # Set model name based on scibert flag
        if use_scibert:
            model_name = 'allenai/scibert_scivocab_uncased'
        
        self.model_name = model_name
        self.use_scibert = use_scibert
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.training_kwargs = training_kwargs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.loss_balancing = loss_balancing
        
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
        """Load and prepare multi-task data with regression."""
        train_dataset = pd.read_csv(self.temp_dir / train_file, index_col=0)
        test_dataset = pd.read_csv(self.temp_dir / test_file, index_col=0)
        
        # Map classification labels to IDs
        train_dataset["causality_labels"] = train_dataset["causality"].map(self.causality_label2id)
        train_dataset["certainty_labels"] = train_dataset["certainty"].map(self.certainty_label2id)
        
        test_dataset["causality_labels"] = test_dataset["causality"].map(self.causality_label2id)
        test_dataset["certainty_labels"] = test_dataset["certainty"].map(self.certainty_label2id)
        
        # Sensationalism is already numeric (regression target)
        train_dataset["sensationalism_labels"] = train_dataset["sensationalism"].astype(float)
        test_dataset["sensationalism_labels"] = test_dataset["sensationalism"].astype(float)
        
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
        
        # Create combined labels for the Trainer (it expects a single 'labels' field)
        def add_combined_labels(examples):
            # Stack causality, certainty, and sensationalism labels
            causality_labels = examples['causality_labels']
            certainty_labels = examples['certainty_labels']
            sensationalism_labels = examples['sensationalism_labels']
            
            # Ensure all labels are float32 to avoid dtype mixing
            combined = [[float(c), float(cert), float(sens)] for c, cert, sens in zip(causality_labels, certainty_labels, sensationalism_labels)]
            examples['labels'] = combined
            return examples
        
        train_dataset = train_dataset.map(add_combined_labels, batched=True)
        test_dataset = test_dataset.map(add_combined_labels, batched=True)
        
        # Set format for PyTorch
        columns = ["input_ids", "attention_mask", "labels", "causality_labels", "certainty_labels", "sensationalism_labels"]
        train_dataset.set_format("torch", columns=columns)
        test_dataset.set_format("torch", columns=columns)
        
        return train_dataset, test_dataset
    
    def create_model(self, loss_balancing: str = "fixed"):
        """Create multi-task model with regression.
        
        Args:
            loss_balancing: Strategy for balancing losses:
                - "fixed": Manual scaling (Classification: 1.0, Regression: 10.0)
                - "adaptive": Simple heuristic based on running loss averages
                - "gradnorm": GradNorm algorithm (Chen et al., 2018)
        """
        model = MultiTaskRegressionModel(
            model_name=self.model_name,
            causality_num_labels=len(self.causality_id2label),
            certainty_num_labels=len(self.certainty_id2label),
            loss_balancing=loss_balancing
        )
        
        # Disable gradient checkpointing for multitask regression to avoid conflicts
        # Gradient checkpointing can cause issues with complex loss computations
        if hasattr(model, 'gradient_checkpointing_disable'):
            try:
                model.gradient_checkpointing_disable()
            except (TypeError, AttributeError):
                pass
        
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation during training."""
        predictions, labels = eval_pred
        
        # Convert to numpy arrays if they aren't already
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        # Split the combined logits back into task-specific outputs
        causality_logits = predictions[:, :len(self.causality_id2label)]
        certainty_logits = predictions[:, len(self.causality_id2label):len(self.causality_id2label)+len(self.certainty_id2label)]
        sensationalism_preds = predictions[:, -1]  # Last column is regression output
        
        # Get classification predictions
        causality_preds = causality_logits.argmax(axis=1)
        certainty_preds = certainty_logits.argmax(axis=1)
        
        # Handle the tuple format - labels come as (causality_array, certainty_array, sensationalism_array)
        try:
            if isinstance(labels, tuple) and len(labels) == 3:
                # Labels are a tuple of (causality_labels, certainty_labels, sensationalism_labels)
                causality_true, certainty_true, sensationalism_true = labels
                
                # Convert to numpy if needed
                if hasattr(causality_true, 'numpy'):
                    causality_true = causality_true.numpy()
                if hasattr(certainty_true, 'numpy'):
                    certainty_true = certainty_true.numpy()
                if hasattr(sensationalism_true, 'numpy'):
                    sensationalism_true = sensationalism_true.numpy()
                    
            elif hasattr(labels, 'numpy'):
                # Convert tensor to numpy first
                labels = labels.numpy()
                if len(labels.shape) == 2 and labels.shape[1] == 3:
                    causality_true = labels[:, 0].astype(int)
                    certainty_true = labels[:, 1].astype(int)
                    sensationalism_true = labels[:, 2].astype(float)
                else:
                    print(f"Warning: Unexpected label shape: {labels.shape}")
                    return {'combined_score': 0.0, 'causality_f1': 0.0, 'certainty_f1': 0.0, 'sensationalism_mse': "Infinity"}
            else:
                print(f"Warning: Unexpected label format: {type(labels)}")
                return {'combined_score': 0.0, 'causality_f1': 0.0, 'certainty_f1': 0.0, 'sensationalism_mse': "Infinity"}
            
            # Calculate F1 scores for classification tasks
            causality_f1 = f1_score(causality_true, causality_preds, average='macro')
            certainty_f1 = f1_score(certainty_true, certainty_preds, average='macro')
            
            # Calculate MSE, MAE, and Pearson correlation for regression task
            sensationalism_mse = mean_squared_error(sensationalism_true, sensationalism_preds)
            sensationalism_mae = mean_absolute_error(sensationalism_true, sensationalism_preds)
            
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
            
            # Combined score: average of F1 scores and correlation (all metrics 0-1 scale)
            # Use correlation instead of MSE since correlation is more interpretable
            combined_score = (causality_f1 + certainty_f1 + abs(sensationalism_corr)) / 3
            
            return {
                'causality_f1': causality_f1,
                'certainty_f1': certainty_f1,
                'sensationalism_mse': sensationalism_mse,
                'sensationalism_mae': sensationalism_mae,
                'sensationalism_correlation': sensationalism_corr,
                'sensationalism_p_value': sensationalism_p_value,
                'combined_score': combined_score
            }
            
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            print(f"Labels type: {type(labels)}")
            if isinstance(labels, tuple):
                print(f"Tuple length: {len(labels)}")
                for i, label in enumerate(labels):
                    print(f"Element {i} type: {type(label)}, shape: {getattr(label, 'shape', 'no shape')}")
            else:
                print(f"Labels shape: {getattr(labels, 'shape', 'no shape')}")
            print(f"Predictions shape: {predictions.shape}")
            return {'combined_score': 0.0, 'causality_f1': 0.0, 'certainty_f1': 0.0, 'sensationalism_mse': "Infinity"}

    def train_model(self, model, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Train the multi-task model with early stopping."""
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
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=50 if eval_dataset is not None else None,
            save_strategy="steps" if eval_dataset is not None else "no",
            save_steps=50 if eval_dataset is not None else None,
            save_total_limit=3,  # Keep only the 3 most recent checkpoints
            load_best_model_at_end=True if eval_dataset is not None else False,
            metric_for_best_model="combined_score" if eval_dataset is not None else None,
            greater_is_better=True,
        )
        
        # Create custom data collator for multi-task learning with regression
        data_collator = MultiTaskRegressionDataCollator(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length
        )
        
        # Prepare callbacks
        callbacks = []
        if eval_dataset is not None:
            early_stopping_callback = EarlyStoppingMultiTaskRegressionCallback(
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
        
        # Load the best model if early stopping was used
        if eval_dataset is not None and callbacks:
            best_checkpoint = callbacks[0].best_model_checkpoint
            if best_checkpoint and Path(best_checkpoint).exists():
                print(f"Loading best model from {best_checkpoint}")
                trainer.model.load_state_dict(
                    torch.load(Path(best_checkpoint) / "pytorch_model.bin", map_location="cpu")
                )
        
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset):
        """Evaluate multi-task model with regression."""
        predictions = trainer.predict(test_dataset)
        
        # Handle different prediction formats
        if hasattr(predictions, 'predictions'):
            combined_logits = predictions.predictions
        else:
            combined_logits = predictions
        
        # Split the combined logits back into task-specific outputs
        causality_logits = combined_logits[:, :len(self.causality_id2label)]
        certainty_logits = combined_logits[:, len(self.causality_id2label):len(self.causality_id2label)+len(self.certainty_id2label)]
        sensationalism_preds = combined_logits[:, -1]  # Last column is regression output
        
        # Get classification predictions
        causality_preds = causality_logits.argmax(axis=1)
        certainty_preds = certainty_logits.argmax(axis=1)
        
        # Get true labels
        causality_true = test_dataset['causality_labels'].numpy()
        certainty_true = test_dataset['certainty_labels'].numpy()
        sensationalism_true = test_dataset['sensationalism_labels'].numpy()
        
        # Calculate F1 scores for classification
        causality_f1_per_class = f1_score(causality_true, causality_preds, average=None)
        causality_macro_f1 = f1_score(causality_true, causality_preds, average='macro')
        
        certainty_f1_per_class = f1_score(certainty_true, certainty_preds, average=None)
        certainty_macro_f1 = f1_score(certainty_true, certainty_preds, average='macro')
        
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
            'causality': {
                'f1_per_class': causality_f1_per_class.tolist(),
                'macro_f1': causality_macro_f1
            },
            'certainty': {
                'f1_per_class': certainty_f1_per_class.tolist(),
                'macro_f1': certainty_macro_f1
            },
            'sensationalism': {
                'mse': sensationalism_mse,
                'mae': sensationalism_mae,
                'rmse': sensationalism_rmse,
                'pearson_correlation': sensationalism_corr,
                'p_value': sensationalism_p_value
            }
        }
    
    def run_training(self, use_validation_split: bool = True, validation_size: float = 0.2):
        """Run the complete training pipeline for multi-task classification and regression."""
        # Load data - assuming we have a combined dataset with all three tasks
        train_dataset, test_dataset = self.load_data(
            'causality_certainty_sensationalism_train.csv',  # This file needs to be created
            'causality_certainty_sensationalism_test.csv'   # This file needs to be created
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
        model = self.create_model(loss_balancing=self.loss_balancing)
        
        # Train with early stopping if validation set is available
        trainer = self.train_model(model, train_dataset, eval_dataset)
        
        # Save the best model
        model_name = 'baseline_scibert_multitask_regression' if self.use_scibert else 'baseline_roberta_multitask_regression'
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path / 'pytorch_model.bin')
        
        # Save tokenizer and config for later use
        self.tokenizer.save_pretrained(model_path)
        
        # Evaluate on test set
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Causality F1 scores per class: {results['causality']['f1_per_class']}")
        print(f"Causality Macro F1: {results['causality']['macro_f1']:.4f}")
        print(f"Certainty F1 scores per class: {results['certainty']['f1_per_class']}")
        print(f"Certainty Macro F1: {results['certainty']['macro_f1']:.4f}")
        print(f"Sensationalism Pearson Correlation: {results['sensationalism']['pearson_correlation']:.4f}")
        print(f"Sensationalism P-value: {results['sensationalism']['p_value']:.4f}")
        print(f"Sensationalism MSE: {results['sensationalism']['mse']:.4f}")
        print(f"Sensationalism MAE: {results['sensationalism']['mae']:.4f}")
        print(f"Sensationalism RMSE: {results['sensationalism']['rmse']:.4f}")
        
        return results
    
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
    
    def _convert_to_json_serializable(self, obj):
        """Convert NumPy types and other non-JSON-serializable types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.inf or obj == float('inf'):
            return "Infinity"
        elif obj is -np.inf or obj == float('-inf'):
            return "-Infinity"
        elif obj != obj:  # Check for NaN
            return "NaN"
        else:
            return obj
    
    def get_hyperparameter_space(self, memory_safe: bool = False) -> Dict[str, List[Any]]:
        """Define hyperparameter search space for multi-task regression."""
        if memory_safe:
            # Conservative settings to avoid OOM
            return {
                'learning_rate': [1e-5, 2e-5, 3e-5],
                'per_device_train_batch_size': [4, 8],  # Smaller batch sizes
                'weight_decay': [0.0, 0.01],
                'warmup_steps': [100, 200],
                'max_length': [256, 512, 1024],  # Shorter sequences
                'loss_balancing': ['fixed', 'adaptive']  # Loss balancing strategies
            }
        else:
            # Full search space
            return {
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'per_device_train_batch_size': [4, 8],
                'weight_decay': [0.0, 0.01, 0.1],
                'warmup_steps': [100, 200, 500],
                'max_length': [256, 512],
                'loss_balancing': ['fixed', 'adaptive', 'gradnorm']  # All loss balancing strategies
            }
    
    def is_memory_intensive_config(self, hyperparams: Dict[str, Any]) -> bool:
        """Check if hyperparameter configuration is likely to cause OOM."""
        batch_size = hyperparams.get('per_device_train_batch_size', 8)
        max_length = hyperparams.get('max_length', 1536)
        
        # Consider configurations with large batch size and long sequences as risky
        memory_score = batch_size * max_length
        return memory_score > 16 * 1024  # Threshold for risky configurations
    
    def evaluate_hyperparameters(self, hyperparams: Dict[str, Any], 
                                train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, float]:
        """Evaluate a specific hyperparameter configuration with early stopping."""
        original_max_length = self.max_length
        original_loss_balancing = self.loss_balancing
        
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
            
            # Update loss balancing strategy if specified
            if 'loss_balancing' in hyperparams:
                self.loss_balancing = hyperparams['loss_balancing']
            
            # Update max_length if specified and re-tokenize datasets
            if 'max_length' in hyperparams:
                self.max_length = hyperparams['max_length']
                train_dataset, val_dataset = self._retokenize_datasets(train_dataset, val_dataset)
            
            # Create model with specified loss balancing
            model = self.create_model(loss_balancing=self.loss_balancing)
            
            # Create training arguments with hyperparameters and early stopping
            training_args = self._create_training_args_with_early_stopping(temp_kwargs)
            
            # Create custom data collator for multi-task regression
            data_collator = MultiTaskRegressionDataCollator(
                tokenizer=self.tokenizer,
                padding="max_length",
                max_length=self.max_length
            )
            
            # Create early stopping callback
            early_stopping_callback = EarlyStoppingMultiTaskRegressionCallback(
                early_stopping_patience=2,  # Shorter patience for hyperparameter search
                early_stopping_threshold=self.early_stopping_threshold
            )
            
            # Create trainer with early stopping
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[early_stopping_callback],
            )
            
            # Train model
            trainer.train()
            
            # Load best model if available
            if early_stopping_callback.best_model_checkpoint and Path(early_stopping_callback.best_model_checkpoint).exists():
                trainer.model.load_state_dict(
                    torch.load(Path(early_stopping_callback.best_model_checkpoint) / "pytorch_model.bin", map_location="cpu")
                )
            
            # Evaluate on validation set
            results = self.evaluate_model(trainer, val_dataset)
            
            # Calculate combined score (average of causality F1, certainty F1, and correlation)
            combined_score = (results['causality']['macro_f1'] + 
                            results['certainty']['macro_f1'] + 
                            abs(results['sensationalism']['pearson_correlation'])) / 3
            
            # Clean up
            del model, trainer
            self.clear_gpu_memory()
            
            return {
                'combined_score': float(combined_score),
                'causality_f1': float(results['causality']['macro_f1']),
                'certainty_f1': float(results['certainty']['macro_f1']),
                'sensationalism_correlation': float(results['sensationalism']['pearson_correlation']),
                'sensationalism_mse': float(results['sensationalism']['mse']),
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
                    'sensationalism_correlation': 0.0,
                    'sensationalism_mse': "Infinity",  # Use string for JSON serialization
                    'hyperparams': hyperparams,
                    'error': 'CUDA_OOM'
                }
            raise e
        finally:
            # Always restore original settings
            self.max_length = original_max_length
            self.loss_balancing = original_loss_balancing
    
    def _retokenize_datasets(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Re-tokenize datasets with current max_length."""
        batch_size = min(1000, len(train_dataset))
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        batch_size = min(1000, len(val_dataset))
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, batch_size=batch_size)
        
        # Create combined labels for the Trainer
        def add_combined_labels(examples):
            # Stack causality, certainty, and sensationalism labels
            causality_labels = examples['causality_labels']
            certainty_labels = examples['certainty_labels']
            sensationalism_labels = examples['sensationalism_labels']
            
            # Ensure all labels are float32 to avoid dtype mixing
            combined = [[float(c), float(cert), float(sens)] for c, cert, sens in zip(causality_labels, certainty_labels, sensationalism_labels)]
            examples['labels'] = combined
            return examples
        
        train_dataset = train_dataset.map(add_combined_labels, batched=True)
        val_dataset = val_dataset.map(add_combined_labels, batched=True)
        
        # Set format for PyTorch
        columns = ["input_ids", "attention_mask", "labels", "causality_labels", "certainty_labels", "sensationalism_labels"]
        train_dataset.set_format("torch", columns=columns)
        val_dataset.set_format("torch", columns=columns)
        
        return train_dataset, val_dataset
    
    def _create_training_args_with_early_stopping(self, temp_kwargs: Dict[str, Any]) -> TrainingArguments:
        """Create training arguments with hyperparameters and early stopping."""
        return TrainingArguments(
            output_dir=str(self.output_dir / "temp_hyperparam_training"),
            num_train_epochs=temp_kwargs.get('num_train_epochs', 10),
            per_device_train_batch_size=temp_kwargs.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=temp_kwargs.get('per_device_eval_batch_size', 8),
            logging_dir=str(self.output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=50,
            learning_rate=temp_kwargs.get('learning_rate', 2e-5),
            weight_decay=temp_kwargs.get('weight_decay', 0.01),
            warmup_steps=temp_kwargs.get('warmup_steps', 200),
            evaluation_strategy="steps",
            eval_steps=100,  # More frequent evaluation for early stopping
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="combined_score",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard for hyperparameter search
            dataloader_pin_memory=False,  # Reduce memory usage
            gradient_checkpointing=False,  # Disable for stability during hyperparameter search
        )
    
    def grid_search_hyperparameters(self, max_combinations: int = 50, memory_safe: bool = False) -> Dict[str, Any]:
        """Perform grid search over hyperparameter space."""
        print("Starting hyperparameter grid search for multi-task regression...")
        
        # Load and prepare data
        train_dataset, test_dataset = self.load_data(
            'causality_certainty_sensationalism_train.csv',
            'causality_certainty_sensationalism_test.csv'
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
                print(f"  Combined Score: {result['combined_score']:.4f} "
                      f"(Causality F1: {result['causality_f1']:.4f}, "
                      f"Certainty F1: {result['certainty_f1']:.4f}, "
                      f"Sensationalism Corr: {result['sensationalism_correlation']:.4f})")
                
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
                    'sensationalism_correlation': 0.0,
                    'sensationalism_mse': "Infinity",
                    'hyperparams': hyperparams,
                    'error': str(e)[:200]
                })
                continue
        
        # Save results - convert NumPy types to JSON-serializable types
        def convert_numpy_types(obj):
            """Convert NumPy types to JSON-serializable types."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity"
            elif obj != obj:  # Check for NaN
                return "NaN"
            else:
                return obj
        
        results_file = self.output_dir / 'hyperparameter_search_results_regression.json'
        with open(results_file, 'w') as f:
            json.dump({
                'best_hyperparams': best_hyperparams,
                'best_score': float(best_score) if best_score is not None else None,
                'all_results': convert_numpy_types(all_results)
            }, f, indent=2)
        
        print(f"\nHyperparameter search completed!")
        print(f"Successful runs: {successful_runs}/{len(all_combinations)}")
        print(f"CUDA OOM errors: {oom_errors}")
        print(f"Other errors: {other_errors}")
        print(f"Best hyperparameters: {best_hyperparams}")
        print(f"Best combined score: {best_score:.4f}")
        print(f"Results saved to: {results_file}")
        
        # Print memory optimization suggestions
        if oom_errors > 0:
            print("\nMemory Optimization Suggestions:")
            print("- Consider reducing max batch size further (current max: 8)")
            print("- Consider reducing max sequence length (current max: 1536)")
            print("- Try memory_safe=True for more conservative hyperparameter space")
        
        return {
            'best_hyperparams': best_hyperparams,
            'best_score': best_score,
            'all_results': all_results,
            'successful_runs': successful_runs,
            'total_runs': len(all_combinations)
        }
    
    def run_training_with_best_hyperparams(self) -> Dict[str, Any]:
        """Run training with the best hyperparameters found from search."""
        # Load hyperparameter search results
        results_file = self.output_dir / 'hyperparameter_search_results_regression.json'
        
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
        
        # Update loss balancing strategy if specified
        if 'loss_balancing' in best_hyperparams:
            self.loss_balancing = best_hyperparams['loss_balancing']
        
        # Run normal training with optimized hyperparameters
        return self.run_training()
    
    def run_hyperparameter_optimization(self, strategy: str = "grid_search", max_combinations: int = 20) -> Dict[str, Any]:
        """Run hyperparameter optimization with specified strategy."""
        if strategy == "grid_search":
            return self.grid_search_hyperparameters(max_combinations=max_combinations)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert NumPy types and other non-JSON-serializable types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.inf or obj == float('inf'):
            return "Infinity"
        elif obj is -np.inf or obj == float('-inf'):
            return "-Infinity"
        elif obj != obj:  # Check for NaN
            return "NaN"
        else:
            return obj