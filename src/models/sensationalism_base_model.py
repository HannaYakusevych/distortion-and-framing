"""Sensationalism regression model."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import pearsonr
from torch import nn
from transformers import (
    AutoConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaForRegression(RobertaForSequenceClassification):
    """RoBERTa model adapted for regression tasks."""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace classifier with regression head
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.num_labels = 1
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out unexpected kwargs that RoBERTa doesn't accept
        roberta_kwargs = {}
        valid_keys = {'token_type_ids', 'position_ids', 'head_mask', 'inputs_embeds', 
                     'output_attentions', 'output_hidden_states', 'return_dict'}
        for key, value in kwargs.items():
            if key in valid_keys:
                roberta_kwargs[key] = value
        
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **roberta_kwargs)
        # Get the pooled output (CLS token representation)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # If no pooler output, use the first token (CLS) from last hidden state
            pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.float())
            
        # Return a dictionary-like object that the Trainer can handle
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


class SensationalismBaselineTrainer:
    """Trainer for sensationalism regression task."""
    
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
        self.max_length = max_length
        self.training_kwargs = training_kwargs
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    
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
        """Tokenize function for single text input (finding)."""
        return self.tokenizer(
            examples["finding"],
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
    
    def prepare_datasets(self, train_dataset: Dataset, test_dataset: Dataset):
        """Tokenize and format datasets for training."""
        # Tokenize datasets
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, batch_size=len(train_dataset))
        test_dataset = test_dataset.map(self.tokenize_function, batched=True, batch_size=len(test_dataset))
        
        # Set format for PyTorch
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return train_dataset, test_dataset
    
    def create_model(self):
        """Create RoBERTa regression model."""
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = 1
        
        # Load pretrained model and modify for regression
        pretrained_model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        model = RobertaForRegression(config)
        
        # Copy pretrained weights (except classifier)
        model.roberta.load_state_dict(pretrained_model.roberta.state_dict())
        
        return model
    
    def train_model(self, model, train_dataset: Dataset):
        """Train the regression model."""
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
            save_strategy="no",  # Don't save intermediate checkpoints
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset):
        """Evaluate regression model using Pearson correlation."""
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.squeeze()
        y_true = test_dataset['labels'].numpy()
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(y_true, y_pred)
        
        # Calculate MSE for additional metric
        mse = np.mean((y_true - y_pred) ** 2)
        
        return {
            'pearson_correlation': correlation,
            'p_value': p_value,
            'mse': mse
        }
    
    def run_training(self):
        """Run the complete training pipeline for sensationalism regression."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'sensationalism_train.csv',
            'sensationalism_test.csv'
        )
        
        # Prepare datasets
        train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        
        # Create model
        model = self.create_model()
        
        # Train
        trainer = self.train_model(model, train_dataset)
        
        # Save model
        model_name = 'baseline_scibert_sensationalism' if self.use_scibert else 'baseline_roberta_sensationalism'
        model_path = self.output_dir / model_name
        model.save_pretrained(model_path)
        
        # Evaluate
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Sensationalism Pearson Correlation: {results['pearson_correlation']:.4f}")
        print(f"Sensationalism P-value: {results['p_value']:.4f}")
        print(f"Sensationalism MSE: {results['mse']:.4f}")
        
        return results