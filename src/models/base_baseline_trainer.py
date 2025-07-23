"""Base trainer class for classification models."""

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import f1_score
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class BaseBaselineTrainer:
    """Base class for training classification models."""
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        output_dir: str = "out",
        temp_dir: str = "temp",
        max_length: int = 1536,
        **training_kwargs
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.max_length = max_length
        
        # Default training arguments
        self.training_args = {
            "num_train_epochs": 5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "logging_strategy": "steps",
            "logging_steps": 10,
        }
        self.training_args.update(training_kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_data(self, train_file: str, test_file: str, label_mapping: Dict[str, int], extra_mapping: Optional[Dict[int, int]] = None):
        """Load and preprocess training and test data."""
        train_df = pd.read_csv(self.temp_dir / train_file, index_col=0)
        test_df = pd.read_csv(self.temp_dir / test_file, index_col=0)
        
        train_df['label'] = train_df['label'].map(label_mapping)
        test_df['label'] = test_df['label'].map(label_mapping)

        if extra_mapping:
            train_df['label'] = train_df['label'].map(extra_mapping)
            test_df['label'] = test_df['label'].map(extra_mapping)
        
        return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)
    
    def tokenize_function(self, examples):
        """Tokenize input text."""
        return self.tokenizer(
            examples["finding"],
            truncation=True,
            max_length=self.max_length
        )
    
    def prepare_datasets(self, train_dataset: Dataset, test_dataset: Dataset):
        """Apply tokenization and set format for PyTorch."""
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)
        
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        return train_dataset, test_dataset
    
    def create_model(self, id2label: Dict[int, str]):
        """Create and configure the model."""
        config = AutoConfig.from_pretrained(self.model_name)
        config.update({"id2label": id2label})
        
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            config=config
        )
    
    def train_model(
        self,
        model,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        """Train the model."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            **self.training_args
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset or train_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        return trainer
    
    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset):
        """Evaluate the trained model."""
        predictions = trainer.predict(test_dataset)
        predicted_labels = predictions.predictions.argmax(axis=1)
        true_labels = test_dataset['label']
        
        f1_scores = f1_score(true_labels, predicted_labels, average=None)
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        
        return {
            'f1_per_class': f1_scores,
            'macro_f1': macro_f1,
            'predictions': predicted_labels,
            'true_labels': true_labels
        }