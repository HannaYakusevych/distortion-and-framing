"""Generalization classification model."""

from .base_baseline_trainer import BaseBaselineTrainer
from datasets import Dataset
import pandas as pd


class GeneralizationBaselineTrainer(BaseBaselineTrainer):
    """Trainer for generalization classification task."""
    
    def __init__(self, use_scibert: bool = False, **kwargs):
        # Set model name based on scibert flag
        if use_scibert:
            kwargs.setdefault('model_name', 'allenai/scibert_scivocab_uncased')
        
        # Use longer max_length for generalization task since it processes two texts
        kwargs.setdefault('max_length', 3072)
        super().__init__(**kwargs)
        
        self.use_scibert = use_scibert
        
        self.id2label = {
            0: "Paper Finding",
            1: "They are at the same level of generality", 
            2: "Reported Finding"
        }
        self.label2id = {
            "Paper Finding": 0,
            "They are at the same level of generality": 1,
            "Reported Finding": 2
        }
    
    def load_data(self, train_file: str, test_file: str):
        """Load and prepare generalization data with two text inputs."""
        train_dataset = pd.read_csv(self.temp_dir / train_file, index_col=0)
        train_dataset["label"] = train_dataset["which-finding-is-more-general"].map(self.label2id)
        
        test_dataset = pd.read_csv(self.temp_dir / test_file, index_col=0)
        test_dataset["label"] = test_dataset["which-finding-is-more-general"].map(self.label2id)
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_dataset)
        test_dataset = Dataset.from_pandas(test_dataset)
        
        return train_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """Tokenize function for two-text input (paper_finding and reported_finding)."""
        return self.tokenizer(
            examples["paper_finding"], 
            examples["reported_finding"],
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
    
    def run_training(self):
        """Run the complete training pipeline for generalization classification."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'generalization_train.csv',
            'generalization_test.csv'
        )
        
        # Prepare datasets
        train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        
        # Create model
        model = self.create_model(self.id2label)
        
        # Train
        trainer = self.train_model(model, train_dataset)
        
        # Save model
        model_name = 'baseline_scibert_generalization' if self.use_scibert else 'baseline_roberta_generalization'
        model_path = self.output_dir / model_name
        model.save_pretrained(model_path)
        
        # Evaluate
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Generalization F1 scores per class: {results['f1_per_class']}")
        print(f"Generalization Macro F1: {results['macro_f1']:.4f}")
        
        return results