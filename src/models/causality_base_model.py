"""Causality classification model."""

from .base_baseline_trainer import BaseBaselineTrainer
from pathlib import Path


class CausalityBaselineTrainer(BaseBaselineTrainer):
    """Trainer for causality classification task."""
    
    def __init__(self, use_compressed_data: bool = False, use_scibert: bool = False, **kwargs):
        # Set model name based on scibert flag
        if use_scibert:
            kwargs.setdefault('model_name', 'allenai/scibert_scivocab_uncased')
        
        super().__init__(**kwargs)
        
        self.use_compressed_data = use_compressed_data
        self.use_scibert = use_scibert
        
        if use_compressed_data:
            # For compressed data, use 3 classes
            self.id2label = {
                0: "Unclear",
                1: "Causation", 
                2: "Correlation"
            }
            self.defaultLabel2id = {
                "Explicitly states: no relation": 0,
                "Causation": 1, 
                "Correlation": 2,
                "No mention of a relation": 0  # Map to "Unclear"
            }
            self.extraLabel2id = {
                0: 0,
                1: 1,
                2: 2
            }
        else:
            # For full data, use 4 classes
            self.id2label = {
                0: "Explicitly states: no relation",
                1: "Causation", 
                2: "Correlation",
                3: "No mention of a relation"
            }
            self.defaultLabel2id = {v: k for k, v in self.id2label.items()}
            self.extraLabel2id = None
    
    def run_training(self):
        """Run the complete training pipeline for causality classification."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'causality_train.csv',
            'causality_test.csv', 
            self.defaultLabel2id,
            self.extraLabel2id
        )
        
        # Prepare datasets
        train_dataset, test_dataset = self.prepare_datasets(train_dataset, test_dataset)
        
        # Create model
        model = self.create_model(self.id2label)
        
        # Train
        trainer = self.train_model(model, train_dataset)
        
        # Save model
        model_name = 'baseline_scibert_causality' if self.use_scibert else 'baseline_roberta_causality'
        model_path = self.output_dir / model_name
        model.save_pretrained(model_path)
        
        # Evaluate
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Causality F1 scores per class: {results['f1_per_class']}")
        print(f"Causality Macro F1: {results['macro_f1']:.4f}")
        
        return results