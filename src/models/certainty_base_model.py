"""Certainty classification model."""

from .base_baseline_trainer import BaseBaselineTrainer


class CertaintyBaselineTrainer(BaseBaselineTrainer):
    """Trainer for certainty classification task."""
    
    def __init__(self, use_compressed_data: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        self.use_compressed_data = use_compressed_data
        
        if use_compressed_data:
            self.id2label = {
                0: "Certain",
                1: "Somewhat certain",
                2: "Uncertain"
            }
            self.defaultLabel2id = {
                "Certain": 0,
                "Somewhat certain": 1, 
                "Somewhat uncertain": 2,
                "Uncertain": 2  # Map to "Uncertain"
            }
            self.extraLabel2id = {
                0: 0,
                1: 1,
                2: 2
            }
        else:
            self.id2label = {
                0: "Certain",
                1: "Somewhat certain",
                2: "Somewhat uncertain", 
                3: "Uncertain"
            }
            self.defaultLabel2id = {v: k for k, v in self.id2label.items()}
            self.extraLabel2id = None
    
    def run_training(self):
        """Run the complete training pipeline for certainty classification."""
        # Load data
        train_dataset, test_dataset = self.load_data(
            'certainty_train.csv',
            'certainty_test.csv',
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
        model_path = self.output_dir / 'baseline_roberta_certainty'
        model.save_pretrained(model_path)
        
        # Evaluate
        results = self.evaluate_model(trainer, test_dataset)
        
        print(f"Certainty F1 scores per class: {results['f1_per_class']}")
        print(f"Certainty Macro F1: {results['macro_f1']:.4f}")
        
        return results