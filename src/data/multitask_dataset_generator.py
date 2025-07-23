"""Multi-task dataset generation utilities for distortion and framing analysis."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset


class MultitaskDatasetGenerator:
    """Handles generation of multi-task datasets combining causality and certainty tasks."""
    
    def __init__(self, data_dir: str = "data", temp_dir: str = "temp"):
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Compressed label mappings (3 classes each)
        self.causality_label_mapping = {
            "Explicitly states: no relation": "Unclear",
            "Causation": "Causation", 
            "Correlation": "Correlation",
            "No mention of a relation": "Unclear"  # Map to "Unclear"
        }
        
        self.certainty_label_mapping = {
            "Certain": "Certain",
            "Somewhat certain": "Somewhat certain", 
            "Somewhat uncertain": "Uncertain",
            "Uncertain": "Uncertain"  # Map to "Uncertain"
        }
        
    def load_spiced_data(self) -> Tuple:
        """Load the SPICED dataset from HuggingFace."""
        spiced_ds = load_dataset("copenlu/spiced")
        return spiced_ds["train"], spiced_ds["validation"], spiced_ds["test"]
    
    def find_finding(self, key: str, train_ds, val_ds, test_ds) -> Optional[dict]:
        """Find a specific finding by ID across train/val/test splits."""
        finding = train_ds.filter(lambda example: example['instance_id'] == key)
        if not finding:
            finding = val_ds.filter(lambda example: example['instance_id'] == key)
        if not finding:
            finding = test_ds.filter(lambda example: example['instance_id'] == key)
        return finding
    
    def generate_multitask_dataset(self, spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Generate combined causality and certainty dataset with compressed labels."""
        # Load causality labels
        causality_train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/causality/train-test-splits/causality_train.tsv",
            sep='\t'
        )
        causality_test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/causality/train-test-splits/causality_test.tsv",
            sep='\t'
        )
        
        # Load certainty labels
        certainty_train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/certainty_train.tsv",
            sep='\t'
        )
        certainty_test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/certainty_test.tsv",
            sep='\t'
        )
        
        # Generate training dataset
        train_ds = self._create_combined_dataset(
            causality_train_labels, 
            certainty_train_labels,
            spiced_train_ds, 
            spiced_val_ds, 
            spiced_test_ds
        )
        
        # Generate test dataset
        test_ds = self._create_combined_dataset(
            causality_test_labels, 
            certainty_test_labels,
            spiced_train_ds, 
            spiced_val_ds, 
            spiced_test_ds
        )
        
        # Save datasets
        train_ds.to_csv(self.temp_dir / 'causality_certainty_train.csv', index=True)
        test_ds.to_csv(self.temp_dir / 'causality_certainty_test.csv', index=True)
        
        print(f"Generated multitask training dataset with {len(train_ds)} samples")
        print(f"Generated multitask test dataset with {len(test_ds)} samples")
        
        return train_ds, test_ds
    
    def _create_combined_dataset(self, causality_labels, certainty_labels, 
                               spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Create combined dataset from causality and certainty labels."""
        combined_ds = pd.DataFrame(columns=["finding", "causality", "certainty"])
        
        for _, causality_row in causality_labels.iterrows():
            key = causality_row["id"]
            
            # Find corresponding certainty labels
            certainty_rows = certainty_labels.loc[certainty_labels['id'] == key]
            
            if len(certainty_rows) == 0:
                print(f"Warning: No certainty labels found for ID {key}")
                continue
                
            certainty_row = certainty_rows.iloc[0]
            
            # Find the finding in SPICED dataset
            finding = self.find_finding(key, spiced_train_ds, spiced_val_ds, spiced_test_ds)
            
            if finding and len(finding) > 0:
                # Add paper finding
                paper_causality = self.causality_label_mapping.get(
                    causality_row["label_paper_finding"], 
                    causality_row["label_paper_finding"]
                )
                paper_certainty = self.certainty_label_mapping.get(
                    certainty_row["label_paper_finding"], 
                    certainty_row["label_paper_finding"]
                )
                
                combined_ds.loc[len(combined_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "causality": paper_causality,
                    "certainty": paper_certainty
                }
                
                # Add reported finding
                reported_causality = self.causality_label_mapping.get(
                    causality_row["label_reported_finding"], 
                    causality_row["label_reported_finding"]
                )
                reported_certainty = self.certainty_label_mapping.get(
                    certainty_row["label_reported_finding"], 
                    certainty_row["label_reported_finding"]
                )
                
                combined_ds.loc[len(combined_ds)] = {
                    "finding": finding["News Finding"][0],
                    "causality": reported_causality,
                    "certainty": reported_certainty
                }
            else:
                print(f"Warning: No finding found for ID {key}")
        
        return combined_ds
    
    def generate_all_multitask_datasets(self):
        """Generate all multi-task datasets."""
        print("Loading SPICED dataset...")
        spiced_train_ds, spiced_val_ds, spiced_test_ds = self.load_spiced_data()
        
        print("Generating multi-task causality-certainty dataset...")
        self.generate_multitask_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
        
        print("Multi-task dataset generation completed!")
    
    def get_label_statistics(self):
        """Get statistics about the generated datasets."""
        try:
            train_ds = pd.read_csv(self.temp_dir / 'causality_certainty_train.csv', index_col=0)
            test_ds = pd.read_csv(self.temp_dir / 'causality_certainty_test.csv', index_col=0)
            
            print("\n=== Dataset Statistics ===")
            print(f"Training samples: {len(train_ds)}")
            print(f"Test samples: {len(test_ds)}")
            
            print("\n=== Causality Label Distribution (Training) ===")
            print(train_ds['causality'].value_counts())
            
            print("\n=== Certainty Label Distribution (Training) ===")
            print(train_ds['certainty'].value_counts())
            
            print("\n=== Causality Label Distribution (Test) ===")
            print(test_ds['causality'].value_counts())
            
            print("\n=== Certainty Label Distribution (Test) ===")
            print(test_ds['certainty'].value_counts())
            
        except FileNotFoundError:
            print("Multi-task datasets not found. Please run generate_all_multitask_datasets() first.")


if __name__ == "__main__":
    # Example usage
    generator = MultitaskDatasetGenerator()
    generator.generate_all_multitask_datasets()
    generator.get_label_statistics()