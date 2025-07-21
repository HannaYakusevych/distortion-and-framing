"""Dataset generation utilities for distortion and framing analysis."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset


class DatasetGenerator:
    """Handles generation of datasets for different classification tasks."""
    
    def __init__(self, data_dir: str = "data", temp_dir: str = "temp"):
        self.data_dir = Path(data_dir)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
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
    
    def generate_causality_dataset(self, spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Generate causality classification dataset."""
        train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/causality/train-test-splits/causality_train.tsv",
            sep='\t'
        )
        test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/causality/train-test-splits/causality_test.tsv",
            sep='\t'
        )
        
        train_ds = pd.DataFrame(columns=["finding", "label"])
        test_ds = pd.DataFrame(columns=["finding", "label"])
        
        for _, labels in train_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "label": labels["label_paper_finding"],
                }
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["News Finding"][0],
                    "label": labels["label_reported_finding"],
                }
        
        for _, labels in test_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "label": labels["label_paper_finding"],
                }
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["News Finding"][0],
                    "label": labels["label_reported_finding"],
                }
        
        train_ds.to_csv(self.temp_dir / 'causality_train.csv', index=True)
        test_ds.to_csv(self.temp_dir / 'causality_test.csv', index=True)
        
    def generate_sensationalism_dataset(self, spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Generate sensationalism regression dataset."""
        train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/sensationalism_train.tsv",
            sep='\t'
        )
        test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/sensationalism_test.tsv",
            sep='\t'
        )
        
        train_ds = pd.DataFrame(columns=["finding", "score"])
        test_ds = pd.DataFrame(columns=["finding", "score"])
        
        for _, labels in train_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "score": labels["score_p"],
                }
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["News Finding"][0],
                    "score": labels["score_r"],
                }
        
        for _, labels in test_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "score": labels["score_p"],
                }
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["News Finding"][0],
                    "score": labels["score_r"],
                }
        
        train_ds.to_csv(self.temp_dir / 'sensationalism_train.csv', index=True)
        test_ds.to_csv(self.temp_dir / 'sensationalism_test.csv', index=True)
    
    def generate_certainty_dataset(self, spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Generate certainty classification dataset."""
        train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/certainty_train.tsv",
            sep='\t'
        )
        test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/certainty_test.tsv",
            sep='\t'
        )
        
        train_ds = pd.DataFrame(columns=["finding", "label"])
        test_ds = pd.DataFrame(columns=["finding", "label"])
        
        for _, labels in train_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "label": labels["label_paper_finding"],
                }
                train_ds.loc[len(train_ds)] = {
                    "finding": finding["News Finding"][0],
                    "label": labels["label_reported_finding"],
                }
        
        for _, labels in test_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["Paper Finding"][0],
                    "label": labels["label_paper_finding"],
                }
                test_ds.loc[len(test_ds)] = {
                    "finding": finding["News Finding"][0],
                    "label": labels["label_reported_finding"],
                }
        
        train_ds.to_csv(self.temp_dir / 'certainty_train.csv', index=True)
        test_ds.to_csv(self.temp_dir / 'certainty_test.csv', index=True)

    def generate_generality_dataset(self, spiced_train_ds, spiced_val_ds, spiced_test_ds):
        """Generate certainty classification dataset."""
        train_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/general_train.tsv",
            sep='\t'
        )
        test_labels = pd.read_csv(
            self.data_dir / "aggregated-labels/certainty/train-test-splits/general_test.tsv",
            sep='\t'
        )
        
        train_ds = pd.DataFrame(columns=["paper_finding", "reported_finding", "which-finding-is-more-general"])
        test_ds = pd.DataFrame(columns=["paper_finding", "reported_finding", "which-finding-is-more-general"])
        
        for _, labels in train_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                train_ds.loc[len(train_ds)] = {
                    "paper_finding": finding["Paper Finding"][0],
                    "reported_finding": finding["News Finding"][0],
                    "which-finding-is-more-general": labels["which-finding-is-more-general"],
                }
        
        for _, labels in test_labels.iterrows():
            finding = self.find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
            if finding:
                test_ds.loc[len(test_ds)] = {
                    "paper_finding": finding["Paper Finding"][0],
                    "reported_finding": finding["News Finding"][0],
                    "which-finding-is-more-general": labels["which-finding-is-more-general"],
                }
        
        train_ds.to_csv(self.temp_dir / 'generalization_train.csv', index=True)
        test_ds.to_csv(self.temp_dir / 'generalization_test.csv', index=True)
        
    def generate_all_datasets(self):
        """Generate all classification datasets."""
        spiced_train_ds, spiced_val_ds, spiced_test_ds = self.load_spiced_data()
        
        self.generate_causality_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
        self.generate_certainty_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
        self.generate_sensationalism_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
        self.generate_generality_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)