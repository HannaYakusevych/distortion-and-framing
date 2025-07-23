"""Evaluation utilities and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


class EvaluationUtils:
    """Utilities for model evaluation and result analysis."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_per_class': f1_score(y_true, y_pred, average=None),
            'classification_report': classification_report(y_true, y_pred, target_names=labels),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix"):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def create_results_table(results: Dict[str, Dict], tasks: List[str]) -> pd.DataFrame:
        """Create a results comparison table."""
        data = []
        for task in tasks:
            if task in results:
                row = {'Task': task}
                f1_scores = results[task]['f1_per_class']
                row.update({f'Class_{i}': f1_scores[i] for i in range(len(f1_scores))})
                row['Macro_F1'] = results[task]['macro_f1']
                data.append(row)
        
        return pd.DataFrame(data)