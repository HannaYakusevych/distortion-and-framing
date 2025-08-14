"""Prepare combined dataset for multi-task regression (causality, certainty, sensationalism)."""

import pandas as pd
from pathlib import Path
import numpy as np


def prepare_multitask_regression_data(temp_dir: str = "temp", output_dir: str = "temp"):
    """
    Combine causality, certainty, and sensationalism datasets for multi-task learning.
    
    Args:
        temp_dir: Directory containing individual task datasets
        output_dir: Directory to save combined datasets
    """
    temp_path = Path(temp_dir)
    output_path = Path(output_dir)
    
    # Load individual datasets
    print("Loading individual task datasets...")
    
    # Load causality and certainty (already combined)
    causality_certainty_train = pd.read_csv(temp_path / "causality_certainty_train.csv", index_col=0)
    causality_certainty_test = pd.read_csv(temp_path / "causality_certainty_test.csv", index_col=0)
    
    # Load sensationalism
    sensationalism_train = pd.read_csv(temp_path / "sensationalism_train.csv", index_col=0)
    sensationalism_test = pd.read_csv(temp_path / "sensationalism_test.csv", index_col=0)
    
    print(f"Causality-Certainty train: {len(causality_certainty_train)} samples")
    print(f"Causality-Certainty test: {len(causality_certainty_test)} samples")
    print(f"Sensationalism train: {len(sensationalism_train)} samples")
    print(f"Sensationalism test: {len(sensationalism_test)} samples")
    
    # Find common samples (intersection based on index/finding text)
    # First, let's check if we can match by index
    cc_train_indices = set(causality_certainty_train.index)
    cc_test_indices = set(causality_certainty_test.index)
    sens_train_indices = set(sensationalism_train.index)
    sens_test_indices = set(sensationalism_test.index)
    
    # Find intersections
    train_common_indices = cc_train_indices.intersection(sens_train_indices)
    test_common_indices = cc_test_indices.intersection(sens_test_indices)
    
    print(f"Common train samples: {len(train_common_indices)}")
    print(f"Common test samples: {len(test_common_indices)}")
    
    if len(train_common_indices) == 0 or len(test_common_indices) == 0:
        print("No common indices found. Trying to match by finding text...")
        
        # Try matching by finding text
        cc_train_findings = set(causality_certainty_train['finding'].values)
        cc_test_findings = set(causality_certainty_test['finding'].values)
        sens_train_findings = set(sensationalism_train['finding'].values)
        sens_test_findings = set(sensationalism_test['finding'].values)
        
        train_common_findings = cc_train_findings.intersection(sens_train_findings)
        test_common_findings = cc_test_findings.intersection(sens_test_findings)
        
        print(f"Common train findings: {len(train_common_findings)}")
        print(f"Common test findings: {len(test_common_findings)}")
        
        if len(train_common_findings) == 0 or len(test_common_findings) == 0:
            print("Warning: No common samples found between datasets!")
            print("Creating separate samples for each task (this will require task-specific training)")
            
            # Create combined datasets with NaN for missing tasks
            # For training set
            combined_train_list = []
            
            # Add causality-certainty samples with NaN sensationalism
            cc_train_copy = causality_certainty_train.copy()
            cc_train_copy['sensationalism'] = np.nan
            combined_train_list.append(cc_train_copy)
            
            # Add sensationalism samples with NaN causality-certainty
            sens_train_copy = sensationalism_train.copy()
            sens_train_copy['causality'] = 'Unclear'  # Default value
            sens_train_copy['certainty'] = 'Uncertain'  # Default value
            sens_train_copy['sensationalism'] = sens_train_copy['score']  # Rename score to sensationalism
            sens_train_copy = sens_train_copy.drop('score', axis=1)  # Remove original score column
            combined_train_list.append(sens_train_copy)
            
            combined_train = pd.concat(combined_train_list, ignore_index=True)
            
            # For test set
            combined_test_list = []
            
            # Add causality-certainty samples with NaN sensationalism
            cc_test_copy = causality_certainty_test.copy()
            cc_test_copy['sensationalism'] = np.nan
            combined_test_list.append(cc_test_copy)
            
            # Add sensationalism samples with NaN causality-certainty
            sens_test_copy = sensationalism_test.copy()
            sens_test_copy['causality'] = 'Unclear'  # Default value
            sens_test_copy['certainty'] = 'Uncertain'  # Default value
            sens_test_copy['sensationalism'] = sens_test_copy['score']  # Rename score to sensationalism
            sens_test_copy = sens_test_copy.drop('score', axis=1)  # Remove original score column
            combined_test_list.append(sens_test_copy)
            
            combined_test = pd.concat(combined_test_list, ignore_index=True)
            
        else:
            # Match by finding text
            # Create train dataset
            cc_train_matched = causality_certainty_train[
                causality_certainty_train['finding'].isin(train_common_findings)
            ].copy()
            sens_train_matched = sensationalism_train[
                sensationalism_train['finding'].isin(train_common_findings)
            ].copy()
            
            # Merge on finding text
            combined_train = pd.merge(
                cc_train_matched, 
                sens_train_matched[['finding', 'score']].rename(columns={'score': 'sensationalism'}), 
                on='finding', 
                how='inner'
            )
            
            # Create test dataset
            cc_test_matched = causality_certainty_test[
                causality_certainty_test['finding'].isin(test_common_findings)
            ].copy()
            sens_test_matched = sensationalism_test[
                sensationalism_test['finding'].isin(test_common_findings)
            ].copy()
            
            # Merge on finding text
            combined_test = pd.merge(
                cc_test_matched, 
                sens_test_matched[['finding', 'score']].rename(columns={'score': 'sensationalism'}), 
                on='finding', 
                how='inner'
            )
    else:
        # Match by index
        # Create train dataset
        train_indices_list = list(train_common_indices)
        cc_train_matched = causality_certainty_train.loc[train_indices_list].copy()
        sens_train_matched = sensationalism_train.loc[train_indices_list].copy()
        
        # Combine datasets
        combined_train = cc_train_matched.copy()
        combined_train['sensationalism'] = sens_train_matched['score']  # Use 'score' column
        
        # Create test dataset
        test_indices_list = list(test_common_indices)
        cc_test_matched = causality_certainty_test.loc[test_indices_list].copy()
        sens_test_matched = sensationalism_test.loc[test_indices_list].copy()
        
        # Combine datasets
        combined_test = cc_test_matched.copy()
        combined_test['sensationalism'] = sens_test_matched['score']  # Use 'score' column
    
    # Ensure sensationalism is numeric
    combined_train['sensationalism'] = pd.to_numeric(combined_train['sensationalism'], errors='coerce')
    combined_test['sensationalism'] = pd.to_numeric(combined_test['sensationalism'], errors='coerce')
    
    # Remove rows with NaN sensationalism if we want complete cases only
    print(f"Train samples before removing NaN sensationalism: {len(combined_train)}")
    print(f"Test samples before removing NaN sensationalism: {len(combined_test)}")
    
    combined_train = combined_train.dropna(subset=['sensationalism'])
    combined_test = combined_test.dropna(subset=['sensationalism'])
    
    print(f"Final train samples: {len(combined_train)}")
    print(f"Final test samples: {len(combined_test)}")
    
    # Verify we have all required columns
    required_columns = ['finding', 'causality', 'certainty', 'sensationalism']
    for col in required_columns:
        if col not in combined_train.columns:
            raise ValueError(f"Missing column {col} in train dataset")
        if col not in combined_test.columns:
            raise ValueError(f"Missing column {col} in test dataset")
    
    # Save combined datasets
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_file = output_path / "causality_certainty_sensationalism_train.csv"
    test_file = output_path / "causality_certainty_sensationalism_test.csv"
    
    combined_train.to_csv(train_file)
    combined_test.to_csv(test_file)
    
    print(f"Saved combined train dataset to: {train_file}")
    print(f"Saved combined test dataset to: {test_file}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("=" * 50)
    
    print(f"\nTrain Dataset ({len(combined_train)} samples):")
    print("Causality distribution:")
    print(combined_train['causality'].value_counts())
    print("\nCertainty distribution:")
    print(combined_train['certainty'].value_counts())
    print(f"\nSensationalism statistics:")
    print(f"  Mean: {combined_train['sensationalism'].mean():.3f}")
    print(f"  Std: {combined_train['sensationalism'].std():.3f}")
    print(f"  Min: {combined_train['sensationalism'].min():.3f}")
    print(f"  Max: {combined_train['sensationalism'].max():.3f}")
    
    print(f"\nTest Dataset ({len(combined_test)} samples):")
    print("Causality distribution:")
    print(combined_test['causality'].value_counts())
    print("\nCertainty distribution:")
    print(combined_test['certainty'].value_counts())
    print(f"\nSensationalism statistics:")
    print(f"  Mean: {combined_test['sensationalism'].mean():.3f}")
    print(f"  Std: {combined_test['sensationalism'].std():.3f}")
    print(f"  Min: {combined_test['sensationalism'].min():.3f}")
    print(f"  Max: {combined_test['sensationalism'].max():.3f}")
    
    return combined_train, combined_test


if __name__ == "__main__":
    prepare_multitask_regression_data()