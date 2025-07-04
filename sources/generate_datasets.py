from datasets import load_dataset
import pandas as pd

def generate_causality_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds):
    causality_train_labels = pd.read_csv("../data/aggregated-labels/causality/train-test-splits/causality_train.tsv",
                                         sep='\t')
    causality_test_labels = pd.read_csv("../data/aggregated-labels/causality/train-test-splits/causality_test.tsv",
                                        sep='\t')

    causality_train_ds = pd.DataFrame(columns=["finding", "label"])
    causality_test_ds = pd.DataFrame(columns=["finding", "label"])

    for (_, labels) in causality_train_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            causality_train_ds.loc[len(causality_train_ds)] = {
                "finding": finding["Paper Finding"][0],
                "label": labels["label_paper_finding"],
            }
            causality_train_ds.loc[len(causality_train_ds)] = {
                "finding": finding["News Finding"][0],
                "label": labels["label_reported_finding"],
            }

    for (_, labels) in causality_test_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            causality_test_ds.loc[len(causality_test_ds)] = {
                "finding": finding["Paper Finding"][0],
                "label": labels["label_paper_finding"],
            }
            causality_test_ds.loc[len(causality_test_ds)] = {
                "finding": finding["News Finding"][0],
                "label": labels["label_reported_finding"],
            }

    causality_train_ds.to_csv('../temp/causality_train.csv', index=True)
    causality_test_ds.to_csv('../temp/causality_test.csv', index=True)

def generate_certainty_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds):
    certainty_train_labels = pd.read_csv("../data/aggregated-labels/certainty/train-test-splits/certainty_train.tsv",
                                         sep='\t')
    certainty_test_labels = pd.read_csv("../data/aggregated-labels/certainty/train-test-splits/certainty_test.tsv",
                                        sep='\t')

    certainty_train_ds = pd.DataFrame(columns=["finding", "label"])
    certainty_test_ds = pd.DataFrame(columns=["finding", "label"])

    for (_, labels) in certainty_train_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            certainty_train_ds.loc[len(certainty_train_ds)] = {
                "finding": finding["Paper Finding"][0],
                "label": labels["label_paper_finding"],
            }
            certainty_train_ds.loc[len(certainty_train_ds)] = {
                "finding": finding["News Finding"][0],
                "label": labels["label_reported_finding"],
            }

    for (_, labels) in certainty_test_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            certainty_test_ds.loc[len(certainty_test_ds)] = {
                "finding": finding["Paper Finding"][0],
                "label": labels["label_paper_finding"],
            }
            certainty_test_ds.loc[len(certainty_test_ds)] = {
                "finding": finding["News Finding"][0],
                "label": labels["label_reported_finding"],
            }

    certainty_train_ds.to_csv('../temp/certainty_train.csv', index=True)
    certainty_test_ds.to_csv('../temp/certainty_test.csv', index=True)

def generate_sensationalism_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds):
    sensationalism_train_labels = pd.read_csv(
        "../data/aggregated-labels/sensationalism/train-test-splits/sensationalism_train.csv", sep='\t')
    sensationalism_test_labels = pd.read_csv(
        "../data/aggregated-labels/sensationalism/train-test-splits/sensationalism_test.csv", sep='\t')

    sensationalism_train_ds = pd.DataFrame(columns=["finding", "score"])
    sensationalism_test_ds = pd.DataFrame(columns=["finding", "score"])

    for (_, labels) in sensationalism_train_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            sensationalism_train_ds.loc[len(sensationalism_train_ds)] = {
                "finding": finding["Paper Finding"][0],
                "score": labels["score_p"],
            }
            sensationalism_train_ds.loc[len(sensationalism_train_ds)] = {
                "finding": finding["News Finding"][0],
                "score": labels["score_r"],
            }

    for (_, labels) in sensationalism_test_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            sensationalism_test_ds.loc[len(sensationalism_test_ds)] = {
                "finding": finding["Paper Finding"][0],
                "score": labels["score_p"],
            }
            sensationalism_test_ds.loc[len(sensationalism_test_ds)] = {
                "finding": finding["News Finding"][0],
                "score": labels["score_r"],
            }

    sensationalism_train_ds.to_csv('../temp/sensationalism_train.csv', index=True)
    sensationalism_test_ds.to_csv('../temp/sensationalism_test.csv', index=True)

def generate_generalization_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds):
    generalization_train_labels = pd.read_csv(
        "../data/aggregated-labels/generalization/train-test-splits/general_train.tsv", sep='\t')
    generalization_test_labels = pd.read_csv(
        "../data/aggregated-labels/generalization/train-test-splits/general_test.tsv", sep='\t')

    generalization_train_ds = pd.DataFrame(
        columns=["paper_finding", "reported_finding", "which-finding-is-more-general"])
    generalization_test_ds = pd.DataFrame(
        columns=["paper_finding", "reported_finding", "which-finding-is-more-general"])

    for (_, labels) in generalization_train_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            generalization_train_ds.loc[len(generalization_train_ds)] = {
                "paper_finding": finding["Paper Finding"][0],
                "reported_finding": finding["News Finding"][0],
                "which-finding-is-more-general": labels["which-finding-is-more-general"],
            }

    for (_, labels) in generalization_test_labels.iterrows():
        finding = find_finding(labels["id"], spiced_train_ds, spiced_val_ds, spiced_test_ds)
        if finding:
            generalization_test_ds.loc[len(generalization_test_ds)] = {
                "paper_finding": finding["Paper Finding"][0],
                "reported_finding": finding["News Finding"][0],
                "which-finding-is-more-general": labels["which-finding-is-more-general"],
            }

    generalization_train_ds.to_csv('../temp/generalization_train.csv', index=True)
    generalization_test_ds.to_csv('../temp/generalization_test.csv', index=True)

def generate_baseline_datasets():
    spiced_train_ds, spiced_val_ds, spiced_test_ds = read_spiced_data()
    generate_causality_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
    generate_certainty_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
    generate_sensationalism_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)
    generate_generalization_dataset(spiced_train_ds, spiced_val_ds, spiced_test_ds)


def read_spiced_data():
    spiced_ds = load_dataset("copenlu/spiced")
    spiced_train_ds = spiced_ds["train"]
    spiced_val_ds = spiced_ds["validation"]
    spiced_test_ds = spiced_ds["test"]
    return spiced_train_ds, spiced_val_ds, spiced_test_ds

def find_finding(key, spiced_train_ds, spiced_val_ds, spiced_test_ds):
    finding = spiced_train_ds.filter(lambda example: example['instance_id'] == key)
    if not finding:
        finding = spiced_val_ds.filter(lambda example: example['instance_id'] == key)
    if not finding:
        finding = spiced_test_ds.filter(lambda example: example['instance_id'] == key)
    return finding

if __name__ == '__main__':
    generate_baseline_datasets()