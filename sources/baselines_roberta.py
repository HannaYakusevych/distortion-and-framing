import pandas as pd
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import f1_score

def run_causality_baseline():
    model_id = "roberta-base"

    id2label = {0: "Explicitly states: no relation", 1: "Causation", 2: "Correlation", 3: "No mention of a relation"}
    label2id = {"Explicitly states: no relation": 0, "Causation": 1, "Correlation": 2, "No mention of a relation": 3}

    train_dataset = pd.read_csv('../temp/causality_train.csv', index_col=0)
    train_dataset.label = train_dataset.label.map(label2id)

    test_dataset = pd.read_csv('../temp/causality_test.csv', index_col=0)
    test_dataset.label = test_dataset.label.map(label2id)

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # This function tokenizes the input text using the RoBERTa tokenizer.
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        return tokenizer(batch["finding"], padding=True, truncation=True, max_length=1536)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="../out",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"../logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained('../out/baseline_roberta_causality.pt')

    pred = trainer.predict(test_dataset)
    score = f1_score(test_dataset['label'], pred.predictions.argmax(axis=1), average=None)
    print(score)
    # [0.4        0.5645933  0.61538462 0.53631285]

def run_certainty_baseline():
    model_id = "roberta-base"

    id2label = {0: "Certain", 1: "Somewhat certain", 2: "Somewhat uncertain", 3: "Uncertain"}
    label2id = {"Certain": 0, "Somewhat certain": 1, "Somewhat uncertain": 2, "Uncertain": 3}

    train_dataset = pd.read_csv('../temp/certainty_train.csv', index_col=0)
    train_dataset.label = train_dataset.label.map(label2id)

    test_dataset = pd.read_csv('../temp/certainty_test.csv', index_col=0)
    test_dataset.label = test_dataset.label.map(label2id)

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # This function tokenizes the input text using the RoBERTa tokenizer.
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        return tokenizer(batch["finding"], padding=True, truncation=True, max_length=1536)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="../out",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"../logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained('../out/baseline_roberta_certainty.pt')

    pred = trainer.predict(test_dataset)
    score = f1_score(test_dataset['label'], pred.predictions.argmax(axis=1), average=None)
    print(score)
    # [0.66435986 0.48275862 0.3697479  0.07142857]

def run_generalization_baseline():
    model_id = "roberta-base"

    id2label = {0: "Paper Finding", 1: "They are at the same level of generality", 2: "Reported Finding"}
    label2id = {"Paper Finding": 0, "They are at the same level of generality": 1, "Reported Finding": 2}

    train_dataset = pd.read_csv('../temp/generalization_train.csv', index_col=0)
    train_dataset["label"] = train_dataset["which-finding-is-more-general"].map(label2id)

    test_dataset = pd.read_csv('../temp/generalization_test.csv', index_col=0)
    test_dataset["label"] = test_dataset["which-finding-is-more-general"].map(label2id)

    train_dataset["finding"] = train_dataset.paper_finding + ";" + train_dataset.reported_finding
    test_dataset["finding"] = test_dataset.paper_finding + ";" + test_dataset.reported_finding

    # train_dataset = train_dataset.drop(columns=['paper_finding', 'reported_finding'])
    # test_dataset = test_dataset.drop(columns=['paper_finding', 'reported_finding'])

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # This function tokenizes the input text using the RoBERTa tokenizer.
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        return tokenizer(batch["paper_finding"], batch["reported_finding"], padding=True, truncation=True, max_length=3072)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="../out",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"../logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained('../out/baseline_roberta_generalization.pt')

    pred = trainer.predict(test_dataset)
    score = f1_score(test_dataset['label'], pred.predictions.argmax(axis=1), average=None)
    print(score)
    # [0.44444444 0.45333333 0.74146341]
    # not really the same as in the paper, but close enough I guess

def run_sensationalism_baseline():
    model_id = "roberta-base"

    id2label = {0: "Paper Finding", 1: "They are at the same level of generality", 2: "Reported Finding"}
    label2id = {"Paper Finding": 0, "They are at the same level of generality": 1, "Reported Finding": 2}

    train_dataset = pd.read_csv('../temp/sensationalism_train.csv', index_col=0)
    train_dataset["label"] = train_dataset["which-finding-is-more-general"].map(label2id)

    test_dataset = pd.read_csv('../temp/sensationalism_test.csv', index_col=0)
    test_dataset["label"] = test_dataset["which-finding-is-more-general"].map(label2id)

    train_dataset["finding"] = train_dataset.paper_finding + ";" + train_dataset.reported_finding
    test_dataset["finding"] = test_dataset.paper_finding + ";" + test_dataset.reported_finding

    # train_dataset = train_dataset.drop(columns=['paper_finding', 'reported_finding'])
    # test_dataset = test_dataset.drop(columns=['paper_finding', 'reported_finding'])

    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    # Preprocessing
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)

    # This function tokenizes the input text using the RoBERTa tokenizer.
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        return tokenizer(batch["paper_finding"], batch["reported_finding"], padding=True, truncation=True, max_length=3072)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_id)
    config.update({"id2label": id2label})

    model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="../out",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"../logs",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained('../out/baseline_roberta_sensationalism.pt')

    full_data = concatenate_datasets([train_dataset, test_dataset])
    pred = trainer.predict(full_data)
    score = f1_score(full_data['score'], pred.predictions.argmax(axis=1), average=None)
    print(score)
    #

if __name__ == '__main__':
    run_causality_baseline()
    run_certainty_baseline()
    run_generalization_baseline()
    run_sensationalism_baseline()