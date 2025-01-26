from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import numpy as np
import torch

def tokenize_and_encode_function(examples):
    tokenized_inputs = tokenizer(examples['review'], padding="max_length", truncation=True, max_length=128)
    labels = torch.tensor([label - 1 for label in examples['label']], dtype=torch.long)  # Assuming labels are 1-indexed
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Replace `load_metric` with `evaluate`
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    return accuracy

def main():
    dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")
    model_name = "huawei-noah/TinyBERT_General_4L_312D"

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    tokenized_datasets = dataset.map(tokenize_and_encode_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("./trained_tinybert")

    results = trainer.evaluate(tokenized_datasets["test"])
    print("Final Evaluation Results:", results)

if __name__ == "__main__":
    main()