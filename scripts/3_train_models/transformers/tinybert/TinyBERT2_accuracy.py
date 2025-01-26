from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np

def tokenize_and_encode_function(examples):
    tokenized_inputs = tokenizer(examples['review'], padding="max_length", truncation=True, max_length=128)
    labels = torch.tensor([label - 1 for label in examples['label']], dtype=torch.long)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    tokenized_datasets = dataset.map(tokenize_and_encode_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.005,
        warmup_ratio=0.1,
        logging_dir='./logs',
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save the trained model
    trainer.save_model("./trained_tinybert")

    results = trainer.evaluate(tokenized_datasets["test"])
    print("Final Evaluation Results:", results)

if __name__ == "__main__":
    main()
