from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = AutoModelForSequenceClassification.from_pretrained("models/distilbert/best_trained_model")
tokenizer = AutoTokenizer.from_pretrained("models/distilbert/best_trained_model")

dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

def tokenize_and_encode_function(examples):
    tokenized_inputs = tokenizer(examples['review'], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs['labels'] = [int(label - 1) for label in examples['label']]
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_encode_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate(tokenized_datasets["validation"])

print("Evaluation Results:", results)
