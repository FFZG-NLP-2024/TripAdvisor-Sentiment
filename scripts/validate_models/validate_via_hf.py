from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.functional import softmax

# Load the model and tokenizer from Hugging Face
model_name = "nhull/distilbert-sentiment-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model and tokenizer loaded successfully!")

# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

# Preprocess the dataset
def preprocess_function(examples):
    examples["label"] = [label - 1 for label in examples["label"]]  # Adjust labels from 1-5 to 0-4
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Custom collate function for DataLoader
def custom_collate_fn(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
    }
    return batch

# Create the validation DataLoader
validation_dataloader = DataLoader(
    tokenized_dataset["validation"], 
    batch_size=64,  # You can adjust the batch size if needed
    shuffle=False, 
    collate_fn=custom_collate_fn
)

print("Validation dataset preprocessed and DataLoader created!")

# Set the model to evaluation mode
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in validation_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        probs = softmax(logits, dim=-1)  # Get probabilities from logits
        preds = torch.argmax(probs, dim=-1)  # Get predicted class
        all_preds.extend(preds.cpu().tolist())  # Convert to list and move to CPU
        all_labels.extend(batch["labels"].cpu().tolist())  # Ground truth labels

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

# Classification report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
