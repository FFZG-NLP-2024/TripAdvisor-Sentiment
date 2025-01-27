from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.functional import softmax
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
        "review": [f["review"] for f in features]  # Add review text to the batch
    }
    return batch

# Create the validation DataLoader
validation_dataloader = DataLoader(
    tokenized_dataset["test"], 
    batch_size=64,  # You can adjust the batch size if needed
    shuffle=False, 
    collate_fn=custom_collate_fn
)

print("Validation dataset preprocessed and DataLoader created!")

# Set the model to evaluation mode
model.eval()
absolute_differences = []  # Track absolute differences
signed_differences = []  # Track signed differences
all_preds = []
all_labels = []
results = []

with torch.no_grad():
    for batch in validation_dataloader:
        batch = {key: value.to(device) if key != "review" else value for key, value in batch.items()}
        outputs = model(**{k: v for k, v in batch.items() if k != "review"})
        logits = outputs.logits
        probs = softmax(logits, dim=-1)  # Get probabilities from logits
        preds = torch.argmax(probs, dim=-1)  # Get predicted class

        all_preds.extend(preds.cpu().tolist())  # Store all predictions
        all_labels.extend(batch["labels"].cpu().tolist())  # Store all true labels

        # Collect differences
        for review, true_label, pred_label in zip(batch["review"], batch["labels"].cpu().tolist(), preds.cpu().tolist()):
            difference = abs((true_label + 1) - (pred_label + 1))  # Calculate absolute difference
            signed_difference = (true_label + 1) - (pred_label + 1)  # Calculate signed difference
            absolute_differences.append(difference)
            signed_differences.append(signed_difference)

            # Append results for file output
            results.append({
                "review": review,
                "true_label": true_label + 1,  # Convert back to 1-5
                "predicted_label": pred_label + 1,  # Convert back to 1-5
                "difference": difference
            })

# Calculate statistics
accuracy = accuracy_score(all_labels, all_preds)
average_absolute_difference = sum(absolute_differences) / len(absolute_differences)
average_signed_difference = sum(signed_differences) / len(signed_differences)
bias_direction = "too low" if average_signed_difference > 0 else "too high"

# Print statistics
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Model predicts {bias_direction} on average by {average_absolute_difference:.4f}.")

# Generate classification report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, digits=4)
print(report)

# Generate confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix DistilBERT')
plt.savefig("scripts/4_validate_models/distilbert/confusion_matrix_bert_multilingual.png")
plt.show()

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("scripts/4_validate_models/distilbert/validation_results_bert_multilingual.csv", index=False)
print("Results saved to 'validation_results_bert_multilingual.csv'")
