from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Load datasets
hotel_dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")
restaurant_dataset = load_dataset("nhull/125-tripadvisor-reviews")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Preprocessing function
def preprocess_function(examples, text_column):
    examples["label"] = [label - 1 for label in examples["label"]]  # Adjust labels to 0-4
    return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=128)

# Tokenize datasets
tokenized_hotel_dataset = hotel_dataset.map(lambda x: preprocess_function(x, text_column="review"), batched=True)
tokenized_restaurant_dataset = restaurant_dataset.map(lambda x: preprocess_function(x, text_column="text"), batched=True)

# Custom collate function
def custom_collate_fn(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
    }
    return batch

# Data loaders
train_hotel_dataloader = DataLoader(
    tokenized_hotel_dataset["train"], batch_size=16, shuffle=True, collate_fn=custom_collate_fn
)
val_hotel_dataloader = DataLoader(
    tokenized_hotel_dataset["validation"], batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)
test_hotel_dataloader = DataLoader(
    tokenized_hotel_dataset["test"], batch_size=16, shuffle=False, collate_fn=custom_collate_fn
)
restaurant_dataloader = DataLoader(
    tokenized_restaurant_dataset["train"], batch_size=8, shuffle=False, collate_fn=custom_collate_fn
)

# Model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
def get_optimizer_scheduler(model, dataloader, lr=5e-5, epochs=3):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = epochs * len(dataloader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return optimizer, scheduler

# Training function
def train_model(dataloader, model, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader)}")
        # Save model checkpoint after each epoch
        model.save_pretrained(f"model_epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"model_epoch_{epoch + 1}")

# Evaluation function
def evaluate_model(dataloader, model, dataset_name):
    model.eval()
    predictions = []  # Initialize predictions list
    true_labels = []  # Initialize true labels list
    instance_results = []  # To store all instances with real, predicted, and correctness

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

            # Store real, predicted, and correctness for each instance
            for i in range(len(preds)):
                real_value = batch["labels"][i].item() + 1  # Adjust label back to original scale (1-5)
                predicted_value = preds[i].item() + 1  # Adjust predicted value back to original scale (1-5)
                correctness = 1 if real_value == predicted_value else 0
                instance_results.append({
                    'Real Value': real_value,
                    'Predicted Value': predicted_value,
                    'Correct': correctness
                })

    # Save instance results to CSV
    df_all = pd.DataFrame(instance_results)
    df_all.to_csv(f"{dataset_name}_instance_results_1.csv", index=False)
    print(f"Instance results saved to {dataset_name}_instance_results_1.csv")

    # Generate and print classification report
    print("\nClassification Report:")
    report = classification_report(true_labels, predictions, target_names=["1", "2", "3", "4", "5"], zero_division=0)
    print(report)

    # Save classification report as CSV
    report_dict = classification_report(true_labels, predictions, target_names=["1", "2", "3", "4", "5"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"{dataset_name}_classification_report_1.csv", index=True)
    print(f"Classification report saved to {dataset_name}_classification_report_1.csv")

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "2", "3", "4", "5"], yticklabels=["1", "2", "3", "4", "5"])
    plt.title(f"{dataset_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{dataset_name}_confusion_matrix.png")
    print(f"Confusion matrix plot saved to {dataset_name}_confusion_matrix.png")
    plt.close()

# Train, validate, and test on datasets
optimizer, scheduler = get_optimizer_scheduler(model, train_hotel_dataloader, epochs=3)
print("Training on Hotel Dataset...")
train_model(train_hotel_dataloader, model, optimizer, scheduler, epochs=3)

print("\nEvaluating on Hotel Validation Dataset...")
evaluate_model(val_hotel_dataloader, model, "Hotel_Validation")

print("\nTesting on Hotel Test Dataset...")
evaluate_model(test_hotel_dataloader, model, "Hotel_Test")

print("\nTesting on Restaurant Dataset...")
evaluate_model(restaurant_dataloader, model, "Restaurant_Test")
