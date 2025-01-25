import csv
from datasets import load_dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Define the custom classifier class
class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=3e-05, batch_size=64, epochs=10, patience=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience  # Increased patience for early stopping
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5  # 5 classes (1 to 5)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y):
        inputs = self.tokenizer(X, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        # Adjust labels to 0-based indexing for the model
        labels = torch.tensor([label - 1 for label in y], dtype=torch.long)
        train_dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        num_training_steps = self.epochs * len(train_dataloader)
        num_warmup_steps = int(0.1 * num_training_steps)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        best_val_accuracy = 0.0
        epochs_without_improvement = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for batch in pbar:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    epoch_loss += loss.item()

                    logits = outputs.logits
                    predicted_labels = torch.argmax(logits, dim=-1)
                    correct_predictions += (predicted_labels == labels).sum().item()
                    total_predictions += labels.size(0)

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                    pbar.set_postfix(loss=loss.item())

            epoch_accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch+1} - Loss: {epoch_loss/len(train_dataloader):.4f}, Accuracy: {epoch_accuracy:.4f}")

        return self

    def predict(self, X, y=None):
        # Tokenize the dataset
        inputs = self.tokenizer(X, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        self.model.eval()

        correct_predictions = []
        misclassified_predictions = []

        predictions = []
        with torch.no_grad():
            for i in range(0, len(inputs["input_ids"]), self.batch_size):
                batch_input_ids = inputs["input_ids"][i:i+self.batch_size].to(self.device)
                batch_attention_mask = inputs["attention_mask"][i:i+self.batch_size].to(self.device)
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits
                # Convert predictions back to 1-indexed range
                predicted_labels = torch.argmax(logits, dim=-1) + 1
                predictions.append(predicted_labels.cpu().numpy())

                if y is not None:
                    for idx, (true_label, pred_label) in enumerate(zip(y[i:i+self.batch_size], predicted_labels)):
                        sentence = X[i + idx]
                        if true_label == pred_label:
                            correct_predictions.append([sentence, true_label, pred_label.item()])
                        else:
                            misclassified_predictions.append([sentence, true_label, pred_label.item(), true_label - pred_label.item()])

        return np.concatenate(predictions, axis=0), correct_predictions, misclassified_predictions

    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "batch_size": self.batch_size, "epochs": self.epochs}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

# Keep the labels as-is (1 to 5)
def preprocess_function(examples):
    return examples  # No label transformation

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Use the whole dataset for training and validation
X_train = tokenized_dataset["train"]["review"]
y_train = tokenized_dataset["train"]["label"]
X_val = tokenized_dataset["validation"]["review"]
y_val = tokenized_dataset["validation"]["label"]

# Train the model with specific hyperparameters
model = DistilBertClassifier(learning_rate=3e-05, batch_size=64, epochs=10)
model.fit(X_train, y_train)

# Save the trained model
model.model.save_pretrained("models/distilbert/best_trained_model")
model.tokenizer.save_pretrained("models/distilbert/best_trained_model")
print("Model and tokenizer saved!")

# Make predictions on the validation set and track correct and misclassified predictions
y_pred, correct_predictions, misclassified_predictions = model.predict(X_val, y_val)

# Save the correct predictions to a CSV file
with open("correct_predictions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Real Label", "Predicted Label"])
    writer.writerows(correct_predictions)

# Save the misclassified predictions to a CSV file
with open("misclassified_predictions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "Real Label", "Predicted Label", "Difference"])
    writer.writerows(misclassified_predictions)

# Evaluate the model metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
