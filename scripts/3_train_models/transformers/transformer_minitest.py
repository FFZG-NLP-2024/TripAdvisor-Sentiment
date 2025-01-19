import torch
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ================= Step 1: Load Dataset =================
print("Loading the Hugging Face dataset...")
dataset = load_dataset("nhull/125-tripadvisor-reviews")  # Replace with your dataset

# Inspect the dataset
print("Dataset example:")
print(dataset["train"][0])

# ================= Step 2: Load Pre-Trained Model =================
print("Loading the pre-trained model and tokenizer...")
# Pre-trained Hugging Face model (no need for custom model files if not saved locally)
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# ================= Step 3: Tokenize Dataset =================
print("Tokenizing the dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

# Tokenize the dataset (for "train" only; replace with "test" if needed)
tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)

# ================= Step 4: Perform Predictions =================
print("Performing predictions...")
def predict(example):
    inputs = {
        "input_ids": torch.tensor(example["input_ids"]),
        "attention_mask": torch.tensor(example["attention_mask"])
    }
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).tolist()
    return {"predictions": predictions}

predicted_dataset = tokenized_dataset.map(predict, batched=True)

# ================= Step 5: Evaluate Predictions =================
print("Evaluating predictions...")
y_true = tokenized_dataset["label"]  # True labels
y_pred = predicted_dataset["predictions"]  # Predicted labels

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["1", "2", "3", "4", "5"],  # Adjust target names as needed
    labels=[0, 1, 2, 3, 4]
))

# ================= Step 6: Save the Predictions (Optional) =================
print("Saving predictions to a CSV file...")
import pandas as pd
predicted_df = pd.DataFrame({"true_labels": y_true, "predicted_labels": y_pred})
predicted_df.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
