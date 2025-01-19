from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

# Load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    examples["label"] = [label - 1 for label in examples["label"]]  # Adjust labels from 1-5 to 0-4
    return tokenizer(examples["review"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Custom collate function for DataLoader
def custom_collate_fn(features):
    batch = {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long)
    }
    return batch

# Create the DataLoader
train_dataloader = DataLoader(
    tokenized_dataset["train"], 
    batch_size=16,
    shuffle=True, 
    collate_fn=custom_collate_fn
)

# Validation dataloader
val_dataloader = DataLoader(
    tokenized_dataset["validation"], 
    batch_size=16, 
    collate_fn=custom_collate_fn
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=5,
    attention_probs_dropout_prob=0.4,  # Experimenting with dropout
    hidden_dropout_prob=0.2  # Experimenting with dropout
)
model.to(device)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,  # Starting learning rate
    weight_decay=0.01
)

# Learning rate scheduler
num_training_steps = 5 * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps as warmup

lr_scheduler = get_scheduler(
    "linear",  # Linear decay scheduler
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Training loop
epochs = 5  # Increase epochs to 5

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # Update learning rate
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} Training Loss: {epoch_loss / len(train_dataloader)}")

    # Validate the model
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in tqdm(val_dataloader, desc="Validation"):
        batch = {key: value.to(device) for key, value in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_predictions += batch["labels"].size(0)

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss}")
    print(f"Epoch {epoch + 1} Validation Accuracy: {val_accuracy}")

# Save the trained model and tokenizer
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
print("Model and tokenizer saved to 'trained_model'!")
