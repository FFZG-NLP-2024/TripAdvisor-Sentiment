import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder
import torch

# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_parquet("hf://datasets/jniimi/tripadvisor-review-rating/data/train-00000-of-00001.parquet")

# Step 2: Inspect and clean the data
print("Inspecting the first few rows of the dataset:")
print(df.head())

print("Checking for missing values...")
print(df.isnull().sum())

print("Cleaning the review text...")
df['review'] = df['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Ensure labels are integers
print("Ensuring labels are integers...")
label_encoder = LabelEncoder()
df['overall'] = label_encoder.fit_transform(df['overall'])

# Step 3: Check class distribution
print("Class distribution:")
print(df['overall'].value_counts())

# Step 4: Balance the dataset to 1000 reviews (250 from each class, assuming 4 classes)
min_samples_per_class = 250  # Define the number of samples per class for a balanced dataset
balanced_df = pd.DataFrame()

# Loop through each class (rating) and sample a balanced number of reviews
for label in df['overall'].unique():
    class_df = df[df['overall'] == label]
    sampled_class_df = class_df.sample(n=min_samples_per_class, random_state=42)
    balanced_df = pd.concat([balanced_df, sampled_class_df])

# Check the distribution of the balanced dataset
print("Balanced class distribution:")
print(balanced_df['overall'].value_counts())

# Step 5: Split the data into train and test sets
print("Splitting the data...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    balanced_df['review'], balanced_df['overall'], test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset format
print("Converting to Hugging Face dataset format...")
train_dataset = Dataset.from_dict({'review': train_texts, 'labels': train_labels})
test_dataset = Dataset.from_dict({'review': test_texts, 'labels': test_labels})
raw_datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

# Step 6: Tokenize the data
print("Tokenizing the dataset...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["review"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

# Include labels explicitly
tokenized_datasets = raw_datasets.map(
    lambda x: {**tokenize_function(x), "labels": x["labels"]},
    batched=True
)

# Step 7: Load the pre-trained model
print("Loading the pre-trained model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=len(label_encoder.classes_)
)

# Step 8: Set up the Trainer
print("Setting up the Trainer...")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=2,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 9: Train the model
print("Training the model on the balanced dataset...")
trainer.train()

# Step 10: Evaluate the model
print("Evaluating the model...")
metrics = trainer.evaluate()
print(metrics)

# Step 11: Save the model, tokenizer, and label encoder
print("Saving the model, tokenizer, and label encoder...")
model.save_pretrained("scripts/modeli/sentiment_model")
tokenizer.save_pretrained("scripts/modeli/sentiment_model")

# Save the label encoder
torch.save(label_encoder.classes_, "scripts/modeli/sentiment_model/label_encoder.pt")

# ========================== Testing the model ==========================

# Step 12: Load the saved model, tokenizer, and label encoder
print("Loading the saved model, tokenizer, and label encoder...")
model_path = "scripts/modeli/sentiment_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = torch.load(f"{model_path}/label_encoder.pt")

# Step 13: Function to predict sentiment for new random texts
def predict_sentiment(texts):
    # Tokenize the texts
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()  # Get the index of the highest logit
    return predictions

# Step 14: Example random texts for prediction
random_texts = [
    "The hotel was amazing, the staff was very friendly!",
    "I didn't like the food, it was cold and bland.",
    "The service was okay, but the room was too small for my liking."
]

# Step 15: Make predictions
predictions = predict_sentiment(random_texts)

# Step 16: Decode predictions
decoded_predictions = label_encoder.inverse_transform(predictions)

# Step 17: Display the results
print("\nPrediction results:")
for text, pred in zip(random_texts, decoded_predictions):
    print(f"Review: {text}\nPredicted Rating: {pred}\n")
