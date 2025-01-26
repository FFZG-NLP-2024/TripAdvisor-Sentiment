from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

model_id = "elo4/TinyBERT-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

dataset_id = "nhull/tripadvisor-split-dataset-v2"
data = load_dataset(dataset_id, split='test')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[:, 1].item()

significant_differences = []

for example in data:
    predicted_sentiment = predict_sentiment(example['review']) * 10
    actual_sentiment = example['label']
    if abs(predicted_sentiment - actual_sentiment) >= 3:
        significant_differences.append({
            'review': example['review'],
            'predicted_sentiment': predicted_sentiment,
            'actual_sentiment': actual_sentiment
        })

for diff in significant_differences[:5]:
    print(f"Review: {diff['review']}\nPredicted Sentiment: {diff['predicted_sentiment']}\nActual Sentiment: {diff['actual_sentiment']}\n")

print(f"Total cases found: {len(significant_differences)}")
