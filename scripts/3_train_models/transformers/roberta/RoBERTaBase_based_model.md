---
license: apache-2.0
datasets:
- nhull/tripadvisor-split-dataset-v2
- nhull/125-tripadvisor-reviews
language:
- en
base_model:
- FacebookAI/roberta-base
pipeline_tag: text-classification
tags:
- sentimentanalyis
- hotelreviews
- restaurantreviews
- ratingprediction
metrics:
- accuracy
- precision
- recall
- f1
- confusion_matrix
---


# Hotel and Restaurant Review Rating Model

This model predicts the rating of a hotel or restaurant review on a scale from 1 to 5. It was trained on a hotel review dataset from Hugging Face and further tested with a small restaurant dataset. The model is based on the **RoBERTa** architecture and uses **PyTorch** for training and inference.

## Model Details

- **Model Name**: ordek899/roberta_1to5rating_pred_for_restaur_trained_on_hotels
- **Training Data**: Trained on the Hugging Face hotel dataset and tested on a small restaurant dataset.
- **Output**: A rating prediction from 1 to 5 based on the input review.

## Key Features

- **Detailed performance metrics**: Including classification report with precision, recall, and F1-score for each rating.
- **Instance-level predictions**: A CSV file containing the true label, predicted label, and correctness for each instance.
- **Confusion matrix visualization**: A confusion matrix to evaluate the model’s performance across different ratings (1 to 5).
- **Comprehensive model information**: Includes detailed insights into model training, evaluation, and predictions.

## How It Works

The model uses a transformer-based architecture (RoBERTa) for classifying reviews into one of five ratings (1 to 5). It processes hotel and restaurant reviews and predicts a numerical rating based on the sentiment and content of the review.

## Dependencies

To use this model, you will need to install the following Python packages:

```bash
pip install transformers datasets torch
```

## Example Usage

Making Predictions with the Model
To use the model for making predictions on a review, you can run the following Python code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "ordek899/roberta_1to5rating_pred_for_restaur_trained_on_hotels"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example review - Replace this text with your own review
review = """Insert your review here."""

# Tokenize the input review
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Make prediction
with torch.no_grad():
logits = model(**inputs).logits

# Convert logits to predicted rating (from 1 to 5)
predicted_rating = torch.argmax(logits, dim=-1) + 1 # Adding 1 because labels are typically 0-indexed
print(f"Predicted rating: {predicted_rating.item()}")
```

## Model Evaluation and Training

The model was trained on the Hugging Face hotel dataset and evaluated with a small restaurant dataset. It predicts the rating of a review on a scale from 1 to 5.

The following files are generated during the evaluation process:
Instance-Level Results: A CSV file containing the true label, predicted label, and correctness for each instance.
Classification Report: A CSV file with detailed metrics like precision, recall, and F1-score for each rating.
Confusion Matrix: A heatmap that visually shows how well the model performed across all five rating categories (1 to 5).

## Example Output

When you run the prediction code with an input review, the output will show something like:

```bash
Predicted rating: 3
```

This indicates that the model has predicted a rating of 3 for the provided review.
