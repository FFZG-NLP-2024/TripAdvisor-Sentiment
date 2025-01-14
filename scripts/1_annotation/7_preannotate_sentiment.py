import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from scipy.special import softmax
from tqdm import tqdm 

# Define the model names
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "siebert/sentiment-roberta-large-english" 
]

# Load models and tokenizers
loaded_models = [AutoModelForSequenceClassification.from_pretrained(model) for model in models]
tokenizers = [AutoTokenizer.from_pretrained(model) for model in models]

# Load your restaurant reviews CSV
df = pd.read_csv('data/2_sample/5_reviews_sentences.csv')

# Initialize the labels list
labels = ["Negative", "Positive"]

# List to store the final predictions
final_predictions = []

# Loop over each sentence in your CSV with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
    sentence = row['sentence']
    
    # Prepare to store scores for each model
    combined_scores = np.zeros(len(labels))
    
    # Run each model on the sentence
    for model, tokenizer in zip(loaded_models, tokenizers):
        # Tokenize the sentence
        encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits
            scores = softmax(logits.numpy())
        
        # Accumulate scores from each model
        combined_scores += scores[0]
    
    # Average the scores across all models
    averaged_scores = combined_scores / len(models)

    # Get the highest confidence sentiment
    max_label_index = np.argmax(averaged_scores)
    max_label = labels[max_label_index]
    confidence = np.round(float(averaged_scores[max_label_index]), 4)
    
    # Store the prediction and confidence
    final_predictions.append(f"{max_label} ({confidence})")
    
    # Print the results for each review
    print(f"Review: {sentence}")
    print(f"Combined Prediction: {max_label} with confidence {confidence}")
    print("\n" + "-"*50)  

# Add the predictions as a new column in the DataFrame
df['Ensembled Sentiment'] = final_predictions

# Save the updated DataFrame to a new CSV file
df.to_csv('data/2_sample/7_reviews_sentences_with_ensembled_sentiment.csv', index=False)

print("Sentiment predictions have been saved to 'data/2_sample/7_reviews_sentences_with_ensembled_sentiment.csv'")
