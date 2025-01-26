import streamlit as st
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from huggingface_hub import hf_hub_download
import torch
import pickle

# Streamlit app
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of the input text on a scale from 1 to 5 using two models: DistilBERT and Logistic Regression.")

# Load DistilBERT model and tokenizer from HuggingFace
distilbert_model_name = "nhull/distilbert-sentiment-model"
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_name)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_model_name)

# Load logistic regression model and vectorizer from HuggingFace
logistic_regression_repo = "nhull/logistic-regression-model"

# Download and load logistic regression model
log_reg_model_path = hf_hub_download(repo_id=logistic_regression_repo, filename="logistic_regression_model.pkl")
with open(log_reg_model_path, "rb") as model_file:
    log_reg_model = pickle.load(model_file)

# Download and load TF-IDF vectorizer
vectorizer_path = hf_hub_download(repo_id=logistic_regression_repo, filename="tfidf_vectorizer.pkl")
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Move DistilBERT model to device (if GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)

# Function to predict sentiment using DistilBERT
def predict_with_distilbert(texts):
    # Tokenize the texts
    encodings = distilbert_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = distilbert_model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()  # Get the index of the highest logit (0-4)
    return predictions + 1  # Convert to 1-5 scale

# Function to predict sentiment using Logistic Regression
def predict_with_logistic_regression(texts):
    # Transform the input using the vectorizer
    transformed_texts = vectorizer.transform(texts)
    
    # Predict using the logistic regression model
    predictions = log_reg_model.predict(transformed_texts)
    return predictions  # Already on the 1-5 scale

# Pre-defined examples for hotel/restaurant reviews
examples = [
    "The hotel staff was incredibly friendly and accommodating, and the room was spotless!",
    "The food was mediocre, and the service was slow. Definitely not worth the price.",
    "Absolutely loved the ambiance and the meal. The dessert was to die for!",
    "The room was noisy, and the air conditioning didn't work. Would not stay here again.",
    "The restaurant had a great view, but the food was overpriced and underwhelming."
]

# Display example buttons and populate the text area when clicked
st.write("**Or select a sample review below:**")
selected_example = None
for i, example in enumerate(examples):
    if st.button(f"Example {i + 1}"):
        selected_example = example

# Input text
user_input = st.text_area("Enter your text here:", value=selected_example or "")

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Make predictions using both models
        distilbert_prediction = int(predict_with_distilbert([user_input])[0])  # Passing as list and converting to integer
        log_reg_prediction = int(predict_with_logistic_regression([user_input])[0])  # Passing as list and converting to integer
        
        # Display predictions from both models
        st.subheader(f"Predicted Sentiment (DistilBERT): {distilbert_prediction}")
        st.subheader(f"Predicted Sentiment (Logistic Regression): {log_reg_prediction}")
    else:
        st.write("Please enter some text to analyze.")
