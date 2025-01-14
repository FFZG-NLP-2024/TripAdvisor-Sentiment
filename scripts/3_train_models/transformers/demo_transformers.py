import streamlit as st
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import torch
from sklearn.preprocessing import LabelEncoder

# Streamlit app
st.title("Sentiment Analysis App")
st.write("This app predicts the sentiment of the input text as positive or negative.")

# Model and tokenizer paths
model_path = "scripts/modeli/sentiment_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = torch.load(f"{model_path}/label_encoder.pt")

# Move model to device (if GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict sentiment for new texts
def predict_sentiment(texts):
    # Tokenize the texts
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()  # Get the index of the highest logit
    return predictions

# Input text
user_input = st.text_area("Enter your text here:")

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input:
        # Make sentiment prediction using custom model
        predictions = predict_sentiment([user_input])  # Passing as list to match expected input
        sentiment = label_encoder.inverse_transform(predictions)[0]
        
        # Output prediction
        st.subheader(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")


# Run
# streamlit run demo_transformers.py