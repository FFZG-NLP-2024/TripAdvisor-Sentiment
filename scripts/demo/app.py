import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU and enforce CPU execution

import gradio as gr
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import hf_hub_download
import torch
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load pre-trained models and tokenizers
models = {
    "DistilBERT": {
        "tokenizer": DistilBertTokenizerFast.from_pretrained("nhull/distilbert-sentiment-model"),
        "model": DistilBertForSequenceClassification.from_pretrained("nhull/distilbert-sentiment-model"),
    },
    "Logistic Regression": {},  # Placeholder for logistic regression
    "BERT Multilingual (NLP Town)": {
        "tokenizer": AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
        "model": AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
    },
    "TinyBERT": {
        "tokenizer": AutoTokenizer.from_pretrained("elo4/TinyBERT-sentiment-model"),
        "model": AutoModelForSequenceClassification.from_pretrained("elo4/TinyBERT-sentiment-model"),
    },
    "RoBERTa": {
        "tokenizer": AutoTokenizer.from_pretrained("ordek899/roberta_1to5rating_pred_for_restaur_trained_on_hotels"),
        "model": AutoModelForSequenceClassification.from_pretrained("ordek899/roberta_1to5rating_pred_for_restaur_trained_on_hotels"),
    }
}

# Load logistic regression model and vectorizer
logistic_regression_repo = "nhull/logistic-regression-model"

# Download and load logistic regression model
log_reg_model_path = hf_hub_download(repo_id=logistic_regression_repo, filename="logistic_regression_model.pkl")
with open(log_reg_model_path, "rb") as model_file:
    log_reg_model = pickle.load(model_file)

# Download and load TF-IDF vectorizer
vectorizer_path = hf_hub_download(repo_id=logistic_regression_repo, filename="tfidf_vectorizer.pkl")
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Move HuggingFace models to device (if GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_data in models.values():
    if "model" in model_data:
        model_data["model"].to(device)

# Load GRU model and tokenizer
gru_repo_id = "arjahojnik/GRU-sentiment-model"
gru_model_path = hf_hub_download(repo_id=gru_repo_id, filename="best_GRU_tuning_model.h5")
gru_model = load_model(gru_model_path)
gru_tokenizer_path = hf_hub_download(repo_id=gru_repo_id, filename="my_tokenizer.pkl")
with open(gru_tokenizer_path, "rb") as f:
    gru_tokenizer = pickle.load(f)

# Preprocessing function for GRU
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text).strip()
    return text

# GRU prediction function
def predict_with_gru(text):
    cleaned = preprocess_text(text)
    seq = gru_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)  # Ensure maxlen matches the GRU training
    probs = gru_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)

# Functions for other model predictions
def predict_with_distilbert(text):
    tokenizer = models["DistilBERT"]["tokenizer"]
    model = models["DistilBERT"]["model"]
    encodings = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()
    return int(predictions[0] + 1)

def predict_with_logistic_regression(text):
    transformed_text = vectorizer.transform([text])
    predictions = log_reg_model.predict(transformed_text)
    return int(predictions[0])

def predict_with_bert_multilingual(text):
    tokenizer = models["BERT Multilingual (NLP Town)"]["tokenizer"]
    model = models["BERT Multilingual (NLP Town)"]["model"]
    encodings = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()
    return int(predictions[0] + 1)

def predict_with_tinybert(text):
    tokenizer = models["TinyBERT"]["tokenizer"]
    model = models["TinyBERT"]["model"]
    encodings = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()
    return int(predictions[0] + 1)

def predict_with_roberta_ordek899(text):
    tokenizer = models["RoBERTa"]["tokenizer"]
    model = models["RoBERTa"]["model"]
    encodings = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = logits.argmax(axis=-1).cpu().numpy()
    return int(predictions[0] + 1)

# Unified function for sentiment analysis and statistics
def analyze_sentiment_and_statistics(text):
    results = {
        "GRU Model": predict_with_gru(text),
        "DistilBERT": predict_with_distilbert(text),
        "Logistic Regression": predict_with_logistic_regression(text),
        "BERT Multilingual (NLP Town)": predict_with_bert_multilingual(text),
        "TinyBERT": predict_with_tinybert(text),
        "RoBERTa": predict_with_roberta_ordek899(text),
    }
    
    # Calculate statistics
    scores = list(results.values())
    min_score = min(scores)
    max_score = max(scores)
    min_score_models = [model for model, score in results.items() if score == min_score]
    max_score_models = [model for model, score in results.items() if score == max_score]
    average_score = np.mean(scores)

    if all(score == scores[0] for score in scores):
        statistics = {
            "Message": "All models predict the same score.",
            "Average Score": f"{average_score:.2f}",
        }
    else:
        statistics = {
            "Lowest Score": f"{min_score} (Models: {', '.join(min_score_models)})",
            "Highest Score": f"{max_score} (Models: {', '.join(max_score_models)})",
            "Average Score": f"{average_score:.2f}",
        }
    return results, statistics

# Gradio Interface
with gr.Blocks(css=".gradio-container { max-width: 900px; margin: auto; padding: 20px; }") as demo:
    gr.Markdown("# Sentiment Analysis App")
    gr.Markdown(
        "This app predicts the sentiment of the input text on a scale from 1 to 5 using multiple models and provides basic statistics."
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text here:", 
                lines=3, 
                placeholder="Type your hotel/restaurant review here..."
            )
            sample_reviews = [
                "The hotel was fantastic! Clean rooms and excellent service.",
                "The food was horrible, and the staff was rude.",
                "Amazing experience overall. Highly recommend!",
                "It was okay, not great but not terrible either.",
                "Terrible! The room was dirty, and the service was non-existent."
            ]
            sample_dropdown = gr.Dropdown(
                choices=sample_reviews, 
                label="Or select a sample review:", 
                interactive=True
            )
            
            # Sync dropdown with text input
            def update_textbox(selected_sample):
                return selected_sample
            
            sample_dropdown.change(
                update_textbox,
                inputs=[sample_dropdown],
                outputs=[text_input]
            )
        
        with gr.Column():
            analyze_button = gr.Button("Analyze Sentiment")

    with gr.Row():
        with gr.Column():
            gru_output = gr.Textbox(label="Predicted Sentiment (GRU Model)", interactive=False)
            distilbert_output = gr.Textbox(label="Predicted Sentiment (DistilBERT)", interactive=False)
            log_reg_output = gr.Textbox(label="Predicted Sentiment (Logistic Regression)", interactive=False)
            bert_output = gr.Textbox(label="Predicted Sentiment (BERT Multilingual)", interactive=False)
            tinybert_output = gr.Textbox(label="Predicted Sentiment (TinyBERT)", interactive=False)
            roberta_ordek_output = gr.Textbox(label="Predicted Sentiment (RoBERTa)", interactive=False)
        
        with gr.Column():
            statistics_output = gr.Textbox(label="Statistics (Lowest, Highest, Average)", interactive=False)

    # Button to analyze sentiment and show statistics
    def process_input_and_analyze(text_input):
        results, statistics = analyze_sentiment_and_statistics(text_input)
        if "Message" in statistics:
            return (
                f"{results['GRU Model']}",
                f"{results['DistilBERT']}",
                f"{results['Logistic Regression']}",
                f"{results['BERT Multilingual (NLP Town)']}",
                f"{results['TinyBERT']}",
                f"{results['RoBERTa']}",
                f"Statistics:\n{statistics['Message']}\nAverage Score: {statistics['Average Score']}"
            )
        else:
            return (
                f"{results['GRU Model']}",
                f"{results['DistilBERT']}",
                f"{results['Logistic Regression']}",
                f"{results['BERT Multilingual (NLP Town)']}",
                f"{results['TinyBERT']}",
                f"{results['RoBERTa']}",
                f"Statistics:\n{statistics['Lowest Score']}\n{statistics['Highest Score']}\nAverage Score: {statistics['Average Score']}"
            )
    
    analyze_button.click(
        process_input_and_analyze,
        inputs=[text_input],
        outputs=[
            gru_output,
            distilbert_output, 
            log_reg_output, 
            bert_output, 
            tinybert_output, 
            roberta_ordek_output, 
            statistics_output
        ]
    )

# Launch the app
demo.launch()
