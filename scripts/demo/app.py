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

# Load GRU, LSTM, and BiLSTM models and tokenizers
gru_repo_id = "arjahojnik/GRU-sentiment-model"
gru_model_path = hf_hub_download(repo_id=gru_repo_id, filename="best_GRU_tuning_model.h5")
gru_model = load_model(gru_model_path)
gru_tokenizer_path = hf_hub_download(repo_id=gru_repo_id, filename="my_tokenizer.pkl")
with open(gru_tokenizer_path, "rb") as f:
    gru_tokenizer = pickle.load(f)

lstm_repo_id = "arjahojnik/LSTM-sentiment-model"
lstm_model_path = hf_hub_download(repo_id=lstm_repo_id, filename="LSTM_model.h5")
lstm_model = load_model(lstm_model_path)
lstm_tokenizer_path = hf_hub_download(repo_id=lstm_repo_id, filename="my_tokenizer.pkl")
with open(lstm_tokenizer_path, "rb") as f:
    lstm_tokenizer = pickle.load(f)

bilstm_repo_id = "arjahojnik/BiLSTM-sentiment-model"
bilstm_model_path = hf_hub_download(repo_id=bilstm_repo_id, filename="BiLSTM_model.h5")
bilstm_model = load_model(bilstm_model_path)
bilstm_tokenizer_path = hf_hub_download(repo_id=bilstm_repo_id, filename="my_tokenizer.pkl")
with open(bilstm_tokenizer_path, "rb") as f:
    bilstm_tokenizer = pickle.load(f)

# Preprocessing function for text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text).strip()
    return text

# Prediction functions for GRU, LSTM, and BiLSTM
def predict_with_gru(text):
    cleaned = preprocess_text(text)
    seq = gru_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = gru_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)

def predict_with_lstm(text):
    cleaned = preprocess_text(text)
    seq = lstm_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = lstm_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)

def predict_with_bilstm(text):
    cleaned = preprocess_text(text)
    seq = bilstm_tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = bilstm_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)

# Load other models
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

# Logistic regression model and TF-IDF vectorizer
logistic_regression_repo = "nhull/logistic-regression-model"
log_reg_model_path = hf_hub_download(repo_id=logistic_regression_repo, filename="logistic_regression_model.pkl")
with open(log_reg_model_path, "rb") as model_file:
    log_reg_model = pickle.load(model_file)

vectorizer_path = hf_hub_download(repo_id=logistic_regression_repo, filename="tfidf_vectorizer.pkl")
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Move HuggingFace models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_data in models.values():
    if "model" in model_data:
        model_data["model"].to(device)

# Prediction functions for other models
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

# Unified function for analysis
def analyze_sentiment_and_statistics(text):
    results = {
        "Logistic Regression": predict_with_logistic_regression(text),
        "GRU Model": predict_with_gru(text),
        "LSTM Model": predict_with_lstm(text),
        "BiLSTM Model": predict_with_bilstm(text),
        "DistilBERT": predict_with_distilbert(text),
        "BERT Multilingual (NLP Town)": predict_with_bert_multilingual(text),
        "TinyBERT": predict_with_tinybert(text),
        "RoBERTa": predict_with_roberta_ordek899(text),
    }
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
with gr.Blocks(
    css="""
    .gradio-container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem;
    }
    footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: gray;
    }
    """
) as demo:
    gr.Markdown("# Sentiment Analysis Demo")
    gr.Markdown(
        """
        This demo analyzes the sentiment of text inputs (e.g., hotel or restaurant reviews) on a scale from 1 to 5 using various machine learning, deep learning, and transformer-based models. 

        - **Machine Learning**: Logistic Regression with TF-IDF.
        - **Deep Learning**: GRU, LSTM, and BiLSTM models.
        - **Transformers**: DistilBERT, TinyBERT, BERT Multilingual, and RoBERTa.

        ### Features:
        - Compare predictions across different models.
        - See which model predicts the highest and lowest scores.
        - Get the average sentiment score across all models.
        - Easily test with your own input or select from suggested reviews.

        Use this app to explore how different models interpret sentiment and compare their outputs!
        """
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
            
            def update_textbox(selected_sample):
                return selected_sample
            
            sample_dropdown.change(
                update_textbox,
                inputs=[sample_dropdown],
                outputs=[text_input]
            )
            analyze_button = gr.Button("Analyze Sentiment")
        
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Machine Learning")
            log_reg_output = gr.Textbox(label="Logistic Regression", interactive=False)

        with gr.Column():
            gr.Markdown("### Deep Learning")
            gru_output = gr.Textbox(label="GRU Model", interactive=False)
            lstm_output = gr.Textbox(label="LSTM Model", interactive=False)
            bilstm_output = gr.Textbox(label="BiLSTM Model", interactive=False)

        with gr.Column():
            gr.Markdown("### Transformers")
            distilbert_output = gr.Textbox(label="DistilBERT", interactive=False)
            bert_output = gr.Textbox(label="BERT Multilingual", interactive=False)
            tinybert_output = gr.Textbox(label="TinyBERT", interactive=False)
            roberta_output = gr.Textbox(label="RoBERTa", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Statistics")
            stats_output = gr.Textbox(label="Statistics", interactive=False)

    # Add footer
    gr.Markdown(
        """
        <footer>
            This demo was built as a part of the NLP course at the University of Zagreb.  
            Check out our GitHub repository:  
            <a href="https://github.com/FFZG-NLP-2024/TripAdvisor-Sentiment/" target="_blank">TripAdvisor Sentiment Analysis</a>  
            Explore our HuggingFace collection:  
            <a href="https://huggingface.co/collections/nhull/nlp-zg-6794604b85fd4216e6470d38" target="_blank">NLP Zagreb HuggingFace Collection</a>
        </footer>
        """
    )

    def process_input_and_analyze(text_input):
        results, statistics = analyze_sentiment_and_statistics(text_input)
        if "Message" in statistics:
            return (
                results["Logistic Regression"],
                results["GRU Model"],
                results["LSTM Model"],
                results["BiLSTM Model"],
                results["DistilBERT"],
                results["BERT Multilingual (NLP Town)"],
                results["TinyBERT"],
                results["RoBERTa"],
                f"Statistics:\n{statistics['Message']}\nAverage Score: {statistics['Average Score']}"
            )
        else:
            return (
                results["Logistic Regression"],
                results["GRU Model"],
                results["LSTM Model"],
                results["BiLSTM Model"],
                results["DistilBERT"],
                results["BERT Multilingual (NLP Town)"],
                results["TinyBERT"],
                results["RoBERTa"],
                f"Statistics:\n{statistics['Lowest Score']}\n{statistics['Highest Score']}\nAverage Score: {statistics['Average Score']}"
            )
    
    analyze_button.click(
        process_input_and_analyze,
        inputs=[text_input],
        outputs=[
            log_reg_output, 
            gru_output, 
            lstm_output, 
            bilstm_output, 
            distilbert_output, 
            bert_output, 
            tinybert_output, 
            roberta_output, 
            stats_output
        ]
    )

demo.launch()
