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

# Load models and tokenizers
models = {
    "DistilBERT": {
        "tokenizer": DistilBertTokenizerFast.from_pretrained("nhull/distilbert-sentiment-model"),
        "model": DistilBertForSequenceClassification.from_pretrained("nhull/distilbert-sentiment-model"),
    },
    "Logistic Regression": {},  # Placeholder for logistic regression
    "BERT Multilingual (NLP Town)": {
        "tokenizer": AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
        "model": AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment"),
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

# Functions for prediction
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

# Unified function for sentiment analysis and statistics
def analyze_sentiment_and_statistics(text):
    results = {
        "DistilBERT": predict_with_distilbert(text),
        "Logistic Regression": predict_with_logistic_regression(text),
        "BERT Multilingual (NLP Town)": predict_with_bert_multilingual(text),
    }
    
    # Calculate statistics
    scores = list(results.values())
    min_score_model = min(results, key=results.get)
    max_score_model = max(results, key=results.get)
    average_score = np.mean(scores)
    
    statistics = {
        "Lowest Score": f"{results[min_score_model]} (Model: {min_score_model})",
        "Highest Score": f"{results[max_score_model]} (Model: {max_score_model})",
        "Average Score": f"{average_score:.2f}",
    }
    return results, statistics

# Gradio Interface
with gr.Blocks(css=".gradio-container { max-width: 900px; margin: auto; padding: 20px; }") as demo:
    gr.Markdown("# Sentiment Analysis App")
    gr.Markdown(
        "This app predicts the sentiment of the input text on a scale from 1 to 5 using multiple models and provides detailed statistics."
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
            distilbert_output = gr.Textbox(label="Predicted Sentiment (DistilBERT)", interactive=False)
            log_reg_output = gr.Textbox(label="Predicted Sentiment (Logistic Regression)", interactive=False)
            bert_output = gr.Textbox(label="Predicted Sentiment (BERT Multilingual)", interactive=False)
        
        with gr.Column():
            statistics_output = gr.Textbox(label="Statistics (Lowest, Highest, Average)", interactive=False)

    # Button to analyze sentiment and show statistics
    def process_input_and_analyze(text_input):
        results, statistics = analyze_sentiment_and_statistics(text_input)
        return (
            f"{results['DistilBERT']}",
            f"{results['Logistic Regression']}",
            f"{results['BERT Multilingual (NLP Town)']}",
            f"Statistics:\n{statistics['Lowest Score']}\n{statistics['Highest Score']}\nAverage Score: {statistics['Average Score']}"
        )
    
    analyze_button.click(
        process_input_and_analyze,
        inputs=[text_input],
        outputs=[distilbert_output, log_reg_output, bert_output, statistics_output]
    )

# Launch the app
demo.launch()
