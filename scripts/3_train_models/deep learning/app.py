import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU and enforce CPU execution

import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load models
gru_model = load_model("best_GRU_tuning_model.h5")
lstm_model = load_model("LSTM_model.h5")
bilstm_model = load_model("BiLSTM_model.h5")

# Load tokenizer
with open("my_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text).strip()
    return text


def predict_with_gru(text):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = gru_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)


def predict_with_lstm(text):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = lstm_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)


def predict_with_bilstm(text):
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=200)
    probs = bilstm_model.predict(padded_seq)
    predicted_class = np.argmax(probs, axis=1)[0]
    return int(predicted_class + 1)


# Unified function for sentiment analysis and statistics
def analyze_sentiment_and_statistics(text):
    results = {
        "GRU Model": predict_with_gru(text),
        "LSTM Model": predict_with_lstm(text),
        "BiLSTM Model": predict_with_bilstm(text),
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
    gr.Markdown("# RNN Sentiment Analysis")
    gr.Markdown(
        "Predict the sentiment of your text review using RNN-based models."
    )

    # Text input box
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter your text here:",
            lines=3,
            placeholder="Type your review here..."
        )

    # Prediction and statistics boxes
    with gr.Row():
        with gr.Column():
            gru_output = gr.Textbox(label="Predicted Sentiment (GRU Model)", interactive=False)
            lstm_output = gr.Textbox(label="Predicted Sentiment (LSTM Model)", interactive=False)
            bilstm_output = gr.Textbox(label="Predicted Sentiment (BiLSTM Model)", interactive=False)

        with gr.Column():
            statistics_output = gr.Textbox(label="Statistics (Lowest, Highest, Average)", interactive=False)

    # Buttons placed together in a row (second line)
    with gr.Row():
        analyze_button = gr.Button("Analyze Sentiment", variant="primary")  # Blue button
        clear_button = gr.ClearButton(
            [text_input, gru_output, lstm_output, bilstm_output, statistics_output])  # Clear button


    # Button to analyze sentiment and show statistics
    def process_input_and_analyze(text_input):
        results, statistics = analyze_sentiment_and_statistics(text_input)
        if "Message" in statistics:
            return (
                f"{results['GRU Model']}",
                f"{results['LSTM Model']}",
                f"{results['BiLSTM Model']}",
                f"Statistics:\n{statistics['Message']}\nAverage Score: {statistics['Average Score']}"
            )
        else:
            return (
                f"{results['GRU Model']}",
                f"{results['LSTM Model']}",
                f"{results['BiLSTM Model']}",
                f"Statistics:\n{statistics['Lowest Score']}\n{statistics['Highest Score']}\nAverage Score: {statistics['Average Score']}"
            )


    analyze_button.click(
        process_input_and_analyze,
        inputs=[text_input],
        outputs=[
            gru_output,
            lstm_output,
            bilstm_output,
            statistics_output
        ]
    )

demo.launch()