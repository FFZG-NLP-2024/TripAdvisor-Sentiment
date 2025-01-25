from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load the trained model and tokenizer
model_path = "models/distilbert/best_trained_model"  # Path to the trained model directory
distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)
distilbert_model.eval()  # Set the model to evaluation mode


# Preprocess the input text (ensure consistency with training)
def preprocess_text(text):
    return text.strip().lower()


# Predict sentiment using the DistilBERT model
def predict_sentiment(text):
    # Preprocess the input text
    text = preprocess_text(text)
    
    # Tokenize the text
    encodings = distilbert_tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=512,  # Ensure this matches training
        return_tensors="pt"
    ).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = distilbert_model(**encodings)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted_label = logits.argmax(axis=-1).cpu().item()  # Get the predicted label (0-4)

    # Debugging: Log intermediate outputs
    print(f"Logits: {logits}")
    print(f"Softmax Probabilities: {probabilities}")
    print(f"Predicted Label (0-4 scale): {predicted_label}")
    
    # Convert to 1-5 scale
    return predicted_label + 1


# Test the inference with a sample text
if __name__ == "__main__":
    # Example review text (you can replace this with any text)
    input_text = (
        "Small room and no noise insulation. "
        "We just spent 4 nights in this hotel. It was the first hotel that we have stayed in the USA. "
        "We booked it based on tripadvisor reviews. Overall the experience was just acceptable compared to previous "
        "hotels which we had stayed in. "
        "Pros: Efficient check-in and out, polite staff, good central location, comfortable bed. "
        "Cons: Small rooms, no noise insulation, very slow draining bath."
    )
    
    # Predict sentiment
    predicted_sentiment = predict_sentiment(input_text)
    
    # Display the result
    print(f"Input Text:\n{input_text}\n")
    print(f"Predicted Sentiment (1-5 scale): {predicted_sentiment}")
