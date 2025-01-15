import gradio as gr
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset and prepare the model (as you already did)
dataset = load_dataset("nhull/tripadvisor-split-dataset")
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Prepare data and labels
X_train, y_train = train_data['review'], train_data['label']
X_val, y_val = val_data['review'], val_data['label']
X_test, y_test = test_data['review'], test_data['label']

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Define the prediction function
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return f"Predicted label: {prediction}"

# Create the Gradio interface
iface = gr.Interface(fn=predict_sentiment,
                     inputs=gr.Textbox(label="Enter a review text", placeholder="Type your review here..."),
                     outputs=gr.Textbox(label="Predicted label"),
                     live=True)

# Launch the interface
iface.launch()
