from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset")

# Extract data from the splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Count label occurrences for each split
train_label_counts = Counter(train_data['label'])
val_label_counts = Counter(val_data['label'])
test_label_counts = Counter(test_data['label'])

# Print label distributions
print("Label distribution in the training set:")
for label, count in train_label_counts.items():
    print(f"Label {label}: {count} samples")

print("\nLabel distribution in the validation set:")
for label, count in val_label_counts.items():
    print(f"Label {label}: {count} samples")

print("\nLabel distribution in the test set:")
for label, count in test_label_counts.items():
    print(f"Label {label}: {count} samples")

# Prepare data and labels
X_train, y_train = train_data['review'], train_data['label']
X_val, y_val = val_data['review'], val_data['label']
X_test, y_test = test_data['review'], test_data['label']

# Vectorize the text using TF-IDF with additional preprocessing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Function to preprocess text
def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove non-alphabetical characters (optional based on your dataset)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Apply preprocessing to the reviews
X_train = [preprocess_text(review) for review in X_train]
X_val = [preprocess_text(review) for review in X_val]
X_test = [preprocess_text(review) for review in X_test]

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')  # Use built-in English stopwords
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model with class weights (to handle potential class imbalance)
model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Use balanced class weights
model.fit(X_train_vec, y_train)

# Validate the model
val_predictions = model.predict(X_val_vec)
print("\nValidation Accuracy:", accuracy_score(y_val, val_predictions))
print("Validation Classification Report:")
print(classification_report(y_val, val_predictions))

# Test the model
test_predictions = model.predict(X_test_vec)
print("\nTest Accuracy:", accuracy_score(y_test, test_predictions))
print("Test Classification Report:")
print(classification_report(y_test, test_predictions))
