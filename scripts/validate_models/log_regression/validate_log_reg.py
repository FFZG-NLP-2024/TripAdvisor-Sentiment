import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")
test_data = dataset['test']

# Download the vectorizer and model directly from HuggingFace
vectorizer_path = hf_hub_download(repo_id="nhull/logistic-regression-model", filename="tfidf_vectorizer.pkl")
model_path = hf_hub_download(repo_id="nhull/logistic-regression-model", filename="logistic_regression_model.pkl")

# Load the TF-IDF vectorizer and logistic regression model
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# Prepare the test data
texts = test_data['review']
true_labels = test_data['label']

# Transform the reviews using the TF-IDF vectorizer
X_test = vectorizer.transform(texts)

# Predict the labels
predicted_labels = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, digits=4)

# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(set(true_labels))), yticklabels=range(len(set(true_labels))))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("scripts/validate_models/log_regression/confusion_matrix_log_regression.png")
plt.show()

# Create a DataFrame for detailed output
output_data = pd.DataFrame({
    'review': texts,
    'true_label': true_labels,
    'predicted_label': predicted_labels,
    'difference': [t - p for t, p in zip(true_labels, predicted_labels)],
    'absolute_difference': [abs(t - p) for t, p in zip(true_labels, predicted_labels)]
})

# Save the DataFrame to a CSV file
output_data.to_csv("scripts/validate_models/log_regression/validation_results_log_regression.csv", index=False)