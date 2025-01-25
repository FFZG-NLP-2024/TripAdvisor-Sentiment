import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from datasets import load_dataset

# Load test dataset
print("Loading dataset...")
dataset = load_dataset("nhull/tripadvisor-split-dataset-v2")

test_data = dataset['test']

# Extract test data
X_test = [x['review'] for x in test_data]
y_test = [x['label'] for x in test_data]

# Adjust labels from 0-4 to 1-5
y_test = [label + 1 for label in y_test]

# Load the saved TF-IDF vectorizer
print("Loading TF-IDF vectorizer...")
with open("models/log_regression/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Transform test data using TF-IDF vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Load the saved logistic regression model
print("Loading logistic regression model...")
with open("models/log_regression/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict on the test data
print("Predicting test data...")
y_pred = model.predict(X_test_tfidf)

# Adjust predictions from 0-4 to 1-5
y_pred = [pred + 1 for pred in y_pred]

# Calculate metrics
print("Calculating metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred)

# Generate confusion matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Logistic Regression')
plt.savefig("scripts/validate_models/confusion_matrix_log_regression.png")
plt.show()

# Create a DataFrame with true labels, predicted labels, their differences, and original examples
print("Creating analysis DataFrame...")
data = {
    "Review": X_test,
    "True Label": y_test,
    "Predicted Label": y_pred,
    "Difference": [pred - true for pred, true in zip(y_pred, y_test)],
    "Absolute Difference": [abs(pred - true) for pred, true in zip(y_pred, y_test)]
}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("scripts/validate_models/validation_results_log_regression.csv", index=False)

# Analyze the prediction bias
average_difference = df["Difference"].mean()
average_absolute_difference = df["Absolute Difference"].mean()
if average_difference > 0:
    bias_message = f"Model predicts on average too high by {average_difference:.2f}."
elif average_difference < 0:
    bias_message = f"Model predicts on average too low by {abs(average_difference):.2f}."
else:
    bias_message = "Model predictions are on average perfectly balanced."

bias_message += f" The average absolute difference is {average_absolute_difference:.2f}."

# Print the results
print("\nTest Accuracy:", accuracy)
print("\nPrecision:", precision)
print("\nRecall:", recall)
print("\nF1 Score:", f1)
print("\nClassification Report:\n", report)
print("\n", bias_message)

# Optionally, save the metrics and bias message to a file
with open("logs/metrics_log.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"\nClassification Report:\n{report}\n")
    f.write(f"\n{bias_message}\n")

print("Evaluation completed.")
