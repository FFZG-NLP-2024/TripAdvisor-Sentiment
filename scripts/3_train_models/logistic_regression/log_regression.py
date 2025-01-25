import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Use pickle for saving as .pkl
import os  # Import os to create a directory if it doesn't exist
from datasets import load_dataset

# Step 1: Load the dataset from HuggingFace
dataset = load_dataset('nhull/tripadvisor-split-dataset-v2')

# Step 2: Extract the train, validation, and test sets
train_data = pd.DataFrame(dataset['train'])
valid_data = pd.DataFrame(dataset['validation'])
test_data = pd.DataFrame(dataset['test'])

# Step 3: Preprocessing - Vectorize the reviews using TF-IDF with n-grams (unigrams, bigrams, and trigrams)
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))

# Apply TF-IDF vectorization
X_train_tfidf = vectorizer.fit_transform(train_data['review'])
X_valid_tfidf = vectorizer.transform(valid_data['review'])
X_test_tfidf = vectorizer.transform(test_data['review'])

# Labels
y_train = train_data['label']
y_valid = valid_data['label']
y_test = test_data['label']

# Step 4: Hyperparameter Tuning with GridSearchCV for Logistic Regression
param_grid = {'C': [0.1, 1, 10, 100]}  # Regularization strength for Logistic Regression

# Initialize Logistic Regression model
model_lr = LogisticRegression(max_iter=1000)

# Perform grid search with cross-validation
grid_search = GridSearchCV(model_lr, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)

# Get the best model from grid search
best_lr_model = grid_search.best_estimator_

# Step 5: Train the best Logistic Regression model and evaluate on the test set
y_pred_test_lr = best_lr_model.predict(X_test_tfidf)

# Step 6: Model Evaluation - Accuracy and Classification Report for Logistic Regression
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred_test_lr))
print("\nLogistic Regression Classification Report (Test):")
print(classification_report(y_test, y_pred_test_lr))

# Step 7: Confusion Matrix for error analysis (Logistic Regression)
cm_lr = confusion_matrix(y_test, y_pred_test_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Step 8: Cross-validation to evaluate Logistic Regression model generalization
cross_val_scores_lr = cross_val_score(best_lr_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("\nLogistic Regression Cross-validation scores:", cross_val_scores_lr)
print("Logistic Regression Mean Cross-validation score:", cross_val_scores_lr.mean())

# --- Save Logistic Regression Model and TF-IDF Vectorizer as .pkl ---

# Create the 'models/log_regression/' directory if it doesn't exist
save_dir = 'models/log_regression/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the Logistic Regression model as a .pkl file
with open(os.path.join(save_dir, 'logistic_regression_model.pkl'), 'wb') as model_file:
    pickle.dump(best_lr_model, model_file)
print("Logistic Regression model saved as 'models/log_regression/logistic_regression_model.pkl'")

# Save the TF-IDF Vectorizer as a .pkl file
with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
print("TF-IDF Vectorizer saved as 'models/log_regression/tfidf_vectorizer.pkl'")
