import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load the dataset
print("Loading dataset...")
df = pd.read_parquet("hf://datasets/jniimi/tripadvisor-review-rating/data/train-00000-of-00001.parquet")

# Inspect the first few rows of the dataframe
print("Inspecting the first few rows of the dataset:")
print(df.head())

# Check for missing values
print("Checking for missing values in the dataset:")
print(df.isnull().sum())

# Drop any rows with missing values in 'review' or 'overall'
print("Dropping rows with missing values in 'review' or 'overall' columns...")
df = df.dropna(subset=['review', 'overall'])

# Optional: Clean the text (remove non-alphanumeric characters, etc.)
print("Cleaning the review text (removing non-alphanumeric characters and converting to lowercase)...")
df['review'] = df['review'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Inspect the cleaned data
print("Inspecting the cleaned dataset:")
print(df.head())

# Features: the 'review' column
X = df['review']

# Target: the 'overall' column (ratings)
y = df['overall']

# Check the shape of the features and labels
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split data into 80% training and 20% testing
print("Splitting data into training and test sets (80% training, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the split data
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Initialize the TfidfVectorizer
print("Initializing the TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)  # Use only the top 5000 words

# Use tqdm to show progress while fitting and transforming the training data
print("Fitting the TF-IDF vectorizer to the training data...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Check the shape of the transformed data
print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

# Instantiate the Logistic Regression model
print("Initializing the Logistic Regression model...")
model = LogisticRegression(max_iter=1000, verbose=1)

# Train the model on the training data
# Manually adding a progress bar for training iterations
print("Training the Logistic Regression model on the training data...")
for i in tqdm(range(1), desc="Training Progress", ncols=100):  # Only 1 iteration for LogisticRegression
    model.fit(X_train_tfidf, y_train)

# Check the model’s coefficients (optional)
print(f"Model coefficients: {model.coef_}")

# Make predictions on the test set
print("Making predictions on the test data...")
y_pred = model.predict(X_test_tfidf)

# Evaluate the model using a classification report
print("Evaluating the model performance...")
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model and vectorizer
joblib.dump(model, 'scripts/modeli/logistic_regression/logistic_regression_model.pkl')
joblib.dump(vectorizer, 'scripts/modeli/logistic_regression/tfidf_vectorizer.pkl')
