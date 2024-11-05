import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Load your CSV file
file_path = "data/test_25_sample_reviews.csv"  # Update this path
data = pd.read_csv(file_path)

# Define mapping for sentiment labels to integer ratings
sentiment_mapping = {
    "1 (negative)": 1,
    "2 (mostly negative)": 2,
    "3 (neutral)": 3,
    "4 (mostly positive)": 4,
    "5 (positive)": 5
}

# Apply the mapping to each sentiment column
sentiment_columns = ["Sentiment 1", "Sentiment 2", "Sentiment 3", "Sentiment 4", "Sentiment 5"]
data[sentiment_columns] = data[sentiment_columns].replace(sentiment_mapping)

# Initialize a matrix to count occurrences of each rating per review
ratings_counts = np.zeros((len(data), 5), dtype=int)

# Count ratings for each review
for idx, row in data[sentiment_columns].iterrows():
    ratings = row.dropna().astype(int)  # Drop empty cells and convert to integers
    for rating in ratings:
        ratings_counts[idx, rating - 1] += 1  # Adjust for zero-indexed array

# Calculate Fleiss' Kappa
kappa = fleiss_kappa(ratings_counts, method='fleiss')
print(f"Fleiss' Kappa: {kappa}")


from sklearn.metrics import cohen_kappa_score

# Assuming your two annotators' ratings are in columns "Sentiment 1" and "Sentiment 2"
annotator1 = data["Sentiment 1"].dropna().astype(int)
annotator2 = data["Sentiment 2"].dropna().astype(int)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa}")